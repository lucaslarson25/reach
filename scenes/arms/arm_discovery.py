"""
Arm auto-discovery: scan models/arms/ and infer DOF, reach, EE sites from MJCF.
For multi-arm models (e.g. ALOHA), detects multiple EE sites and actuator groups.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco


# Site names (or substrings) that indicate an end-effector
EE_SITE_PATTERNS = (
    "gripper", "eetip", "hand", "attachment", "pin_site", "tool0", "ee", "ee_site",
    "end_effector",
)
# Exclude: finger sites are gripper sub-parts, not main EE (e.g. left/left_finger)
EE_SITE_EXCLUDE = ("left_finger", "right_finger", "/finger")


def get_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def discover_arm_dirs() -> list[tuple[str, Path]]:
    """
    Scan models/arms/ for arm directories.
    Returns list of (arm_id, dir_path). arm_id = dir name.
    """
    models_dir = get_models_dir()
    arms_root = models_dir / "arms"
    if not arms_root.is_dir():
        return []
    result = []
    for d in sorted(arms_root.iterdir()):
        if not d.is_dir():
            continue
        arm_id = d.name
        for name in ("arm.xml", f"{arm_id}.xml", "scene.xml"):
            candidate = d / name
            if candidate.exists():
                result.append((arm_id, d))
                break
    return result


def find_arm_xml(arm_dir: Path, arm_id: str) -> Path | None:
    """Return path to arm XML in arm_dir."""
    for name in (f"{arm_id}.xml", "arm.xml", "scene.xml"):
        p = arm_dir / name
        if p.exists():
            return p
    return None


def _compose_minimal_scene_for_load(arm_path: Path, models_dir: Path) -> str:
    """Create minimal composed XML (floor + ball + arm) so MuJoCo can load and resolve mesh paths."""
    rel = arm_path.relative_to(models_dir)
    depth = len(rel.parent.parts)
    prefix = "/".join([".."] * depth) if depth else "."
    arm_basename = arm_path.name
    content = f'''<mujoco model="discover">
  <include file="{prefix}/floor.xml"/>
  <include file="{prefix}/ball.xml"/>
  <include file="{arm_basename}"/>
</mujoco>
'''
    composed = arm_path.parent / "_discover_temp.xml"
    composed.write_text(content)
    return str(composed.resolve())


def _is_ee_site(name: str) -> bool:
    """Return True if site name looks like an end-effector (excludes finger sub-sites)."""
    lower = name.lower()
    if any(ex in lower for ex in EE_SITE_EXCLUDE):
        return False
    return any(p in lower for p in EE_SITE_PATTERNS)


def _group_sites_by_arm(site_names: list[str]) -> list[list[str]]:
    """
    Group EE site names by arm prefix (left/, right/, arm0/, etc.).
    Single-arm: returns [[site]]. Multi-arm: returns [[left/gripper], [right/gripper]].
    """
    ee_sites = [n for n in site_names if _is_ee_site(n)]
    if not ee_sites:
        return []
    if len(ee_sites) == 1:
        return [ee_sites]
    # Multi-arm: group by prefix (e.g. left/, right/)
    groups: dict[str, list[str]] = {}
    for s in ee_sites:
        if "/" in s:
            prefix = s.split("/")[0] + "/"
            groups.setdefault(prefix, []).append(s)
        else:
            groups.setdefault("", []).append(s)
    if len(groups) == 1 and "" in groups and len(groups[""]) > 1:
        return [[s] for s in groups[""]]
    return list(groups.values())


def infer_ee_sites(model: Any) -> list[str]:
    """
    Infer EE site names from model.
    Returns flat list: single arm -> [site], multi-arm -> [left/gripper, right/gripper].
    """
    site_names = []
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and _is_ee_site(name):
            site_names.append(name)
    return sorted(site_names)


def infer_actuator_groups(model: Any, ee_site_names: list[str]) -> list[list[int]]:
    """
    Map actuators to arms by name prefix.
    ALOHA: actuators named left/waist, right/waist etc. -> [left_indices], [right_indices]
    Single arm: returns [list(range(nu))].
    """
    nu = model.nu
    if nu == 0:
        return []
    if len(ee_site_names) <= 1:
        return [list(range(nu))]
    prefixes = []
    for name in ee_site_names:
        if "/" in name:
            prefixes.append(name.split("/")[0] + "/")
        else:
            prefixes.append("")
    groups: dict[int, list[int]] = {i: [] for i in range(len(prefixes))}
    for a in range(nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        assigned = False
        for i, pre in enumerate(prefixes):
            if pre and aname.startswith(pre):
                groups[i].append(a)
                assigned = True
                break
        if not assigned and len(prefixes) == 1 and prefixes[0] == "":
            groups[0].append(a)
        elif not assigned and len(prefixes) > 1:
            # fallback: assign by order (first half to first arm, etc.)
            g = min(a * len(prefixes) // nu, len(prefixes) - 1)
            groups[g].append(a)
    return [groups[i] for i in range(len(prefixes))]


def compute_reach_from_model(
    model: Any, data: Any, ee_site_id: int, n_samples: int = 200
) -> float:
    """Compute max horizontal reach for one EE site."""
    import numpy as np
    np.random.seed(42)
    data.qpos[:] = model.qpos0
    max_reach = 0.0
    for _ in range(n_samples):
        qpos = model.qpos0.copy()
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                if model.jnt_limited[j]:
                    qpos[adr] = np.random.uniform(model.jnt_range[j, 0], model.jnt_range[j, 1])
                else:
                    qpos[adr] = np.random.uniform(-np.pi, np.pi)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[ee_site_id]
        r = float((pos[0] ** 2 + pos[1] ** 2) ** 0.5)
        if r > max_reach:
            max_reach = r
    return max(0.15, min(max_reach, 2.0))


def discover_arm_config(arm_id: str, arm_dir: Path) -> dict[str, Any] | None:
    """
    Load arm MJCF and infer config. Returns dict:
      arm_id, arm_path (rel to models/), ee_sites (list), actuator_groups, reach_min, reach_max (per EE),
      n_act, home_keyframe_name
    """
    models_dir = get_models_dir()
    arm_xml = find_arm_xml(arm_dir, arm_id)
    if not arm_xml:
        return None
    arm_rel = arm_xml.relative_to(models_dir)
    composed_path = _compose_minimal_scene_for_load(arm_xml, models_dir)
    try:
        model = mujoco.MjModel.from_xml_path(composed_path)
        data = mujoco.MjData(model)
    except Exception:
        return None
    finally:
        temp = Path(composed_path)
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    ee_sites = infer_ee_sites(model)
    if not ee_sites:
        return None
    actuator_groups = infer_actuator_groups(model, ee_sites)
    reach_max_list = []
    for site_name in ee_sites:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid >= 0:
            r = compute_reach_from_model(model, data, sid)
            reach_max_list.append(r)
        else:
            reach_max_list.append(0.5)
    reach_max = max(reach_max_list) if reach_max_list else 0.5
    reach_min = max(0.08, 0.45 * reach_max)
    return {
        "arm_id": arm_id,
        "arm_path": str(arm_rel).replace("\\", "/"),
        "ee_sites": ee_sites,
        "actuator_groups": actuator_groups,
        "reach_min": reach_min,
        "reach_max": reach_max,
        "n_act": model.nu,
        "n_q": model.nq,
        "home_keyframe_name": "home",
    }


def discover_all_arms() -> dict[str, dict[str, Any]]:
    """Scan models/arms/ and return {arm_id: config} for each discovered arm."""
    result = {}
    for arm_id, arm_dir in discover_arm_dirs():
        cfg = discover_arm_config(arm_id, arm_dir)
        if cfg:
            result[arm_id] = cfg
    return result
