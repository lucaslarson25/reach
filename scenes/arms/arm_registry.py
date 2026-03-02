"""
Arm registry: arm-only upload. Each entry points to an arm XML; the scene
(floor + ball + arm) is composed at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenes.arms.scene_compose import compose_scene, get_models_dir

# End-effector site names to try when not set in registry
DEFAULT_EE_SITE_CANDIDATES = (
    "eetip", "hand", "attachment", "pin_site", "tool0", "ee", "ee_site",
    "end_effector", "gripper", "finger",
)


@dataclass
class ArmConfig:
    """Arm-only config: path to arm XML, EE site, reach, optional fallback scene."""
    name: str
    arm_path: str  # relative to scenes/arms/models/, e.g. "arms/z1/z1_arm.xml"
    ee_site_name: str | None = None
    reach_min: float = 0.08
    reach_max: float | None = None
    home_keyframe_name: str = "home"
    # If arm_path doesn't exist (e.g. arm not yet copied), load this scene instead
    scene_path_fallback: str | None = None  # relative to project root


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ARM_REGISTRY: dict[str, ArmConfig] = {
    "z1": ArmConfig(
        name="Unitree Z1",
        arm_path="arms/z1/z1_arm.xml",
        ee_site_name="eetip",
        reach_min=0.12,
        reach_max=0.55,
        home_keyframe_name="home",
        scene_path_fallback="scenes/industrial_arm_reaching/models/z1scene.xml",
    ),
    "arm_2link": ArmConfig(
        name="2-link arm",
        arm_path="arms/arm_2link/arm.xml",
        ee_site_name="ee_site",
        reach_min=0.05,
        reach_max=0.38,
        home_keyframe_name="home",
    ),
}


def get_arm_config(arm_id: str | None) -> ArmConfig | None:
    if not arm_id:
        return None
    return ARM_REGISTRY.get(arm_id)


def get_registered_arm_ids() -> list[str]:
    return list(ARM_REGISTRY.keys())


def resolve_model_path(arm_id: str | None, model_path: str | None) -> str:
    """
    Resolve path to a loadable scene XML.
    - If model_path is set, return it (resolved to absolute).
    - Else if arm_id is set: if arm XML exists under arms/models/, compose scene and return;
      else if config has scene_path_fallback, return that.
    - Else use arm_id "z1" and resolve again.
    """
    root = _project_root()
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = (root / model_path).resolve()
        return str(p)

    aid = arm_id or "z1"
    cfg = get_arm_config(aid)
    if not cfg:
        # Unknown arm_id; try composing with arm_id as path segment
        fallback = get_models_dir() / "arms" / aid
        if fallback.is_dir():
            for name in ("arm.xml", f"{aid}.xml", "scene.xml"):
                candidate = fallback / name
                if candidate.exists():
                    rel = f"arms/{aid}/{name}"
                    return compose_scene(rel, aid)
        raise FileNotFoundError(f"Unknown arm_id={aid} and no model_path given.")

    models_dir = get_models_dir()
    arm_full = models_dir / cfg.arm_path
    if arm_full.exists():
        return compose_scene(cfg.arm_path, aid)
    if cfg.scene_path_fallback:
        return str((root / cfg.scene_path_fallback).resolve())
    raise FileNotFoundError(f"Arm XML not found: {arm_full}. Add it or set scene_path_fallback.")


def resolve_ee_site_name(model: Any, arm_id: str | None, ee_site_name: str | None) -> str:
    import mujoco
    def has_site(name: str) -> bool:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    if ee_site_name and has_site(ee_site_name):
        return ee_site_name
    cfg = get_arm_config(arm_id) if arm_id else None
    if cfg and cfg.ee_site_name and has_site(cfg.ee_site_name):
        return cfg.ee_site_name
    for c in DEFAULT_EE_SITE_CANDIDATES:
        if has_site(c):
            return c
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise KeyError("No site found for end-effector. Add a <site> at the arm tip or set ee_site_name.")


def compute_reach_from_model(model: Any, data: Any, ee_site_id: int, n_samples: int = 200) -> float:
    import numpy as np
    import mujoco
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
        r = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
        if r > max_reach:
            max_reach = r
    return max(0.15, min(max_reach, 2.0))
