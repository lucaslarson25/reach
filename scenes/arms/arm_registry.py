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


# MuJoCo Menagerie arms: https://mujoco.readthedocs.io/en/stable/models.html (Arms)
ARM_REGISTRY: dict[str, ArmConfig] = {
    "arm_2link": ArmConfig(
        name="2-link arm (demo)",
        arm_path="arms/arm_2link/arm.xml",
        ee_site_name="ee_site",
        reach_min=0.05,
        reach_max=0.38,
        home_keyframe_name="home",
    ),
    "panda": ArmConfig(
        name="Franka Emika Panda",
        arm_path="arms/panda/panda_nohand.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.85,
        home_keyframe_name="home",
    ),
    "fr3": ArmConfig(
        name="Franka FR3",
        arm_path="arms/fr3/fr3.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.9,
        home_keyframe_name="home",
    ),
    "ur5e": ArmConfig(
        name="Universal Robots UR5e",
        arm_path="arms/ur5e/ur5e.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.85,
        home_keyframe_name="home",
    ),
    "ur10e": ArmConfig(
        name="Universal Robots UR10e",
        arm_path="arms/ur10e/ur10e.xml",
        ee_site_name="attachment_site",
        reach_min=0.2,
        reach_max=1.3,
        home_keyframe_name="home",
    ),
    "iiwa14": ArmConfig(
        name="KUKA LBR iiwa14",
        arm_path="arms/iiwa14/iiwa14.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.85,
        home_keyframe_name="home",
    ),
    "xarm7": ArmConfig(
        name="UFACTORY xArm7",
        arm_path="arms/xarm7/xarm7_nohand.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.75,
        home_keyframe_name="home",
    ),
    "sawyer": ArmConfig(
        name="Rethink Robotics Sawyer",
        arm_path="arms/sawyer/sawyer.xml",
        ee_site_name="attachment_site",
        reach_min=0.15,
        reach_max=0.84,
        home_keyframe_name="home",
    ),
    "lite6": ArmConfig(
        name="UFactory Lite 6",
        arm_path="arms/lite6/lite6.xml",
        ee_site_name="attachment_site",
        reach_min=0.1,
        reach_max=0.55,
        home_keyframe_name="home",
    ),
    "vx300s": ArmConfig(
        name="Trossen ViperX 300 6DOF",
        arm_path="arms/vx300s/vx300s.xml",
        ee_site_name="pinch",
        reach_min=0.1,
        reach_max=0.5,
        home_keyframe_name="home",
    ),
    "wx250s": ArmConfig(
        name="Trossen WidowX 250 6DOF",
        arm_path="arms/wx250s/wx250s.xml",
        ee_site_name="ee",
        reach_min=0.1,
        reach_max=0.45,
        home_keyframe_name="home",
    ),
    "aloha": ArmConfig(
        name="ALOHA 2 (dual arm, right EE)",
        arm_path="arms/aloha/aloha.xml",
        ee_site_name="right/gripper",
        reach_min=0.2,
        reach_max=0.6,
        home_keyframe_name="home",
    ),
    "unitree_z1": ArmConfig(
        name="Unitree Z1",
        arm_path="arms/unitree_z1/z1.xml",
        ee_site_name="eetip",
        reach_min=0.12,
        reach_max=0.55,
        home_keyframe_name="home",
    ),
    # Legacy id: point to same arm as unitree_z1
    "z1": ArmConfig(
        name="Unitree Z1 (alias)",
        arm_path="arms/unitree_z1/z1.xml",
        ee_site_name="eetip",
        reach_min=0.12,
        reach_max=0.55,
        home_keyframe_name="home",
        scene_path_fallback="scenes/industrial_arm_reaching/models/z1scene.xml",
    ),
}


def get_arm_config(arm_id: str | None) -> ArmConfig | None:
    if not arm_id:
        return None
    return ARM_REGISTRY.get(arm_id)


def get_arm_config_from_discovery(arm_id: str | None) -> dict[str, Any] | None:
    """Auto-discover arm from models/arms/<arm_id>/ (drop MJCF into folder)."""
    if not arm_id:
        return None
    from scenes.arms.arm_discovery import discover_all_arms
    discovered = discover_all_arms()
    return discovered.get(arm_id)


def get_arm_info(arm_id: str | None) -> dict[str, Any] | None:
    """
    Unified arm info: discovery (preferred) or registry.
    Returns dict: arm_path, ee_sites (list), reach_min, reach_max, actuator_groups, n_act, etc.
    For multi-arm (e.g. ALOHA), ee_sites = [left/gripper, right/gripper].
    """
    if not arm_id:
        return None
    discovered = get_arm_config_from_discovery(arm_id)
    if discovered:
        return discovered
    cfg = get_arm_config(arm_id)
    if not cfg:
        return None
    return {
        "arm_id": arm_id,
        "arm_path": cfg.arm_path,
        "ee_sites": [cfg.ee_site_name] if cfg.ee_site_name else [],
        "actuator_groups": None,  # unknown for registry entries
        "reach_min": cfg.reach_min,
        "reach_max": cfg.reach_max or 0.5,
        "n_act": None,
        "n_q": None,
        "home_keyframe_name": cfg.home_keyframe_name or "home",
    }


def get_registered_arm_ids() -> list[str]:
    return list(ARM_REGISTRY.keys())


def resolve_model_path(
    arm_id: str | None, model_path: str | None, ball_count: int = 1
) -> str:
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
    disc = get_arm_config_from_discovery(aid) if not cfg else None
    if disc:
        models_dir = get_models_dir()
        arm_full = models_dir / disc["arm_path"]
        if arm_full.exists():
            return compose_scene(disc["arm_path"], aid, ball_count)
    if not cfg:
        # Fallback: try arm_id as path segment (e.g. models/arms/<aid>/)
        fallback = get_models_dir() / "arms" / aid
        if fallback.is_dir():
            for name in ("arm.xml", f"{aid}.xml", "scene.xml"):
                candidate = fallback / name
                if candidate.exists():
                    rel = f"arms/{aid}/{name}"
                    return compose_scene(rel, aid, ball_count)
        raise FileNotFoundError(f"Unknown arm_id={aid} and no model_path given.")

    models_dir = get_models_dir()
    arm_full = models_dir / cfg.arm_path
    if arm_full.exists():
        return compose_scene(cfg.arm_path, aid, ball_count)
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


def compute_reach_min_from_model(
    model: Any, data: Any, ee_site_id: int, n_samples: int = 300
) -> float:
    """
    Compute minimum horizontal reach (inner workspace boundary) by sampling joint configs.
    The EE cannot reach points closer than this—use as reach_min so the ball is always
    reachable. Uses 10th percentile to avoid singularities / degenerate configs.
    """
    import numpy as np
    import mujoco

    np.random.seed(43)
    distances = []
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
        distances.append(r)
    if not distances:
        return 0.15
    # 10th percentile: conservative inner boundary, avoids singularities
    p10 = float(np.percentile(distances, 10))
    return max(0.05, min(p10, 0.5))
