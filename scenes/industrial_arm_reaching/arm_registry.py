"""
Arm registry for multi-arm reach training.

Supports arms from MuJoCo Menagerie (Arms section):
https://mujoco.readthedocs.io/en/stable/models.html

Each arm has a scene XML (floor + ball + arm), an end-effector site name,
and optional reach bounds. Ball drop is sampled within reach so training
adapts to arm length and DOF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default end-effector site names to try (in order) when not specified in registry.
# Menagerie arms use various names: eetip (Z1), hand (Panda), attachment, etc.
DEFAULT_EE_SITE_CANDIDATES = (
    "eetip",
    "hand",
    "attachment",
    "pin_site",
    "tool0",
    "ee",
    "ee_site",
    "end_effector",
    "gripper",
    "finger",
)


@dataclass
class ArmConfig:
    """Configuration for one arm: scene path, EE site, and reach (for ball sampling)."""
    name: str
    scene_path: str
    ee_site_name: str | None = None
    reach_min: float = 0.08
    reach_max: float | None = None
    home_keyframe_name: str = "home"


def _models_dir() -> Path:
    base = Path(__file__).resolve().parent
    return base / "models"


def _scene_path(rel: str) -> str:
    return str(_models_dir() / rel)


# Registry: arm_id -> ArmConfig
# scene_path is relative to scenes/industrial_arm_reaching/models/
# Add Menagerie arms here after placing their scene XML in models/ (see README).
ARM_REGISTRY: dict[str, ArmConfig] = {
    "z1": ArmConfig(
        name="Unitree Z1",
        scene_path=_scene_path("z1scene.xml"),
        ee_site_name="eetip",
        reach_min=0.12,
        reach_max=0.55,
        home_keyframe_name="home",
    ),
    "arm_2link": ArmConfig(
        name="2-link arm (demo)",
        scene_path=_scene_path("arm_scene.xml"),
        ee_site_name="ee_site",
        reach_min=0.05,
        reach_max=0.38,
        home_keyframe_name="home",
    ),
}


def get_arm_config(arm_id: str | None) -> ArmConfig | None:
    """Return ArmConfig for arm_id, or None if not found."""
    if not arm_id:
        return None
    return ARM_REGISTRY.get(arm_id)


def get_registered_arm_ids() -> list[str]:
    """Return list of registered arm IDs (for CLI/config)."""
    return list(ARM_REGISTRY.keys())


def resolve_model_path(arm_id: str | None, model_path: str | None) -> str:
    """
    Resolve the MuJoCo scene XML path.
    - If model_path is set, use it (must exist).
    - Else if arm_id is set and in registry, use that arm's scene_path.
    - Else use default z1 scene.
    """
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = (project_root / model_path).resolve()
        return str(path)
    cfg = get_arm_config(arm_id) if arm_id else None
    if cfg:
        return cfg.scene_path
    return _scene_path("z1scene.xml")


def resolve_ee_site_name(model: Any, arm_id: str | None, ee_site_name: str | None) -> str:
    """
    Resolve end-effector site name: use explicit name if given and present,
    else use arm config, else try DEFAULT_EE_SITE_CANDIDATES.
    Raises KeyError if no site found.
    """
    import mujoco

    def has_site(name: str) -> bool:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return sid >= 0

    if ee_site_name and has_site(ee_site_name):
        return ee_site_name
    cfg = get_arm_config(arm_id) if arm_id else None
    if cfg and cfg.ee_site_name and has_site(cfg.ee_site_name):
        return cfg.ee_site_name
    for candidate in DEFAULT_EE_SITE_CANDIDATES:
        if has_site(candidate):
            return candidate
    # Last resort: first site in model (often the only one for simple arms)
    if model.nsite > 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0)
    raise KeyError("No site found in model for end-effector. Add a <site> at the arm tip or set ee_site_name.")


def compute_reach_from_model(model: Any, data: Any, ee_site_id: int, n_samples: int = 200) -> float:
    """
    Estimate max reach (meters) by sampling random joint configs within limits
    and taking the max horizontal distance of the end-effector from the origin.
    Used when reach_max is not in the registry.
    """
    import numpy as np
    import mujoco

    np.random.seed(42)
    data.qpos[:] = model.qpos0
    nq = model.nq
    max_reach = 0.0
    for _ in range(n_samples):
        qpos = model.qpos0.copy()
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE or model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
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
