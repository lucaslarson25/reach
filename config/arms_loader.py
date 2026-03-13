# config/arms_loader.py – load arms config from YAML

import os
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ARM_OVERRIDES_PATH = os.path.join(REPO_ROOT, "config", "arm_overrides.yaml")


def load_arm_overrides(arm_id: str) -> dict:
    """Load per-arm overrides from config/arm_overrides.yaml. Returns {} if arm not listed."""
    if not arm_id or not os.path.isfile(ARM_OVERRIDES_PATH):
        return {}
    with open(ARM_OVERRIDES_PATH, "r") as f:
        raw = yaml.safe_load(f) or {}
    overrides = raw.get(arm_id, {})
    return dict(overrides) if isinstance(overrides, dict) else {}


def apply_arm_overrides(cfg: dict, arm_id: str) -> dict:
    """Merge arm_overrides for arm_id into cfg. Overrides go into scene/train as appropriate."""
    ov = load_arm_overrides(arm_id)
    if not ov:
        return cfg
    # Map override keys to scene vs train
    scene_keys = {"model_path", "ball_mode", "per_arm_policies", "initial_pose", "initial_keyframe"}
    train_keys = {"reach_min_mode", "reach_min_fraction", "reach_min_floor", "reach_max_cap",
                  "ee_priority_scale", "joint_limit_margin_penalty"}
    for k, v in ov.items():
        if k in scene_keys:
            cfg["scene"][k] = v
        elif k in train_keys:
            cfg["train"][k] = v
    return cfg


def load_arms_config(yaml_path: str) -> dict:
    """Load arms.yaml; paths stay relative to repo root (caller can resolve)."""
    path = yaml_path if os.path.isabs(yaml_path) else os.path.join(REPO_ROOT, yaml_path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _normalize(raw)


def _normalize(raw: dict) -> dict:
    cfg = dict(raw)
    cfg.setdefault("scene", {})
    cfg.setdefault("train", {})
    cfg.setdefault("run", {})
    cfg["scene"].setdefault("arm_id", "panda")
    cfg["scene"].setdefault("model_path", None)
    cfg["scene"].setdefault("ball_mode", "shared")
    cfg["scene"].setdefault("per_arm_policies", False)
    cfg["train"].setdefault("total_steps", 300000)
    cfg["train"].setdefault("policy_dir", "policies")
    cfg["train"].setdefault("n_steps", 2048)
    cfg["train"].setdefault("batch_size", 128)
    cfg["train"].setdefault("n_epochs", 10)
    cfg["train"].setdefault("learning_rate", 0.0003)
    cfg["train"].setdefault("seed", 42)
    cfg["train"].setdefault("device", None)
    cfg["train"].setdefault("use_mps", False)
    cfg["train"].setdefault("reward_time_penalty", 0.0005)
    cfg["train"].setdefault("reward_smoothness", 0.02)
    cfg["train"].setdefault("reward_move_away_penalty", 0.5)
    cfg["run"].setdefault("policy_path", None)
    cfg["run"].setdefault("steps", 5000)
    cfg["run"].setdefault("deterministic", True)
    cfg["run"].setdefault("debug", False)
    cfg["run"].setdefault("stochastic", False)
    return cfg


def resolve_policy_paths(
    cfg: dict, arm_id_override: str | None = None, n_arms: int = 1
) -> str | list[str]:
    """
    Resolve policy path(s). If per_arm_policies and n_arms>1, returns list of N paths.
    Else returns single path.
    """
    """Resolve run.policy_path; if null, derive from arm_id and train.total_steps.
    arm_id_override: use this arm_id when deriving (e.g. from --arm-id); else use scene.arm_id from config.
    """
    path = cfg["run"].get("policy_path")
    if path:
        p = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
        return p
    arm_id = (arm_id_override or cfg["scene"].get("arm_id")) or "custom"
    total = int(cfg["train"].get("total_steps", 300000))
    k = total // 1000
    policy_dir = cfg["train"].get("policy_dir") or "policies"
    policy_dir = policy_dir if os.path.isabs(policy_dir) else os.path.join(REPO_ROOT, policy_dir)
    if cfg["scene"].get("per_arm_policies") and n_arms > 1:
        return [
            os.path.join(policy_dir, f"ppo_arms_{arm_id}_arm{i}_mac_{k}k.zip")
            for i in range(n_arms)
        ]
    return os.path.join(policy_dir, f"ppo_arms_{arm_id}_mac_{k}k.zip")


def resolve_policy_path(cfg: dict, arm_id_override: str | None = None) -> str:
    """Single policy path (legacy). For per_arm_policies use resolve_policy_paths."""
    out = resolve_policy_paths(cfg, arm_id_override, n_arms=1)
    return out[0] if isinstance(out, list) else out
