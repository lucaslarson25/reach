# config/render_loader.py

import os
import yaml
import importlib

# Anchor relative paths at the repo root (one level up from /config)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_path(p: str | None) -> str | None:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(REPO_ROOT, p))

def _normalize_schema(raw: dict) -> dict:
    """
    Normalize legacy keys to the new schema:
      Required (after normalization):
        scene: { env_class: "package.module:ClassName", model_xml: "path/to.xml" }
        policy: { path: "path/to/model.zip" }
      Optional:
        run: { episodes, max_seconds_per_ep, deterministic }
        viewer: { mode, fps_limit }
    """
    cfg = dict(raw)

    # Legacy top-level fallbacks -> wrap into blocks
    if "scene" not in cfg:
        scene = {}
        if "env_class" in cfg:
            scene["env_class"] = cfg.pop("env_class")
        if "model_xml" in cfg:
            scene["model_xml"] = cfg.pop("model_xml")
        if scene:
            cfg["scene"] = scene

    if "policy" not in cfg:
        policy = {}
        if "policy_path" in cfg:
            policy["path"] = cfg.pop("policy_path")
        if policy:
            cfg["policy"] = policy

    if "run" not in cfg:
        run = {}
        if "episodes" in cfg:
            run["episodes"] = cfg.pop("episodes")
        if "max_seconds_per_ep" in cfg:
            run["max_seconds_per_ep"] = cfg.pop("max_seconds_per_ep")
        if "deterministic" in cfg:
            run["deterministic"] = cfg.pop("deterministic")
        if run:
            cfg["run"] = run

    # Validate
    missing = []
    if "scene" not in cfg:
        missing.append("scene")
    else:
        for k in ("env_class", "model_xml"):
            if k not in cfg["scene"]:
                missing.append(f"scene.{k}")

    if "policy" not in cfg or "path" not in cfg["policy"]:
        # policy is required for render_demo, but not for render_model
        # We won't hard-fail here; renderers can decide if policy is required.
        pass

    # Defaults for run block
    run = cfg.setdefault("run", {})
    run.setdefault("episodes", 10)
    run.setdefault("max_seconds_per_ep", 30.0)
    run.setdefault("deterministic", True)
    run.setdefault("disable_logging", False)

    # Resolve paths
    if "scene" in cfg and "model_xml" in cfg["scene"]:
        cfg["scene"]["model_xml"] = resolve_path(cfg["scene"]["model_xml"])
    if cfg.get("policy", {}).get("path"):
        cfg["policy"]["path"] = resolve_path(cfg["policy"]["path"])

    return cfg

def load_render_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    return _normalize_schema(raw)

def import_env(dotted_path: str):
    """
    dotted_path like: 'scenes.industrial_arm_reaching.env:Z1ReachEnv'
    Returns the class object.
    """
    try:
        mod_name, class_name = dotted_path.split(":")
    except ValueError:
        raise ValueError(
            f"Invalid env_class '{dotted_path}'. Expected format: 'package.module:ClassName'"
        )
    module = importlib.import_module(mod_name)
    return getattr(module, class_name)