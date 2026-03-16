# Arm-only reach training (Mac). YAML-driven. Run from project root:
#   python -m scenes.arms.training.arm_train_mac --config config/arms.yaml
# Env ARM_ID / TOTAL_STEPS / MODEL_PATH override the config.

import os
import sys
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback, CallbackList

from scenes.arms.env import ArmReachEnv

# Repo root for default config path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)


class TerminationRatioCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_rollout_start(self) -> None:
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if isinstance(dones, (list, np.ndarray)):
            for done, info in zip(dones, infos or []):
                if done:
                    self.total_episodes += 1
                    if isinstance(info, dict) and info.get("terminated", False):
                        self.terminated_episodes += 1
        elif isinstance(dones, (bool, getattr(np, "bool_", bool))):
            if dones:
                self.total_episodes += 1
                if isinstance(infos, dict) and infos.get("terminated", False):
                    self.terminated_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.total_episodes > 0:
            ratio = self.terminated_episodes / self.total_episodes
            self.logger.record("custom/terminated_ratio",
                f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%")
        self.terminated_episodes = 0
        self.total_episodes = 0


def get_ppo_kwargs(train, device):
    """Build PPO kwargs from train config (config/arms.yaml). Uses SB3 defaults for omitted keys."""
    return dict(
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        n_steps=train.get("n_steps", 2048),
        batch_size=train.get("batch_size", 64),
        n_epochs=train.get("n_epochs", 10),
        learning_rate=float(train.get("learning_rate", 3e-4)),
        clip_range=float(train.get("clip_range", 0.2)),
        gamma=float(train.get("gamma", 0.99)),
        gae_lambda=float(train.get("gae_lambda", 0.95)),
        vf_coef=float(train.get("vf_coef", 0.5)),
        ent_coef=float(train.get("ent_coef", 0.0)),
        max_grad_norm=float(train.get("max_grad_norm", 0.5)),
        verbose=1,
        seed=int(train.get("seed", 42)),
    )


def make_env(arm_id=None, model_path=None, ball_mode=None, reward_time_penalty=None, reward_smoothness=None, reward_move_away_penalty=None, reward_style=None, reach_min_mode=None, reach_min_fraction=None, reach_min_floor=None, reach_max_cap=None, ee_priority_scale=True, ctrl_blend_new=None, initial_pose="home", initial_keyframe=None, joint_limit_margin_penalty=None):
    reach_max = float(reach_max_cap) if reach_max_cap is not None else None
    def _init():
        kwargs = dict(
            arm_id=arm_id,
            model_path=model_path,
            ball_mode=ball_mode or "shared",
            reward_time_penalty=reward_time_penalty or 0.0005,
            reward_smoothness=reward_smoothness or 0.02,
            reward_move_away_penalty=reward_move_away_penalty or 0.5,
            reward_style=reward_style or "z1",
            reach_min_mode=reach_min_mode or "auto",
            reach_min_fraction=reach_min_fraction,
            reach_min_floor=reach_min_floor,
            ee_priority_scale=ee_priority_scale,
            ctrl_blend_new=ctrl_blend_new,
        )
        if reach_max is not None:
            kwargs["reach_max"] = reach_max
        kwargs["initial_pose"] = initial_pose
        if initial_keyframe is not None:
            kwargs["initial_keyframe"] = initial_keyframe
        if joint_limit_margin_penalty is not None:
            kwargs["joint_limit_margin_penalty"] = joint_limit_margin_penalty
        return ArmReachEnv(**kwargs)
    return _init


def main():
    import argparse
    from config.arms_loader import load_arms_config, apply_arm_overrides, REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Train PPO for arm reach. Default: panda, 300k steps. CLI overrides YAML.",
    )
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "config", "arms.yaml"),
                        help="Path to arms.yaml (default: config/arms.yaml)")
    parser.add_argument("--arm-id", type=str, default=None,
                        help="Arm to train (e.g. panda, aloha, ur5e). Overrides config.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Total timesteps (default: 300000). Overrides config.")
    parser.add_argument("--ball-mode", type=str, default=None, choices=["shared", "per_arm"],
                        help="Ball mode for multi-arm. Overrides config.")
    parser.add_argument("--per-arm-policies", action="store_true",
                        help="Train separate policy per arm (multi-arm only). Overrides config.")
    args = parser.parse_args()

    cfg = load_arms_config(args.config)
    arm_id = args.arm_id or os.getenv("ARM_ID", "").strip() or cfg["scene"].get("arm_id") or "panda"
    cfg = apply_arm_overrides(cfg, arm_id)
    scene = cfg["scene"]
    train = cfg["train"]

    model_path = os.getenv("MODEL_PATH", "").strip() or scene.get("model_path")
    ball_mode = args.ball_mode or scene.get("ball_mode", "shared")
    per_arm_policies = args.per_arm_policies or scene.get("per_arm_policies", False)
    scene["arm_id"] = arm_id
    scene["ball_mode"] = ball_mode
    scene["per_arm_policies"] = per_arm_policies
    reward_time_penalty = train.get("reward_time_penalty", 0.0005)
    reward_smoothness = train.get("reward_smoothness", 0.02)
    reward_move_away_penalty = train.get("reward_move_away_penalty", 0.5)
    reward_style = train.get("reward_style", "z1")
    reach_min_mode = train.get("reach_min_mode", "auto")
    reach_max_cap = train.get("reach_max_cap")
    reach_min_fraction = train.get("reach_min_fraction")
    reach_min_floor = train.get("reach_min_floor")
    ee_priority_scale = train.get("ee_priority_scale", True)
    ctrl_blend_new = train.get("ctrl_blend_new")
    initial_pose = scene.get("initial_pose") or train.get("initial_pose") or "home"
    initial_keyframe = scene.get("initial_keyframe")
    joint_limit_margin_penalty = train.get("joint_limit_margin_penalty")
    total_timesteps = args.steps or int(os.getenv("TOTAL_STEPS", str(train.get("total_steps", 300000))))
    use_mps_env = os.getenv("USE_MPS", "").strip().lower() in ("1", "true", "yes")
    use_mps = use_mps_env or train.get("use_mps", False)
    print("Config:", args.config, "| arm_id:", arm_id)
    if model_path:
        print("Model path:", model_path)

    # Per-arm policies: train N policies (multi-arm only)
    from scenes.arms.arm_registry import get_arm_info
    info = get_arm_info(arm_id)
    n_arms = len(info["ee_sites"]) if info and info.get("ee_sites") else 1
    if per_arm_policies and n_arms > 1:
        from scenes.arms.training.arm_train_per_arm import make_env as per_arm_make_env
        policy_dir = train.get("policy_dir") or "policies"
        if not os.path.isabs(policy_dir):
            policy_dir = os.path.join(REPO_ROOT, policy_dir)
        os.makedirs(policy_dir, exist_ok=True)
        k = total_timesteps // 1000
        for arm_i in range(n_arms):
            fix = set(range(n_arms)) - {arm_i}
            print(f"\n--- Training arm {arm_i} (fixing {list(fix)}) ---")
            env = DummyVecEnv([per_arm_make_env(arm_id, model_path, ball_mode, fix, reward_time_penalty, reward_smoothness, reward_move_away_penalty, reward_style, reach_min_mode, reach_min_fraction, reach_min_floor, ee_priority_scale, ctrl_blend_new)])
            device = "mps" if (use_mps and th.backends.mps.is_available()) else "cpu"
            model = PPO("MlpPolicy", env, **get_ppo_kwargs(train, device))
            model.learn(total_timesteps=total_timesteps, callback=CallbackList([ProgressBarCallback(), TerminationRatioCallback()]))
            save_path = os.path.join(policy_dir, f"ppo_arms_{arm_id}_arm{arm_i}_mac_{k}k")
            model.save(save_path)
            print("Saved:", save_path + ".zip")
            env.close()
        return

    cores = os.cpu_count() or 8
    th.set_num_threads(max(1, cores - 1))
    device = train.get("device")
    if device is None or (isinstance(device, str) and device.lower() == "null"):
        device = "mps" if (use_mps and th.backends.mps.is_available()) else "cpu"
    print("Cores:", cores, "Device:", device)

    env = DummyVecEnv([
        make_env(
            arm_id=arm_id,
            model_path=model_path,
            ball_mode=ball_mode,
            reward_time_penalty=reward_time_penalty,
            reward_smoothness=reward_smoothness,
            reward_move_away_penalty=reward_move_away_penalty,
            reward_style=reward_style,
            reach_min_mode=reach_min_mode,
            reach_min_fraction=reach_min_fraction,
            reach_min_floor=reach_min_floor,
            reach_max_cap=reach_max_cap,
            ee_priority_scale=ee_priority_scale,
            ctrl_blend_new=ctrl_blend_new,
            initial_pose=initial_pose,
            initial_keyframe=initial_keyframe,
            joint_limit_margin_penalty=joint_limit_margin_penalty,
        )
    ])
    model = PPO("MlpPolicy", env, **get_ppo_kwargs(train, device))
    callbacks = CallbackList([ProgressBarCallback(), TerminationRatioCallback()])
    print("Training for", total_timesteps, "timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    policy_dir = train.get("policy_dir") or "policies"
    if not os.path.isabs(policy_dir):
        policy_dir = os.path.join(REPO_ROOT, policy_dir)
    os.makedirs(policy_dir, exist_ok=True)
    tag = arm_id or "custom"
    save_path = os.path.join(policy_dir, f"ppo_arms_{tag}_mac_{total_timesteps//1000}k")
    model.save(save_path)
    print("Saved:", save_path + ".zip")
    env.close()


if __name__ == "__main__":
    th.set_default_dtype(th.float32)
    main()
