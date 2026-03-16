# Train separate policies per arm (multi-arm only).
# Usage: python -m scenes.arms.training.arm_train_per_arm --config config/arms.yaml
#
# Requires scene.per_arm_policies: true and arm_id with multiple arms (e.g. aloha).
# Trains N policies; each policy i controls arm i only (other arms held fixed).

import os
import sys
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback, CallbackList

from scenes.arms.env import ArmReachEnv
from scenes.arms.arm_registry import get_arm_info

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
            self.logger.record(
                "custom/terminated_ratio",
                f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%",
            )
        self.terminated_episodes = 0
        self.total_episodes = 0


def make_env(arm_id, model_path, ball_mode, fix_arm_indices, reward_time_penalty, reward_smoothness, reward_move_away_penalty=0.5, reward_style="z1", reach_min_mode=None, reach_min_fraction=None, reach_min_floor=None, ee_priority_scale=True, ctrl_blend_new=None):
    def _init():
        return ArmReachEnv(
            arm_id=arm_id,
            model_path=model_path,
            ball_mode=ball_mode,
            fix_arm_indices=list(fix_arm_indices),
            reward_time_penalty=reward_time_penalty,
            reward_smoothness=reward_smoothness,
            reward_move_away_penalty=reward_move_away_penalty,
            reward_style=reward_style,
            reach_min_mode=reach_min_mode or "auto",
            reach_min_fraction=reach_min_fraction,
            reach_min_floor=reach_min_floor,
            ee_priority_scale=ee_priority_scale,
            ctrl_blend_new=ctrl_blend_new,
        )

    return _init


def main():
    import argparse
    from config.arms_loader import load_arms_config, REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Train per-arm PPO policies (multi-arm only)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "config", "arms.yaml"),
        help="Path to arms.yaml",
    )
    args = parser.parse_args()

    cfg = load_arms_config(args.config)
    scene = cfg["scene"]
    train = cfg["train"]

    arm_id = os.getenv("ARM_ID", "").strip() or scene.get("arm_id") or "aloha"
    model_path = os.getenv("MODEL_PATH", "").strip() or scene.get("model_path")
    ball_mode = scene.get("ball_mode", "shared")

    info = get_arm_info(arm_id)
    if not info or not info.get("ee_sites"):
        print("ERROR: Arm", arm_id, "not found or has no EE sites")
        return
    n_arms = len(info["ee_sites"])
    if n_arms <= 1:
        print("ERROR: per_arm_policies requires multi-arm model (e.g. aloha). n_arms =", n_arms)
        return

    total_steps = int(os.getenv("TOTAL_STEPS", str(train.get("total_steps", 300000))))
    reward_time_penalty = train.get("reward_time_penalty", 0.0005)
    reward_smoothness = train.get("reward_smoothness", 0.02)
    reward_move_away_penalty = train.get("reward_move_away_penalty", 0.5)
    reward_style = train.get("reward_style", "z1")
    reach_min_mode = train.get("reach_min_mode", "auto")
    reach_min_fraction = train.get("reach_min_fraction")
    reach_min_floor = train.get("reach_min_floor")
    ee_priority_scale = train.get("ee_priority_scale", True)
    ctrl_blend_new = train.get("ctrl_blend_new")
    use_mps_env = os.getenv("USE_MPS", "").strip().lower() in ("1", "true", "yes")
    use_mps = use_mps_env or train.get("use_mps", False)

    cores = os.cpu_count() or 8
    th.set_num_threads(max(1, cores - 1))
    device = train.get("device")
    if device is None or (isinstance(device, str) and device.lower() == "null"):
        device = "mps" if (use_mps and th.backends.mps.is_available()) else "cpu"

    policy_dir = train.get("policy_dir") or "policies"
    if not os.path.isabs(policy_dir):
        policy_dir = os.path.join(REPO_ROOT, policy_dir)
    os.makedirs(policy_dir, exist_ok=True)

    k = total_steps // 1000

    for arm_i in range(n_arms):
        fix = set(range(n_arms)) - {arm_i}
        print(f"\n--- Training policy for arm {arm_i} (fixing arms {list(fix)}) ---")
        env = DummyVecEnv([
            make_env(arm_id, model_path, ball_mode, fix, reward_time_penalty, reward_smoothness, reward_move_away_penalty, reward_style, reach_min_mode, reach_min_fraction, reach_min_floor, ee_priority_scale, ctrl_blend_new)
        ])
        model = PPO(
            "MlpPolicy",
            env,
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
        callbacks = CallbackList([ProgressBarCallback(), TerminationRatioCallback()])
        model.learn(total_timesteps=total_steps, callback=callbacks)
        save_path = os.path.join(policy_dir, f"ppo_arms_{arm_id}_arm{arm_i}_mac_{k}k")
        model.save(save_path)
        print("Saved:", save_path + ".zip")
        env.close()

    print("\nDone. All per-arm policies saved.")


if __name__ == "__main__":
    th.set_default_dtype(th.float32)
    main()
