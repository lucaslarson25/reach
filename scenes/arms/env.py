"""
Arm-only reach env: you upload only the arm XML. The scene (floor + ball + arm)
is composed at load time. Training adapts to arm length and DOF.
"""

import os
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import csv

from scenes.arms.arm_registry import (
    resolve_model_path,
    resolve_ee_site_name,
    get_arm_config,
    compute_reach_from_model,
)


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        model_path: str | None = None,
        arm_id: str | None = None,
        ee_site_name: str | None = None,
        reach_min: float | None = None,
        reach_max: float | None = None,
        metrics_csv_path: str | None = None,
        disable_logging: bool = False,
    ):
        """
        Reach env: upload only the arm. Scene (floor + ball + arm) is composed at load time.

        - model_path: optional full scene XML; if set, used as-is. Otherwise scene is composed.
        - arm_id: registry key (e.g. "z1", "arm_2link"). Composed scene uses this arm.
        - ee_site_name: end-effector site; auto-resolved if None.
        - reach_min / reach_max: ball sampling radius; from registry or computed if not set.
        """
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self._arm_id = arm_id

        resolved = resolve_model_path(arm_id, model_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Model not found: {resolved}")
        print("Loading model from:", resolved)

        self.model = mujoco.MjModel.from_xml_path(resolved)
        self.data = mujoco.MjData(self.model)

        self._ee_site_name = resolve_ee_site_name(self.model, arm_id, ee_site_name)
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name)
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        cfg = get_arm_config(arm_id)
        self._reach_min = reach_min if reach_min is not None else (cfg.reach_min if cfg else 0.08)
        if reach_max is not None:
            self._reach_max = reach_max
        elif cfg and cfg.reach_max is not None:
            self._reach_max = cfg.reach_max
        else:
            self._reach_max = compute_reach_from_model(self.model, self.data, self._ee_site_id)
            print("Computed reach_max =", round(self._reach_max, 3))

        self._home_keyframe_name = (cfg.home_keyframe_name if cfg else "home") or "home"
        self._has_home_key = False
        for i in range(self.model.nkey):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i) == self._home_keyframe_name:
                self._has_home_key = True
                break

        n_act = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        n_obs = self.model.nq + self.model.nv + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.disable_logging = disable_logging
        if not disable_logging:
            if metrics_csv_path is None:
                project_root = Path(__file__).resolve().parents[2]
                metrics_csv_path = os.path.join(project_root, "logs", "episode_metrics.csv")
            self.metrics_csv_path = metrics_csv_path
        else:
            self.metrics_csv_path = None
        self.episode_count = 0
        self._csv_initialized = False

    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos]).astype(np.float32)

    def _set_ball_random_pos(self):
        r = np.random.uniform(self._reach_min, self._reach_max)
        theta = np.random.uniform(0, 2 * np.pi)
        x, y, z = r * np.cos(theta), r * np.sin(theta), 0.05
        self.model.body("ball").pos[:] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    def _initialize_csv(self):
        if self._csv_initialized or not self.metrics_csv_path:
            return
        csv_dir = os.path.dirname(self.metrics_csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "distance_avg", "distance_final", "orientation_avg",
                     "action_norm_avg", "reward_sum", "success"]
                )
        self._csv_initialized = True

    def _log_episode_metrics(self):
        if self.disable_logging or not self.metrics_csv_path or not hasattr(self, "_episode_metrics"):
            return
        self._initialize_csv()
        m = self._episode_metrics
        if m["step_count"] == 0:
            return
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count,
                m["distance_sum"] / m["step_count"],
                m["distance_final"],
                m["orientation_sum"] / m["step_count"],
                m["action_norm_sum"] / m["step_count"],
                m["reward_sum"],
                int(m["success"]),
            ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self._episode_metrics = {
            "step_count": 0, "distance_sum": 0.0, "distance_final": 0.0,
            "orientation_sum": 0.0, "action_norm_sum": 0.0, "reward_sum": 0.0, "success": False,
        }
        if self._has_home_key:
            self.data.qpos[:] = self.model.key(self._home_keyframe_name).qpos
        else:
            self.data.qpos[:] = self.model.qpos0
        self._set_ball_random_pos()
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self._ee_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        self.prev_dist = float(np.linalg.norm(ee_pos - ball_pos))
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        low, high = self.model.actuator_ctrlrange.T
        scaled = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = 0.5 * self.data.ctrl + 0.5 * scaled
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        ee_pos = self.data.site_xpos[self._ee_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        dist = float(np.linalg.norm(ee_pos - ball_pos))
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist
        near = float(np.clip(dist / 0.1, 0.0, 1.0))
        dense = float((1.0 / ((1.0 + 10 * dist) ** 1.5)) * near)
        progress = float(np.clip(self.prev_dist - dist, -0.03, 0.03) * near)
        xmat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        target_dir = (ball_pos - ee_pos)
        n = np.linalg.norm(target_dir)
        if n > 0:
            target_dir /= n
        ori = float(np.dot(xmat[:, 0], target_dir))
        reward = 3.0 * dense + 1.5 * progress + 0.45 * ori - 0.05 * float(np.sum(np.square(action)))
        terminated = dist < 0.05
        truncated = self.step_count >= 1000
        if terminated:
            reward += 550.0
        if dist < 0.1 and not terminated:
            reward -= 0.01
        self.prev_dist = dist
        if hasattr(self, "_episode_metrics"):
            self._episode_metrics["step_count"] += 1
            self._episode_metrics["distance_sum"] += dist
            self._episode_metrics["distance_final"] = dist
            self._episode_metrics["orientation_sum"] += ori
            self._episode_metrics["action_norm_sum"] += float(np.linalg.norm(action))
            self._episode_metrics["reward_sum"] += reward
            if terminated:
                self._episode_metrics["success"] = True
        if (terminated or truncated) and hasattr(self, "_episode_metrics") and self._episode_metrics["step_count"] > 0:
            self._log_episode_metrics()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"terminated": bool(terminated)}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if hasattr(self, "_episode_metrics") and self._episode_metrics.get("step_count", 0) > 0:
            self._log_episode_metrics()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
