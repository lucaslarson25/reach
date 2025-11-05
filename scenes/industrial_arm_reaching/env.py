import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class Z1ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, model_path: str | None = None):
        """
        Minimal migration:
        - new optional `model_path`. If None, use the scene's default z1scene.xml
        - keep everything else the same
        """
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0

        # Resolve XML path (default to this scene's z1scene.xml)
        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_path, "models", "z1scene.xml"))
        print("Loading model from:", model_path)

        # Load MuJoCo model/data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Action space (same as before)
        n_act = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        # Observation: qpos + qvel + ee_pos (3) + ball_pos (3)
        n_obs = self.model.nq + self.model.nv + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        # Ensure float32 to avoid MPS float64 issues
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos]).astype(np.float32)

    def _set_ball_random_pos(self):
        x = np.random.uniform(0.2, 0.3)
        y = np.random.uniform(0.2, 0.3) * np.random.choice([1, -1])
        z = 0.05
        self.model.body("ball").pos[:] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # home keyframe if present
        self.data.qpos[:] = self.model.key("home").qpos
        self._set_ball_random_pos()
        mujoco.mj_forward(self.model, self.data)

        # initialize previous distance
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        self.prev_dist = float(np.linalg.norm(ee_pos - ball_pos))
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        # Scale actions into actuator ctrl range
        low, high = self.model.actuator_ctrlrange.T
        scaled_action = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = 0.5 * self.data.ctrl + 0.5 * scaled_action

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        max_steps = 1000  # ~10s at 100Hz

        # positions
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        dist = float(np.linalg.norm(ee_pos - ball_pos))

        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist

        # Reward shaping (unchanged, cast to float to be safe)
        near_goal_scale = float(np.clip(dist / 0.1, 0.0, 1.0))
        dense_reward = float((1.0 / ((1.0 + 10 * dist) ** 1.5)) * near_goal_scale)
        progress = float(np.clip(self.prev_dist - dist, -0.03, 0.03) * near_goal_scale)

        xmat = self.data.site_xmat[ee_id].reshape(3, 3)
        ee_dir = xmat[:, 0]
        target_dir = (ball_pos - ee_pos)
        norm = np.linalg.norm(target_dir)
        if norm > 0:
            target_dir /= norm
        orientation_reward = float(np.dot(ee_dir, target_dir))

        action_penalty = float(0.05 * np.sum(np.square(action)))

        reward = 3.0 * dense_reward + 1.5 * progress + 0.45 * orientation_reward - action_penalty

        success_threshold = 0.05
        terminated = dist < success_threshold
        truncated = self.step_count >= max_steps

        if terminated:
            reward += 550.0

        if dist < 0.1 and not terminated:
            reward -= 0.01

        self.prev_dist = dist

        # Return obs as float32
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"terminated": bool(terminated)}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                # keep default (non-passive) for generic use; mac viewer will be launched externally
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None