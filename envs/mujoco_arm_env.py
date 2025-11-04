import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import mujoco
import mujoco.viewer
import time

class Z1ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        base_path = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(base_path, "..", "models", "z1scene.xml")
        xml_path = os.path.normpath(xml_path)
        print("Loading model from:", xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0

        self.render_mode = render_mode
        self.viewer = None

        # Number of actuators = 6 (from z1.xml)
        n_act = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        # Observation: joint angles + end-effector pos + ball pos
        n_obs = self.model.nq + self.model.nv + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    def _get_obs(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        # Ensure float32 for MPS
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos]).astype(np.float32)

    def _set_ball_random_pos(self):
        x = np.random.uniform(0.2, 0.3)
        y = np.random.uniform(0.2, 0.3)
        y *= np.random.choice([1, -1])
        z = 0.05
        self.model.body("ball").pos[:] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.model.key("home").qpos
        self._set_ball_random_pos()
        mujoco.mj_forward(self.model, self.data)

        # Initialize previous distance for reward shaping
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        self.prev_dist = np.linalg.norm(ee_pos - ball_pos)
        self.step_count = 0

        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        # Apply scaled actions to actuators
        low, high = self.model.actuator_ctrlrange.T
        scaled_action = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = 0.5 * self.data.ctrl + 0.5 * scaled_action

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        max_steps = 1000  # 10 seconds * 100 Hz

        # Compute 3D positions
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        dist = np.linalg.norm(ee_pos - ball_pos)

        # Initialize previous distance (for progress reward)
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist

        # --- Shaping rewards ---
        # Scale dense/progress reward when near the goal
        near_goal_scale = np.clip(dist / 0.1, 0.0, 1.0)  # 0 at 0, 1 at 0.1m away

        # Dense distance-based reward (closer = higher reward)
        dense_reward = (1.0 / ((1.0 + 10 * dist) ** 1.5)) * near_goal_scale

        # Reward for progress toward the ball since last step
        progress = np.clip(self.prev_dist - dist, -0.03, 0.03) * near_goal_scale

        # Orientation reward (encourages tip to face target)
        xmat = self.data.site_xmat[ee_id].reshape(3, 3)
        ee_dir = xmat[:, 0]  # local x-axis in world frame
        target_dir = (ball_pos - ee_pos)
        if np.linalg.norm(target_dir) > 0:
            target_dir /= np.linalg.norm(target_dir)
        orientation_reward = np.dot(ee_dir, target_dir)

        # Small penalty for large actions to encourage smooth control
        action_penalty = 0.05 * np.sum(np.square(action))

        # Combine rewards
        reward = 3.0 * dense_reward + 1.5 * progress + 0.45 * orientation_reward - action_penalty

        # --- Success bonus ---
        success_threshold = 0.05
        terminated = dist < success_threshold
        truncated = self.step_count >= max_steps

        if terminated:
            reward += 550.0  # Strong terminal reward
        elif truncated:
            pass

        # Optional: small penalty for hovering too long above ball
        if dist < 0.1 and not terminated:
            reward -= 0.01  # encourages downward movement toward ball

        # Update previous distance
        self.prev_dist = dist

        # Ensure obs (and reward for SB3) are correct dtypes on MPS
        return self._get_obs().astype(np.float32), float(reward), terminated, truncated, {"terminated": terminated}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None