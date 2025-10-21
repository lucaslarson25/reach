import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
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

        self.render_mode = render_mode
        self.viewer = None

        # Number of actuators = 6 (from z1.xml)
        n_act = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        # Observation: joint angles + end-effector pos + ball pos
        n_obs = self.model.nq + self.model.nv + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    def _get_obs(self):
        ee_pos = self.data.body("link06").xpos
        ball_pos = self.data.body("ball").xpos
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos])

    def _set_ball_random_pos(self):
        x = np.random.uniform(0.2, 0.3)
        y = np.random.uniform(0.2, 0.3)

        x *= np.random.choice([1, -1])
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
        ee_pos = self.data.body("link06").xpos
        ball_pos = self.data.body("ball").xpos
        self.prev_dist = np.linalg.norm(ee_pos - ball_pos)

        return self._get_obs(), {}

    def step(self, action):
        # Apply scaled actions to actuators
        low, high = self.model.actuator_ctrlrange.T
        scaled_action = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = 0.5 * self.data.ctrl + 0.5 * scaled_action
        mujoco.mj_step(self.model, self.data)

        # Compute 3D positions
        ee_pos = self.data.body("link06").xpos
        ball_pos = self.data.body("ball").xpos
        dist = np.linalg.norm(ee_pos - ball_pos)

        # Initialize previous distance (for progress reward)
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist

        # Reward components
        dense_reward = 1.0 / (1.0 + 10 * dist)             # Smooth distance-based reward
        progress = self.prev_dist - dist                   # Reward for getting closer
        action_penalty = 0.01 * np.sum(np.square(action))  # Penalize excessive motion

        # Combine components
        reward = dense_reward + 10.0 * progress - action_penalty

        # Success condition
        terminated = dist < 0.005
        if terminated:
            reward += 5.0  # Bonus for reaching the goal

        # Update previous distance
        self.prev_dist = dist

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
