import os
import numpy as np
import mujoco
import mujoco.viewer  # ensure the viewer submodule is imported for mujoco >= 3.2.x
import gymnasium as gym
from gymnasium import spaces


class ReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "../models/arm_scene.xml")
        xml_path = os.path.abspath(xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 2 joint actions (torque normalized to [-1,1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: qpos(2) + qvel(2) + ee_pos(3) + target_pos(3) = 10 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.target_site = "ee_site"
        self.target_body = "target"
        self.render_mode = render_mode

        # viewer is optional — only created when render_mode == "human"
        self.viewer = None
        if self.render_mode == "human":
            # launch viewer (keeps control until closed)
            # mujoco.viewer.launch returns a viewer object (3.2+). We keep reference to it.
            self.viewer = mujoco.viewer.launch(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # randomize target position in reachable workspace
        x, y = np.random.uniform(-0.2, 0.2, size=2)
        body_id = self.model.body(self.target_body).id
        # update body position array (world-frame)
        self.model.body_pos[body_id] = np.array([x, y, 0.2], dtype=np.float64)

        # reset joint positions & velocities
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # render one frame if requested
        if self.render_mode == "human" and self.viewer is not None:
            self._viewer_render_once()

        return self._get_obs(), {}

    def step(self, action):
        # clip action to action_space and apply to mujoco ctrl
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, self.action_space.low, self.action_space.high)

        # If model has actuators, apply; otherwise do nothing (safety)
        if self.data.ctrl.size == act.size:
            self.data.ctrl[:] = act
        elif self.data.ctrl.size > 0:
            # if there are more actuators than actions, fill leading subset
            self.data.ctrl[: min(act.size, self.data.ctrl.size)] = act[: min(act.size, self.data.ctrl.size)]
        else:
            # no actuators defined — do nothing
            pass

        # simulate several physics steps per control step (control frequency)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # optionally render
        if self.render_mode == "human" and self.viewer is not None:
            self._viewer_render_once()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_done()
        truncated = False
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        # qpos(2), qvel(2)
        qpos = np.asarray(self.data.qpos[:2], dtype=np.float32)
        qvel = np.asarray(self.data.qvel[:2], dtype=np.float32)

        # end-effector position via site
        ee_site_id = self.model.site(self.target_site).id
        ee_pos = np.asarray(self.data.site_xpos[ee_site_id], dtype=np.float32)

        # target body world position
        body_id = self.model.body(self.target_body).id
        target_pos = np.asarray(self.model.body_pos[body_id], dtype=np.float32)

        # concatenate to 10-dim vector
        return np.concatenate([qpos, qvel, ee_pos, target_pos], axis=0)

    def _compute_reward(self):
        ee_site_id = self.model.site(self.target_site).id
        ee_pos = np.asarray(self.data.site_xpos[ee_site_id], dtype=np.float32)

        body_id = self.model.body(self.target_body).id
        target_pos = np.asarray(self.model.body_pos[body_id], dtype=np.float32)

        dist = np.linalg.norm(ee_pos - target_pos)
        reward = -float(dist)
        # small control penalty could be added if desired
        return reward

    def _check_done(self):
        ee_site_id = self.model.site(self.target_site).id
        ee_pos = np.asarray(self.data.site_xpos[ee_site_id], dtype=np.float32)

        body_id = self.model.body(self.target_body).id
        target_pos = np.asarray(self.model.body_pos[body_id], dtype=np.float32)

        return float(np.linalg.norm(ee_pos - target_pos)) < 0.03

    def render(self):
        if self.render_mode == "human" and self.viewer is not None:
            self._viewer_render_once()
        elif self.render_mode == "rgb_array":
            # Offscreen render into an RGB array
            width, height = 640, 480
            # mujoco.render returns HxWx3 uint8 in newer bindings
            img = mujoco.render(self.model, self.data, width, height, mode="offscreen", camera_id=-1)
            return img
        else:
            return None

    def close(self):
        # close viewer if launched
        if self.viewer is not None:
            try:
                # some viewer objects provide close(), some use context management
                if hasattr(self.viewer, "close"):
                    self.viewer.close()
                elif hasattr(self.viewer, "destroy"):
                    self.viewer.destroy()
            except Exception:
                pass
            self.viewer = None

    def _viewer_render_once(self):
        # call sync() if present (preferred), else fallback to render()
        if self.viewer is None:
            return
        if hasattr(self.viewer, "sync"):
            # sync advances the viewer to the model/data state
            try:
                self.viewer.sync()
                return
            except Exception:
                pass
        if hasattr(self.viewer, "render"):
            try:
                self.viewer.render()
                return
            except Exception:
                pass