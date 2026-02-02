import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class AINexEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        max_steps: int = 1000,
        disable_logging: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.max_steps = max_steps
        self.disable_logging = disable_logging

        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_path, "models", "ainex_stable.xml"))
        print("Loading AINex model from:", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        n_act = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        n_obs = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._target_height = None
        self._free_root = self._has_free_root()
        self._foot_geom_ids = self._find_foot_geoms()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _has_free_root(self) -> bool:
        return self.model.njnt > 0 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    def _find_foot_geoms(self) -> list[int]:
        ids = []
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if name and "foot_col" in name:
                ids.append(gid)
        return ids

    def _apply_keyframe(self, key_name: str) -> bool:
        try:
            key = self.model.key(key_name)
        except Exception:
            return False

        if key is None:
            return False

        self.data.qpos[:] = key.qpos
        self.data.qvel[:] = key.qvel
        return True

    def _set_root_pose(self, height: float = 0.6) -> None:
        if not self._free_root:
            return
        # free joint: x,y,z, qw,qx,qy,qz
        self.data.qpos[0:7] = np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0], dtype=self.data.qpos.dtype)
        self.data.qvel[0:6] = 0.0

    def _snap_feet_to_ground(self, penetration: float = 0.01):
        if not self._free_root or not self._foot_geom_ids:
            return
        mujoco.mj_forward(self.model, self.data)
        min_z = min(float(self.data.geom_xpos[gid][2]) for gid in self._foot_geom_ids)
        desired_min_z = -abs(penetration)
        delta = min_z - desired_min_z
        self.data.qpos[2] -= delta
        self.data.qvel[0:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if not self._apply_keyframe("squat_start"):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
        self._set_root_pose(height=0.6)
        self._snap_feet_to_ground(penetration=0.01)

        mujoco.mj_forward(self.model, self.data)

        torso_pos = self.data.xpos[self._torso_id]
        self._target_height = float(torso_pos[2]) if torso_pos is not None else 0.5

        return self._get_obs(), {}

    def step(self, action):
        low, high = self.model.actuator_ctrlrange.T
        scaled_action = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = scaled_action

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        torso_pos = self.data.xpos[self._torso_id]
        torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_up = torso_mat[:, 2]

        height = float(torso_pos[2])
        upright = float(torso_up[2])

        target = self._target_height if self._target_height else 0.5
        height_err = abs(height - target)
        height_bonus = 1.0 - min(height_err / max(target, 1e-3), 1.0)
        action_penalty = float(0.001 * np.sum(np.square(action)))

        reward = 1.0 * upright + 0.5 * height_bonus - action_penalty

        terminated = height < (0.5 * target)
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"terminated": bool(terminated)}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class AINexReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        max_steps: int = 600,
        disable_logging: bool = False,
        smooth_alpha: float = 0.85,
        arm_joint_names: tuple[str, ...] = (
            "r_sho_pitch",
            "r_sho_roll",
            "r_el_pitch",
            "r_el_yaw",
            "r_gripper",
        ),
    ):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.max_steps = max_steps
        self.disable_logging = disable_logging
        self.smooth_alpha = smooth_alpha

        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_path, "models", "ainex_stable.xml"))
        print("Loading AINex reach model from:", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._arm_joint_names = set(arm_joint_names)
        self._arm_actuators, self._non_arm_actuators = self._split_actuators()
        self._arm_qpos_adr, self._arm_ranges = self._arm_joint_targets()

        n_act = len(self._arm_actuators)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        n_obs = self.model.nq + self.model.nv + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self._gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_tip")
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._free_root = self._has_free_root()
        self._foot_geom_ids = self._find_foot_geoms()

        self._home_qpos = None
        self._prev_action = np.zeros(n_act, dtype=np.float32)

    def _split_actuators(self):
        arm_actuators = []
        non_arm_actuators = []
        for act_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[act_id][0])
            joint_name = self.model.joint(joint_id).name
            if joint_name in self._arm_joint_names:
                arm_actuators.append(act_id)
            else:
                non_arm_actuators.append(act_id)
        return arm_actuators, non_arm_actuators

    def _arm_joint_targets(self):
        qpos_adrs = []
        ranges = []
        for act_id in self._arm_actuators:
            joint_id = int(self.model.actuator_trnid[act_id][0])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            jrange = self.model.jnt_range[joint_id]
            qpos_adrs.append(qpos_adr)
            ranges.append((float(jrange[0]), float(jrange[1])))
        return qpos_adrs, ranges

    def _has_free_root(self) -> bool:
        return self.model.njnt > 0 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    def _find_foot_geoms(self) -> list[int]:
        ids = []
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if name and "foot_col" in name:
                ids.append(gid)
        return ids

    def _get_obs(self) -> np.ndarray:
        grip_pos = self.data.site_xpos[self._gripper_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        return np.concatenate([self.data.qpos, self.data.qvel, grip_pos, ball_pos]).astype(np.float32)

    def _apply_keyframe(self, key_name: str) -> bool:
        try:
            key = self.model.key(key_name)
        except Exception:
            return False

        if key is None:
            return False

        self.data.qpos[:] = key.qpos
        self.data.qvel[:] = key.qvel
        return True

    def _set_root_pose(self, height: float = 0.6) -> None:
        if not self._free_root:
            return
        self.data.qpos[0:7] = np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0], dtype=self.data.qpos.dtype)
        self.data.qvel[0:6] = 0.0

    def _snap_feet_to_ground(self, penetration: float = 0.01):
        if not self._free_root or not self._foot_geom_ids:
            return
        mujoco.mj_forward(self.model, self.data)
        min_z = min(float(self.data.geom_xpos[gid][2]) for gid in self._foot_geom_ids)
        desired_min_z = -abs(penetration)
        delta = min_z - desired_min_z
        self.data.qpos[2] -= delta
        self.data.qvel[0:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _reset_ball(self):
        if self._table_geom_id < 0 or self._ball_geom_id < 0:
            return

        table_pos = self.model.geom_pos[self._table_geom_id]
        table_size = self.model.geom_size[self._table_geom_id]
        ball_radius = float(self.model.geom_size[self._ball_geom_id][0])

        margin = 0.02
        x = np.random.uniform(table_pos[0] - table_size[0] + margin, table_pos[0] + table_size[0] - margin)
        y = np.random.uniform(table_pos[1] - table_size[1] + margin, table_pos[1] + table_size[1] - margin)
        z = table_pos[2] + table_size[2] + ball_radius

        joint_id = int(self.model.body_jntadr[self._ball_body_id])
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=self.data.qpos.dtype)
        self.data.qvel[qpos_adr:qpos_adr + 6] = 0.0

    def _apply_action(self, action: np.ndarray):
        # Smooth actions for steadier arm motion
        smoothed = self.smooth_alpha * self._prev_action + (1.0 - self.smooth_alpha) * action
        self._prev_action = smoothed

        ctrl = self.data.ctrl
        ctrl[:] = 0.0

        for idx, act_id in enumerate(self._arm_actuators):
            qpos_adr = self._arm_qpos_adr[idx]
            min_q, max_q = self._arm_ranges[idx]
            span = max_q - min_q
            if span <= 0.0:
                target = self._home_qpos[qpos_adr]
            else:
                target = self._home_qpos[qpos_adr] + smoothed[idx] * 0.5 * span
                target = float(np.clip(target, min_q, max_q))
            ctrl[act_id] = target

        # Hold non-arm joints at home positions to keep legs still
        if self._home_qpos is not None:
            for act_id in self._non_arm_actuators:
                joint_id = int(self.model.actuator_trnid[act_id][0])
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                ctrl[act_id] = self._home_qpos[qpos_adr]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action[:] = 0.0

        if not self._apply_keyframe("squat_start"):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
        self._set_root_pose(height=0.6)
        self._snap_feet_to_ground(penetration=0.01)

        self._home_qpos = self.data.qpos.copy()
        self._reset_ball()
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        grip_pos = self.data.site_xpos[self._gripper_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        dist = float(np.linalg.norm(grip_pos - ball_pos))

        action_delta = float(np.linalg.norm(self._prev_action - action))
        reward = 1.5 * (1.0 / (1.0 + 10.0 * dist)) - 0.05 * action_delta

        terminated = dist < 0.05
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"terminated": bool(terminated)}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
