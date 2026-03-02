import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from scenes.ainex_soccer.action_group_reach import (
    ActionGroupReach,
    load_action_group_reach,
    get_interpolated_reference,
)


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
        action_groups_dir: Optional[str | Path] = None,
        action_group_names: Optional[list[str]] = None,
        action_group_blend: float = 0.5,
        include_ref_in_obs: bool = True,
        elbow_yaw_limit_rad: Optional[float] = 0.4,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.max_steps = max_steps
        self.disable_logging = disable_logging
        self.smooth_alpha = smooth_alpha
        self.action_group_blend = float(np.clip(action_group_blend, 0.0, 1.0))
        self.include_ref_in_obs = include_ref_in_obs
        self.elbow_yaw_limit_rad = elbow_yaw_limit_rad

        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_path, "models", "ainex_stable.xml"))
        print("Loading AINex reach model from:", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._arm_joint_names = set(arm_joint_names)
        self._arm_actuators, self._non_arm_actuators = self._split_actuators()
        self._arm_qpos_adr, self._arm_ranges = self._arm_joint_targets()

        # r_el_yaw is index 3 (r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper)
        self._elbow_yaw_idx = 3

        n_act = len(self._arm_actuators)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        n_obs = self.model.nq + self.model.nv + 3 + 3
        if include_ref_in_obs:
            n_obs += n_act
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self._gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_tip")
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._free_root = self._has_free_root()
        self._foot_geom_ids = self._find_foot_geoms()

        self._home_qpos = None
        self._prev_action = np.zeros(n_act, dtype=np.float32)

        self._action_groups: list[ActionGroupReach] = []
        self._current_ag: Optional[ActionGroupReach] = None
        self._current_ref = np.zeros(n_act, dtype=np.float32)

        if action_groups_dir is not None and action_group_blend > 0:
            ag_dir = Path(action_groups_dir)
            names = action_group_names or [
                "raise_right_hand", "place_block", "right_hand_put_block", "hand_open", "hand_back"
            ]
            joint_ranges = list(self._arm_ranges)
            for name in names:
                path = ag_dir / f"{name}.csv"
                if path.exists():
                    try:
                        ag = load_action_group_reach(path, joint_ranges)
                        if ag.frames:
                            self._action_groups.append(ag)
                    except Exception as e:
                        if not disable_logging:
                            print(f"[AINexReach] Skip {name}: {e}")
            if self._action_groups and not disable_logging:
                print(f"[AINexReach] Loaded {len(self._action_groups)} action groups, blend={action_group_blend}")

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

    def _ref_to_normalized(self, ref_rad: np.ndarray) -> np.ndarray:
        """Convert reference joint angles (rad) to policy space [-1, 1]."""
        out = np.zeros_like(ref_rad, dtype=np.float32)
        for i, (min_q, max_q) in enumerate(self._arm_ranges):
            span = max_q - min_q
            home = self._home_qpos[self._arm_qpos_adr[i]] if self._home_qpos is not None else 0.0
            if span > 1e-6:
                out[i] = (ref_rad[i] - home) / (0.5 * span)
            else:
                out[i] = 0.0
        return np.clip(out, -1.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        grip_pos = self.data.site_xpos[self._gripper_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        obs = np.concatenate([self.data.qpos, self.data.qvel, grip_pos, ball_pos]).astype(np.float32)
        if self.include_ref_in_obs:
            obs = np.concatenate([obs, self._current_ref.astype(np.float32)])
        return obs

    def get_end_effector_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._gripper_site_id].copy()

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
        # Blend with action group reference when available
        if self._current_ag is not None and self.action_group_blend > 0:
            phase = self.step_count / max(1, self.max_steps)
            ref_rad = get_interpolated_reference(self._current_ag, phase)
            ref_norm = self._ref_to_normalized(ref_rad)
            blended = (1.0 - self.action_group_blend) * action + self.action_group_blend * ref_norm
            blended = np.clip(blended, -1.0, 1.0).astype(np.float32)
            self._current_ref[:] = ref_norm
            action = blended
        else:
            self._current_ref[:] = 0.0

        # Smooth actions for steadier arm motion
        smoothed = self.smooth_alpha * self._prev_action + (1.0 - self.smooth_alpha) * action
        self._prev_action = smoothed

        ctrl = self.data.ctrl
        ctrl[:] = 0.0

        for idx, act_id in enumerate(self._arm_actuators):
            qpos_adr = self._arm_qpos_adr[idx]
            min_q, max_q = self._arm_ranges[idx]

            # Constrain r_el_yaw to avoid "pouring" forearm rotation during reach
            if idx == self._elbow_yaw_idx and self.elbow_yaw_limit_rad is not None:
                home = self._home_qpos[qpos_adr]
                min_q = float(max(min_q, home - self.elbow_yaw_limit_rad))
                max_q = float(min(max_q, home + self.elbow_yaw_limit_rad))

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

        if self._action_groups:
            self._current_ag = random.choice(self._action_groups)
        else:
            self._current_ag = None

        if not self._apply_keyframe("squat_start"):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
        self._set_root_pose(height=0.6)
        self._snap_feet_to_ground(penetration=0.01)

        self._home_qpos = self.data.qpos.copy()
        self._reset_ball()
        mujoco.mj_forward(self.model, self.data)

        if self._current_ag is not None:
            ref_rad = get_interpolated_reference(self._current_ag, 0.0)
            self._current_ref[:] = self._ref_to_normalized(ref_rad)
        else:
            self._current_ref[:] = 0.0

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

        # Penalize robot-table/obstacle collisions so policy learns to maneuver around structures
        if self._table_geom_id >= 0 and self._ball_geom_id >= 0:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                table_hit = g1 == self._table_geom_id or g2 == self._table_geom_id
                ball_involved = g1 == self._ball_geom_id or g2 == self._ball_geom_id
                if table_hit and not ball_involved:
                    reward -= 0.3
                    break

        success = dist < 0.05
        terminated = success
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {
            "terminated": bool(terminated),
            "is_success": bool(success),
            "distance": dist,
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class AINexWalkToBallEnv(gym.Env):
    """
    Walk around table + reach: legs driven by action groups (walking), policy
    controls right arm for reaching. Uses IMU-like obs (torso orientation,
    angular/linear vel) for balance.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        max_steps: int = 800,
        spawn_radius: float = 0.4,
        spawn_behind_table: bool = True,
        use_action_groups: bool = True,
        action_group_name: str = "forward_one_step",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.max_steps = max_steps
        self.spawn_radius = spawn_radius
        self.spawn_behind_table = spawn_behind_table
        self.use_action_groups = use_action_groups

        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(base_path, "models", "ainex_stable.xml"))

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._action_group_player = None
        if use_action_groups:
            try:
                from scenes.ainex_soccer.action_groups import ActionGroupPlayer, get_actiongroup_path
                path = get_actiongroup_path(action_group_name)
                self._action_group_player = ActionGroupPlayer(path, legs_only=True)
            except Exception as e:
                print(f"Action groups disabled: {e}")
                self.use_action_groups = False

        self._arm_actuators = [14, 15, 16, 17, 18]
        n_act = len(self._arm_actuators)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        n_obs = 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self._gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_tip")
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._free_root = self._has_free_root()
        self._foot_geom_ids = self._find_foot_geoms()
        self._stand_qpos = self._get_stand_qpos()
        self._arm_stand = self._get_arm_stand_targets()

    def _has_free_root(self) -> bool:
        return self.model.njnt > 0 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    def _find_foot_geoms(self) -> list[int]:
        ids = []
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if name and "foot_col" in name:
                ids.append(gid)
        return ids

    def _get_stand_qpos(self) -> np.ndarray:
        """Approximate standing pose - legs slightly bent for stability."""
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[0:7] = np.array([0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])
        stand_offsets = {
            "r_hip_pitch": 0.1, "r_knee": 0.35, "r_ank_pitch": -0.15,
            "l_hip_pitch": 0.1, "l_knee": 0.35, "l_ank_pitch": -0.15,
            "r_sho_pitch": -0.2, "l_sho_pitch": -0.2,
        }
        for j in range(self.model.njnt):
            j_type = int(self.model.jnt_type[j])
            if j_type in (2, 3):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                qadr = int(self.model.jnt_qposadr[j])
                if name and name in stand_offsets:
                    qpos[qadr] = stand_offsets.get(name, 0.0)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.copy()

    def _get_arm_stand_targets(self) -> np.ndarray:
        """Right arm joint targets for neutral pose (from stand_qpos)."""
        low, high = self.model.actuator_ctrlrange.T
        targets = np.zeros(5)
        for i, act_id in enumerate(self._arm_actuators):
            jid = int(self.model.actuator_trnid[act_id, 0])
            qadr = int(self.model.jnt_qposadr[jid])
            targets[i] = self._stand_qpos[qadr]
        return targets

    def _get_obs(self) -> np.ndarray:
        """IMU-like obs: torso up, ang_vel, lin_vel, ball_pos, gripper_pos (all in world frame)."""
        torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_up = torso_mat[:, 2]
        torso_pos = self.data.xpos[self._torso_id]
        torso_lin_vel = self.data.cvel[self._torso_id][3:6]
        torso_ang_vel = self.data.cvel[self._torso_id][0:3]
        ball_pos = self.data.xpos[self._ball_body_id]
        grip_pos = self.data.site_xpos[self._gripper_site_id]
        ball_rel = ball_pos - torso_pos
        grip_rel = grip_pos - torso_pos
        return np.concatenate([
            torso_up, torso_ang_vel, torso_lin_vel,
            ball_rel, grip_rel,
        ]).astype(np.float32)

    def get_end_effector_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._gripper_site_id].copy()

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

    def _spawn_robot(self):
        if not self._free_root:
            return
        if self.spawn_behind_table:
            x = np.random.uniform(-0.25, 0.08)
            y = np.random.uniform(-self.spawn_radius, self.spawn_radius)
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.25, self.spawn_radius)
            x = 0.16 + r * np.cos(angle)
            y = r * np.sin(angle)
        z = 0.6
        self.data.qpos[0:7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=self.data.qpos.dtype)
        self.data.qvel[0:6] = 0.0
        self.data.qpos[7:] = self._stand_qpos[7:]
        self.data.qvel[6:] = 0.0

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

    def _apply_action(self, action: np.ndarray):
        """Legs from action group (walking), arm from policy (reaching)."""
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = 0.0

        if self.use_action_groups and self._action_group_player:
            leg_ctrl = self._action_group_player.get_leg_ctrl(self.model.nu)
            for i in range(12):
                self.data.ctrl[i] = leg_ctrl[i]
            for i in (12, 13):
                self.data.ctrl[i] = self._stand_qpos[self.model.jnt_qposadr[int(self.model.actuator_trnid[i, 0])]]
        else:
            for a in range(12):
                jid = int(self.model.actuator_trnid[a, 0])
                if jid >= 0:
                    qadr = int(self.model.jnt_qposadr[jid])
                    self.data.ctrl[a] = self._stand_qpos[qadr]
            for a in (12, 13):
                jid = int(self.model.actuator_trnid[a, 0])
                if jid >= 0:
                    qadr = int(self.model.jnt_qposadr[jid])
                    self.data.ctrl[a] = self._stand_qpos[qadr]

        for i, act_id in enumerate(self._arm_actuators):
            base = self._arm_stand[i]
            span = 0.6 * (high[act_id] - low[act_id])
            target = base + float(action[i]) * span
            self.data.ctrl[act_id] = np.clip(target, low[act_id], high[act_id])

        for a in range(19, 24):
            jid = int(self.model.actuator_trnid[a, 0])
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.ctrl[a] = self._stand_qpos[qadr]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._reset_ball()
        self._spawn_robot()
        self._snap_feet_to_ground(penetration=0.01)
        if self._action_group_player:
            self._action_group_player.reset()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if self._action_group_player:
            self._action_group_player.advance(self.model.opt.timestep)

        grip_pos = self.data.site_xpos[self._gripper_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        torso_pos = self.data.xpos[self._torso_id]
        torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_up = torso_mat[:, 2]
        ang_vel = self.data.cvel[self._torso_id][0:3]

        dist = float(np.linalg.norm(grip_pos - ball_pos))
        upright = float(torso_up[2])
        ang_vel_mag = float(np.linalg.norm(ang_vel))

        reward = 2.0 * (1.0 / (1.0 + 6.0 * dist))
        reward += 0.5 * max(0, upright)
        reward -= 0.15 * min(ang_vel_mag, 2.0)
        reward -= 0.01 * float(np.mean(np.square(action)))

        if self._table_geom_id >= 0 and self._ball_geom_id >= 0:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                table_hit = g1 == self._table_geom_id or g2 == self._table_geom_id
                ball_involved = g1 == self._ball_geom_id or g2 == self._ball_geom_id
                if table_hit and not ball_involved:
                    reward -= 0.4
                    break

        success = dist < 0.06
        height = float(torso_pos[2])
        terminated = height < 0.18
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {
            "terminated": bool(terminated),
            "is_success": bool(success),
            "distance": dist,
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
