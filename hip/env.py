"""Gymnasium env: limp Ainex + hip-mounted xArm7 (wrist weld), random ball, 7-DoF arm control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


def _model_path() -> str:
    return str(Path(__file__).resolve().parent / "models" / "hip_reach.xml")


def _free_joint_qposadr(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    raise RuntimeError("no free joint (ball) in model")


def _xarm_qpos_slice(model: mujoco.MjModel) -> slice:
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    a = int(model.jnt_qposadr[j1])
    return slice(a, a + 7)


def _xarm_dof_slice(model: mujoco.MjModel) -> slice:
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    a = int(model.jnt_dofadr[j1])
    return slice(a, a + 7)


def _free_joint_dofadr(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_dofadr[j])
    raise RuntimeError("no free joint (ball) in model")


class HipReachEnv(gym.Env):
    """Control only the xArm; Ainex is passive (high damping, no actuators)."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        max_steps: int = 500,
        ball_xyz_low: tuple[float, float, float] = (0.32, -0.18, 0.95),
        ball_xyz_high: tuple[float, float, float] = (0.52, 0.18, 1.22),
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.ball_low = np.array(ball_xyz_low, dtype=np.float64)
        self.ball_high = np.array(ball_xyz_high, dtype=np.float64)
        path = model_path or _model_path()
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._ball_qadr = _free_joint_qposadr(self.model)
        self._xarm_q = _xarm_qpos_slice(self.model)
        self._xarm_dof = _xarm_dof_slice(self.model)
        self._site_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_tip")
        self._site_xarm_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._ball_dofadr = _free_joint_dofadr(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        n_obs = self.model.nq + self.model.nv + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self._step_count = 0
        self._viewer = None

    def freeze_ball_velocity(self) -> None:
        self.data.qvel[self._ball_dofadr : self._ball_dofadr + 6] = 0.0

    def _get_obs(self) -> np.ndarray:
        ball_pos = self.data.qpos[self._ball_qadr : self._ball_qadr + 3].astype(np.float64)
        return np.concatenate(
            [self.data.qpos, self.data.qvel, ball_pos.astype(np.float32)]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[self._ball_qadr : self._ball_qadr + 3] = rng.uniform(
            self.ball_low, self.ball_high
        )
        self.data.qpos[self._ball_qadr + 3 : self._ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self._set_nominal_limp_pose()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _set_nominal_limp_pose(self) -> None:
        """Standing-ish legs and right arm extended forward-ish; xArm near menagerie home."""
        m, d = self.model, self.data

        def setj(name: str, val: float) -> None:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[j]] = val

        for nm, v in (
            ("r_hip_yaw", 0.0),
            ("r_hip_roll", 0.0),
            ("r_hip_pitch", -0.28),
            ("r_knee", 0.55),
            ("r_ank_pitch", -0.25),
            ("r_ank_roll", 0.0),
            ("l_hip_yaw", 0.0),
            ("l_hip_roll", 0.0),
            ("l_hip_pitch", -0.28),
            ("l_knee", 0.55),
            ("l_ank_pitch", -0.25),
            ("l_ank_roll", 0.0),
            ("head_pan", 0.0),
            ("head_tilt", 0.0),
            ("r_sho_pitch", 0.25),
            ("r_sho_roll", -0.35),
            ("r_el_pitch", -0.95),
            ("r_el_yaw", 0.15),
            ("r_gripper", 0.0),
            ("l_sho_pitch", 0.1),
            ("l_sho_roll", 0.35),
            ("l_el_pitch", -0.5),
            ("l_el_yaw", 0.0),
            ("l_gripper", 0.0),
        ):
            setj(nm, v)
        xhome = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], dtype=np.float64)
        d.qpos[self._xarm_q] = xhome
        d.ctrl[:] = np.clip(xhome, self._ctrl_low, self._ctrl_high)

    def step(self, action: np.ndarray):
        a = np.clip(action.astype(np.float64), -1.0, 1.0)
        self.data.ctrl[:] = self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        mujoco.mj_step(self.model, self.data)
        self.freeze_ball_velocity()
        self._step_count += 1
        obs = self._get_obs()
        mujoco.mj_forward(self.model, self.data)
        err = self.data.qpos[self._ball_qadr : self._ball_qadr + 3] - self.data.site_xpos[
            self._site_ee
        ]
        reward = -float(np.linalg.norm(err))
        terminated = bool(np.linalg.norm(err) < 0.05)
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def apply_ik_to_ball(self, gain: float = 0.22, pos_step: float = 0.035) -> None:
        """Jacobian-transpose step driving xArm ``attachment_site`` toward the ball.

        MuJoCo's site Jacobian for ``r_gripper_tip`` does not depend on xArm
        dofs under this closed chain, so IK uses ``attachment_site`` on link7;
        the wrist weld pulls the Ainex hand along.
        """
        mujoco.mj_forward(self.model, self.data)
        ball = self.data.qpos[self._ball_qadr : self._ball_qadr + 3]
        att = self.data.site_xpos[self._site_xarm_ee]
        err = ball - att
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._site_xarm_ee)
        J = jacp[:, self._xarm_dof]
        dq = (J.T @ err) * gain
        q = self.data.qpos[self._xarm_q].copy()
        q_new = q + np.clip(dq, -pos_step, pos_step)
        for i, name in enumerate(f"joint{i}" for i in range(1, 8)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.model.jnt_range[jid]
            q_new[i] = np.clip(q_new[i], lo, hi)
        self.data.ctrl[:] = np.clip(q_new, self._ctrl_low, self._ctrl_high)
