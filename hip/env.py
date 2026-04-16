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
    """Control only the xArm; Ainex is passive (high damping, no actuators).

    Reward (each step, roughly): dense distance ``-dist``, progress ``+progress_coef * Δdist``,
    inverse-distance pull ``+inverse_dist_coef / (dist + eps)``, smoothness
    ``-action_smooth_coef * ||a-a_prev||^2``, magnitude ``-action_mag_coef * ||a||^2``,
    optional success bonus, minus clearance penalty when the arm crowds the humanoid arm.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | None = None,
        max_steps: int = 500,
        # Ball mostly in front of the torso (world +X); narrow Y/Z for the assistive reach task.
        ball_xyz_low: tuple[float, float, float] = (0.28, -0.12, 0.28),
        ball_xyz_high: tuple[float, float, float] = (0.52, 0.12, 0.50),
        success_bonus: float = 0.0,
        progress_coef: float = 2.0,
        inverse_dist_coef: float = 0.15,
        inverse_dist_eps: float = 0.07,
        action_smooth_coef: float = 0.04,
        action_mag_coef: float = 0.012,
        torso_clearance_soft_m: float = 0.11,
        torso_clearance_hard_m: float = 0.055,
        torso_violation_penalty: float = 8.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._success_bonus = float(success_bonus)
        self._progress_coef = float(progress_coef)
        self._clear_soft = float(torso_clearance_soft_m)
        self._clear_hard = float(torso_clearance_hard_m)
        self._torso_penalty = float(torso_violation_penalty)
        self._inv_dist_coef = float(inverse_dist_coef)
        self._inv_dist_eps = float(inverse_dist_eps)
        self._smooth_coef = float(action_smooth_coef)
        self._mag_coef = float(action_mag_coef)
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
        # qpos, qvel, (ball - ee) world; ball pose is still in qpos for the free joint.
        n_obs = self.model.nq + self.model.nv + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self._step_count = 0
        self._viewer = None
        self._prev_dist: float = 1.0
        self._prev_action = np.zeros(7, dtype=np.float64)
        self._humanoid_probe_bids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm))
            for nm in ("r_sho_roll_link", "r_el_pitch_link", "r_el_yaw_link")
        )
        self._arm_probe_bids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm))
            for nm in ("link3", "link4", "link5", "link6")
        )

    def freeze_ball_velocity(self) -> None:
        self.data.qvel[self._ball_dofadr : self._ball_dofadr + 6] = 0.0

    def _xarm_ctrl_home(self) -> np.ndarray:
        xhome = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], dtype=np.float64)
        return np.clip(xhome, self._ctrl_low, self._ctrl_high)

    def _arm_humanoid_min_dist(self) -> float:
        """Min COM distance (m) between mid-xArm links and right humanoid arm (not torso COM)."""
        d = self.data
        best = 1e9
        for ab in self._arm_probe_bids:
            pa = d.xpos[ab]
            for hb in self._humanoid_probe_bids:
                ph = d.xpos[hb]
                best = min(best, float(np.linalg.norm(pa - ph)))
        return best

    def safeguard_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Blend ctrl toward a safe home pose when mid-arm links are close to the humanoid arm."""
        mujoco.mj_forward(self.model, self.data)
        c = self._arm_humanoid_min_dist()
        if c >= self._clear_soft:
            return np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        span = max(self._clear_soft - self._clear_hard, 1e-6)
        t = np.clip((c - self._clear_hard) / span, 0.0, 1.0)
        home = self._xarm_ctrl_home()
        out = t * np.clip(ctrl, self._ctrl_low, self._ctrl_high) + (1.0 - t) * home
        return np.clip(out, self._ctrl_low, self._ctrl_high)

    def _get_obs(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        ball_pos = self.data.qpos[self._ball_qadr : self._ball_qadr + 3].astype(np.float64)
        ee = self.data.site_xpos[self._site_ee].astype(np.float64)
        delta = (ball_pos - ee).astype(np.float32)
        return np.concatenate(
            [self.data.qpos, self.data.qvel, delta]
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
        obs = self._get_obs()
        bp = self.data.qpos[self._ball_qadr : self._ball_qadr + 3]
        ee = self.data.site_xpos[self._site_ee]
        self._prev_dist = float(np.linalg.norm(bp - ee))
        self._prev_action[:] = 0.0
        return obs, {}

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
        desired = self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        self.data.ctrl[:] = self.safeguard_ctrl(desired)
        mujoco.mj_step(self.model, self.data)
        self.freeze_ball_velocity()
        self._step_count += 1
        mujoco.mj_forward(self.model, self.data)
        err = self.data.qpos[self._ball_qadr : self._ball_qadr + 3] - self.data.site_xpos[
            self._site_ee
        ]
        dist = float(np.linalg.norm(err))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        inv = self._inv_dist_coef / (dist + self._inv_dist_eps)
        da = a - self._prev_action
        smooth_pen = self._smooth_coef * float(np.dot(da, da))
        mag_pen = self._mag_coef * float(np.dot(a, a))
        self._prev_action = a.copy()
        reward = (
            -dist
            + self._progress_coef * progress
            + inv
            - smooth_pen
            - mag_pen
        )
        c_h = self._arm_humanoid_min_dist()
        if c_h < self._clear_soft:
            reward -= self._torso_penalty * float(self._clear_soft - c_h)
        terminated = bool(dist < 0.05)
        if terminated and self._success_bonus != 0.0:
            reward += self._success_bonus
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"terminated": terminated, "truncated": truncated}

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
        self.data.ctrl[:] = self.safeguard_ctrl(np.clip(q_new, self._ctrl_low, self._ctrl_high))
