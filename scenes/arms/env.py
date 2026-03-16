"""
Arm-only reach env: you upload only the arm XML. The scene (floor + ball + arm)
is composed at load time. Supports single-arm and multi-arm (e.g. ALOHA).
ball_mode: "shared" = 1 ball, all arms reach it; "per_arm" = N balls, arm i reaches ball_i.
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
    get_arm_info,
    compute_reach_from_model,
    compute_reach_min_from_model,
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
        ball_mode: str = "shared",
        fix_arm_indices: list[int] | None = None,
        reward_time_penalty: float = 0.0005,
        reward_smoothness: float = 0.02,
        reward_move_away_penalty: float = 0.5,
        reward_style: str = "z1",
        reach_min_mode: str = "auto",
        reach_min_fraction: float | None = None,
        reach_min_floor: float | None = None,
        ee_priority_scale: bool = True,
        ctrl_blend_new: float | None = None,
        initial_pose: str = "home",
        initial_keyframe: str | None = None,
        joint_limit_margin_penalty: float | None = None,
        metrics_csv_path: str | None = None,
        disable_logging: bool = False,
    ):
        """
        Reach env: upload only the arm. Scene (floor + ball + arm) is composed at load time.

        - model_path: optional full scene XML; if set, used as-is. Otherwise scene is composed.
        - arm_id: registry key (e.g. "z1", "aloha"). Composed scene uses this arm.
        - ee_site_name: end-effector site (single-arm); auto-resolved if None. Ignored if arm has multiple EEs.
        - reach_min / reach_max: ball sampling radius; from registry or discovery if not set.
        - ball_mode: "shared" (1 ball, all arms reach it) or "per_arm" (N balls, arm i reaches ball_i).
        - reward_style: "z1" = match original Z1 industrial (0.05 action penalty, 50/50 ctrl blend,
          no time/jerk/move-away penalties). "arms" = time/smoothness/move-away penalties, 70/30 blend.
        - reach_min_mode: "auto" = infer reach_min from arm model; "registry" = use fraction/floor.
        - ee_priority_scale: if True, scale actions so base joints have higher gain, EE joints lower (coarse-then-fine reaching).
        - ctrl_blend_new: fraction of new ctrl per step (lower = smoother). None = use reward_style default.
        - initial_pose: "home" = use keyframe/qpos0; "random" = sample qpos in joint limits (spreads bimanual arms).
        - initial_keyframe: override keyframe name for reset (e.g. "spread"). None = use registry home.
        - joint_limit_margin_penalty: if set, penalize joints within margin of limits (avoids self-folding).
        """
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self._arm_id = arm_id
        self._ball_mode = (ball_mode or "shared").lower()
        if self._ball_mode not in ("shared", "per_arm"):
            self._ball_mode = "shared"
        self._fix_arm_indices = set(fix_arm_indices or [])
        self._reward_time_penalty = float(reward_time_penalty)
        self._reward_smoothness = float(reward_smoothness)
        self._reward_move_away_penalty = float(reward_move_away_penalty)
        self._reward_style = (reward_style or "z1").lower()
        self._reach_min_mode = (reach_min_mode or "auto").lower()
        self._reach_min_fraction = reach_min_fraction
        self._reach_min_floor = reach_min_floor
        self._ee_priority_scale = bool(ee_priority_scale)
        self._ctrl_blend_new = ctrl_blend_new
        self._initial_pose = (initial_pose or "home").lower()
        self._initial_keyframe = initial_keyframe
        self._joint_limit_margin_penalty = float(joint_limit_margin_penalty) if joint_limit_margin_penalty is not None else None

        info = get_arm_info(arm_id)
        n_arms = len(info["ee_sites"]) if info and info.get("ee_sites") else 1
        ball_count = n_arms if (n_arms > 1 and self._ball_mode == "per_arm") else 1

        resolved = resolve_model_path(arm_id, model_path, ball_count)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Model not found: {resolved}")
        print("Loading model from:", resolved)

        self.model = mujoco.MjModel.from_xml_path(resolved)
        self.data = mujoco.MjData(self.model)

        info = get_arm_info(arm_id)
        if info and info.get("ee_sites"):
            self._ee_site_names = list(info["ee_sites"])
        else:
            ee_name = resolve_ee_site_name(self.model, arm_id, ee_site_name)
            self._ee_site_names = [ee_name]
        self._ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in self._ee_site_names
        ]
        self._n_arms = len(self._ee_site_names)
        self._actuator_groups = info.get("actuator_groups") if info else None
        if not self._actuator_groups or len(self._actuator_groups) != self._n_arms:
            self._actuator_groups = [list(range(self.model.nu))]  # fallback: all actuators
        self._ball_body_ids = []
        for i in range(ball_count):
            name = f"ball_{i}" if ball_count > 1 else "ball"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._ball_body_ids.append(bid)

        cfg = get_arm_config(arm_id)
        if reach_max is not None:
            _reach_max = reach_max
        elif info and info.get("reach_max") is not None:
            _reach_max = info["reach_max"]
        elif cfg and cfg.reach_max is not None:
            _reach_max = cfg.reach_max
        else:
            _reach_max = compute_reach_from_model(
                self.model, self.data, self._ee_site_ids[0]
            )
            print("Computed reach_max =", round(_reach_max, 3))
        # reach_min: auto = infer from model; registry = from config; manual = fraction/floor
        if self._reach_min_mode == "auto":
            _reach_min = compute_reach_min_from_model(
                self.model, self.data, self._ee_site_ids[0]
            )
            print("Reach min (auto from arm model):", round(_reach_min, 3), "m")
        else:
            _reach_min = reach_min if reach_min is not None else (cfg.reach_min if cfg else 0.08)
            if 0.07 <= _reach_min <= 0.09 and _reach_max > 0:
                _reach_min = max(0.05, 0.15 * _reach_max)
            if self._reach_min_fraction is not None:
                min_allowed = self._reach_min_fraction * _reach_max
                if _reach_min < min_allowed:
                    _reach_min = min_allowed
            if self._reach_min_floor is not None and _reach_min < self._reach_min_floor:
                _reach_min = self._reach_min_floor
        self._reach_min = min(_reach_min, _reach_max * 0.98)
        self._reach_max = _reach_max

        self._home_keyframe_name = (cfg.home_keyframe_name if cfg else "home") or "home"
        self._reset_keyframe = self._initial_keyframe or self._home_keyframe_name
        self._has_home_key = False
        self._has_reset_key = False
        for i in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name == self._home_keyframe_name:
                self._has_home_key = True
            if name == self._reset_keyframe:
                self._has_reset_key = True

        n_act = self.model.nu
        if self._fix_arm_indices:
            act_indices = []
            for i in range(self._n_arms):
                if i not in self._fix_arm_indices and i < len(self._actuator_groups):
                    act_indices.extend(self._actuator_groups[i])
            n_act = len(act_indices) if act_indices else n_act
            self._action_indices = act_indices
        else:
            self._action_indices = list(range(n_act))
        # Per-actuator weights: base-first (proximal higher, distal lower) for coarse-then-fine reaching
        self._action_weights = np.ones(self.model.nu, dtype=np.float32)
        if self._ee_priority_scale and self._actuator_groups:
            for group in self._actuator_groups:
                n = len(group)
                for j, a_idx in enumerate(group):
                    # j=0 (base) -> 1.0, j=n-1 (distal) -> 0.65
                    self._action_weights[a_idx] = 1.0 - 0.35 * (j / max(1, n - 1))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self._action_indices),), dtype=np.float32)
        # obs: qpos, qvel, then per arm: ee_pos(3), ball_pos(3), dist(1), dir_ee_to_ball(3), delta_dist(1)
        n_obs = self.model.nq + self.model.nv + self._n_arms * (3 + 3 + 1 + 3 + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        # Joint limits: (qpos_index, low, high) for limited hinge/slide joints (used for random init and limit penalty)
        self._joint_limit_list: list[tuple[int, float, float]] = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                if self.model.jnt_limited[j]:
                    adr = int(self.model.jnt_qposadr[j])
                    lo, hi = float(self.model.jnt_range[j, 0]), float(self.model.jnt_range[j, 1])
                    self._joint_limit_list.append((adr, lo, hi))

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
        parts = [self.data.qpos, self.data.qvel]
        for i in range(self._n_arms):
            ee_pos = self.data.site_xpos[self._ee_site_ids[i]]
            ball_id = self._ball_body_ids[min(i, len(self._ball_body_ids) - 1)]
            ball_pos = self.data.xpos[ball_id]
            diff = ball_pos - ee_pos
            dist = float(np.linalg.norm(diff))
            dir_to_ball = (diff / dist) if dist > 1e-6 else np.zeros(3, dtype=np.float32)
            prev_d = self.prev_dists[i] if i < len(self.prev_dists) else dist
            delta_d = prev_d - dist  # positive = getting closer
            parts.append(ee_pos)
            parts.append(ball_pos)
            parts.append(np.array([dist], dtype=np.float32))
            parts.append(dir_to_ball.astype(np.float32))
            parts.append(np.array([delta_d], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _set_ball_random_pos(self):
        for i in range(len(self._ball_body_ids)):
            r = np.random.uniform(self._reach_min, self._reach_max)
            theta = np.random.uniform(0, 2 * np.pi)
            x, y, z = r * np.cos(theta), r * np.sin(theta), 0.05
            body_name = f"ball_{i}" if len(self._ball_body_ids) > 1 else "ball"
            self.model.body(body_name).pos[:] = [x, y, z]
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

    def _sample_random_qpos(self):
        """Sample qpos uniformly inside joint limits (80% of range to avoid singularities)."""
        self.data.qpos[:] = self.model.qpos0.copy()
        rng = np.random.default_rng()
        for adr, lo, hi in self._joint_limit_list:
            span = hi - lo
            margin = 0.1 * span
            self.data.qpos[adr] = rng.uniform(lo + margin, hi - margin)
        for i in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[i, 0])
            qadr = int(self.model.jnt_qposadr[j_id])
            self.data.ctrl[i] = self.data.qpos[qadr]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self._episode_metrics = {
            "step_count": 0, "distance_sum": 0.0, "distance_final": 0.0,
            "orientation_sum": 0.0, "action_norm_sum": 0.0, "reward_sum": 0.0, "success": False,
        }
        if self._initial_pose == "random":
            self._sample_random_qpos()
        elif self._has_reset_key:
            key = self.model.key(self._reset_keyframe)
            self.data.qpos[:] = key.qpos
            self.data.ctrl[:] = key.ctrl
        elif self._has_home_key:
            key = self.model.key(self._home_keyframe_name)
            self.data.qpos[:] = key.qpos
            self.data.ctrl[:] = key.ctrl
        else:
            self.data.qpos[:] = self.model.qpos0
            for i in range(self.model.nu):
                j_id = int(self.model.actuator_trnid[i, 0])
                qadr = int(self.model.jnt_qposadr[j_id])
                self.data.ctrl[i] = self.data.qpos[qadr]
        self._set_ball_random_pos()
        mujoco.mj_forward(self.model, self.data)
        self.prev_dists = []
        for i in range(self._n_arms):
            ee_pos = self.data.site_xpos[self._ee_site_ids[i]]
            ball_id = self._ball_body_ids[min(i, len(self._ball_body_ids) - 1)]
            ball_pos = self.data.xpos[ball_id]
            self.prev_dists.append(float(np.linalg.norm(ee_pos - ball_pos)))
        self.step_count = 0
        self._prev_action = None
        return self._get_obs(), {}

    def step(self, action):
        low, high = self.model.actuator_ctrlrange.T
        full_action = np.zeros(self.model.nu, dtype=np.float32)
        for k, a_idx in enumerate(self._action_indices):
            if k < len(action):
                full_action[a_idx] = action[k] * self._action_weights[a_idx]
            else:
                full_action[a_idx] = 0.0
        for arm_i in self._fix_arm_indices:
            if arm_i < len(self._actuator_groups):
                for a_idx in self._actuator_groups[arm_i]:
                    full_action[a_idx] = 0.0  # will use current ctrl for fixed arms
        scaled = low + 0.5 * (full_action + 1.0) * (high - low)
        for arm_i in self._fix_arm_indices:
            if arm_i < len(self._actuator_groups):
                for a_idx in self._actuator_groups[arm_i]:
                    scaled[a_idx] = self.data.ctrl[a_idx]
        # Control blend: explicit ctrl_blend_new, or 50/50 (z1) / 70/30 (arms). Lower blend_new = smoother.
        if self._ctrl_blend_new is not None:
            blend_new = float(self._ctrl_blend_new)
        else:
            blend_new = 0.5 if self._reward_style == "z1" else 0.3
        self.data.ctrl[:] = (1.0 - blend_new) * self.data.ctrl + blend_new * scaled
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        dists = []
        # Z1-style (original industrial): 0.05 action penalty, no time/jerk/move-away penalties.
        if self._reward_style == "z1":
            reward = -0.05 * float(np.sum(np.square(action)))
        else:
            reward = -0.1 * float(np.sum(np.square(action)))
            reward -= self._reward_time_penalty
            if hasattr(self, "_prev_action") and self._prev_action is not None:
                jerk = float(np.sum(np.square(action - self._prev_action)))
                reward -= self._reward_smoothness * jerk
        self._prev_action = action.copy() if hasattr(action, "copy") else np.array(action)
        term_flags = []
        ori_sum = 0.0
        for i in range(self._n_arms):
            ee_pos = self.data.site_xpos[self._ee_site_ids[i]]
            ball_id = self._ball_body_ids[min(i, len(self._ball_body_ids) - 1)]
            ball_pos = self.data.xpos[ball_id]
            dist = float(np.linalg.norm(ee_pos - ball_pos))
            dists.append(dist)
            prev_d = self.prev_dists[i] if i < len(self.prev_dists) else dist
            near = float(np.clip(dist / 0.1, 0.0, 1.0))
            dense = float((1.0 / ((1.0 + 10 * dist) ** 1.5)) * near)
            delta_d = prev_d - dist
            progress = float(np.clip(delta_d, -0.03, 0.03) * near)
            xmat = self.data.site_xmat[self._ee_site_ids[i]].reshape(3, 3)
            target_dir = ball_pos - ee_pos
            n = np.linalg.norm(target_dir)
            if n > 0:
                target_dir /= n
            ori = float(np.dot(xmat[:, 0], target_dir))
            ori_sum += ori
            reward += 3.0 * dense + 1.5 * progress + 0.45 * ori
            # Same near-but-not-reached penalty as Z1 industrial
            if dist < 0.1 and dist >= 0.05:
                reward -= 0.01
            # Move-away penalty: encourages policy to correct back toward ball (both z1 and arms styles)
            if delta_d < 0:
                penalty = self._reward_move_away_penalty if self._reward_style != "z1" else 0.15
                reward -= penalty * min(-delta_d, 0.05) * near
            term_flags.append(dist < 0.05)
            if dist < 0.05:
                reward += 550.0 / self._n_arms
        # Joint-limit margin penalty: discourages self-folding / singularities
        if self._joint_limit_margin_penalty is not None and self._joint_limit_margin_penalty > 0 and self._joint_limit_list:
            margin = 0.1
            for adr, lo, hi in self._joint_limit_list:
                q = float(self.data.qpos[adr])
                span = hi - lo
                if span <= 0:
                    continue
                t = (q - lo) / span
                if t < margin:
                    reward -= self._joint_limit_margin_penalty * (1.0 - t / margin)
                elif t > 1.0 - margin:
                    reward -= self._joint_limit_margin_penalty * (t - (1.0 - margin)) / margin
        self.prev_dists = dists
        # shared: any arm reaches; per_arm: all arms reach
        if self._ball_mode == "shared":
            terminated = any(term_flags)
        else:
            terminated = all(term_flags)
        truncated = self.step_count >= 1000
        if hasattr(self, "_episode_metrics"):
            self._episode_metrics["step_count"] += 1
            self._episode_metrics["distance_sum"] += np.mean(dists)
            self._episode_metrics["distance_final"] = float(np.mean(dists))
            self._episode_metrics["orientation_sum"] += ori_sum / max(1, self._n_arms)
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
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except RuntimeError as e:
                    if "mjpython" in str(e).lower() and "macos" in str(e).lower():
                        raise RuntimeError(
                            "On macOS, run the viewer with mjpython instead of python:\n"
                            "  mjpython -m scenes.arms.training.run_simulation --model <path> --arm-id <arm_id>"
                        ) from e
                    raise
            if self.viewer is not None:
                self.viewer.sync()

    def close(self):
        if hasattr(self, "_episode_metrics") and self._episode_metrics.get("step_count", 0) > 0:
            self._log_episode_metrics()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
