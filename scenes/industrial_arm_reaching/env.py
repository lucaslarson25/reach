import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import csv

from scenes.industrial_arm_reaching.arm_registry import (
    resolve_model_path,
    resolve_ee_site_name,
    get_arm_config,
    compute_reach_from_model,
)


class Z1ReachEnv(gym.Env):
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
        Reach env that supports multiple arms (different lengths and DOF).

        - model_path: path to scene XML (floor + ball + arm). If None, uses arm_id or default z1.
        - arm_id: key from arm registry (e.g. "z1", "arm_2link"). Used when model_path is None.
        - ee_site_name: MuJoCo site name for end-effector (tip of arm). Auto-resolved if None.
        - reach_min / reach_max: ball is sampled in [reach_min, reach_max] radius. If reach_max
          is None, it is computed from the model or taken from arm registry.
        - metrics_csv_path / disable_logging: same as before.
        """
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self._arm_id = arm_id

        model_path = resolve_model_path(arm_id, model_path)
        if not os.path.isfile(model_path):
            base_path = os.path.dirname(os.path.abspath(__file__))
            fallback = os.path.normpath(os.path.join(base_path, "models", os.path.basename(model_path)))
            if os.path.isfile(fallback):
                model_path = fallback
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        print("Loading model from:", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._ee_site_name = resolve_ee_site_name(self.model, arm_id, ee_site_name)
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name)
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # Reach bounds for ball sampling
        cfg = get_arm_config(arm_id)
        self._reach_min = reach_min if reach_min is not None else (cfg.reach_min if cfg else 0.08)
        if reach_max is not None:
            self._reach_max = reach_max
        elif cfg and cfg.reach_max is not None:
            self._reach_max = cfg.reach_max
        else:
            self._reach_max = compute_reach_from_model(self.model, self.data, self._ee_site_id)
            print("Computed reach_max =", round(self._reach_max, 3))

        # Home keyframe (optional)
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
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                metrics_csv_path = os.path.join(project_root, "logs", "episode_metrics.csv")
            self.metrics_csv_path = metrics_csv_path
        else:
            self.metrics_csv_path = None
        self.episode_count = 0
        self._csv_initialized = False

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos]).astype(np.float32)

    def _set_ball_random_pos(self):
        # Sample ball in reachable region: horizontal disk [reach_min, reach_max], table height
        r = np.random.uniform(self._reach_min, self._reach_max)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.05
        self.model.body("ball").pos[:] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    # ---------- helpers ----------
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if self._csv_initialized or not self.metrics_csv_path:
            return
        
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(self.metrics_csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        
        # Write header if file doesn't exist
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'distance_avg', 'distance_final', 'orientation_avg', 
                               'action_norm_avg', 'reward_sum', 'success'])
        
        self._csv_initialized = True
    
    def _log_episode_metrics(self):
        """Log episode metrics to CSV file."""
        if self.disable_logging or not self.metrics_csv_path or not hasattr(self, '_episode_metrics'):
            return
        
        self._initialize_csv()
        
        metrics = self._episode_metrics
        steps = metrics['step_count']
        if steps == 0:
            return
        
        # Calculate averages
        distance_avg = metrics['distance_sum'] / steps
        orientation_avg = metrics['orientation_sum'] / steps
        action_norm_avg = metrics['action_norm_sum'] / steps
        
        # Write to CSV
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                distance_avg,
                metrics['distance_final'],
                orientation_avg,
                action_norm_avg,
                metrics['reward_sum'],
                int(metrics['success'])
            ])

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start new episode (metrics will be logged when episode ends in step())
        self.episode_count += 1
        self._episode_metrics = {
            'step_count': 0,
            'distance_sum': 0.0,
            'distance_final': 0.0,
            'orientation_sum': 0.0,
            'action_norm_sum': 0.0,
            'reward_sum': 0.0,
            'success': False
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
        # Scale actions into actuator ctrl range
        low, high = self.model.actuator_ctrlrange.T
        scaled_action = low + 0.5 * (action + 1.0) * (high - low)
        self.data.ctrl[:] = 0.5 * self.data.ctrl + 0.5 * scaled_action

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        max_steps = 1000  # ~10s at 100Hz

        ee_pos = self.data.site_xpos[self._ee_site_id]
        ball_pos = self.data.xpos[self._ball_body_id]
        dist = float(np.linalg.norm(ee_pos - ball_pos))

        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist

        # Reward shaping (unchanged, cast to float to be safe)
        near_goal_scale = float(np.clip(dist / 0.1, 0.0, 1.0))
        dense_reward = float((1.0 / ((1.0 + 10 * dist) ** 1.5)) * near_goal_scale)
        progress = float(np.clip(self.prev_dist - dist, -0.03, 0.03) * near_goal_scale)

        xmat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
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

        # Record metrics for this step (non-intrusive instrumentation)
        if hasattr(self, '_episode_metrics'):
            self._episode_metrics['step_count'] += 1
            self._episode_metrics['distance_sum'] += dist
            self._episode_metrics['distance_final'] = dist
            self._episode_metrics['orientation_sum'] += orientation_reward
            action_norm = float(np.linalg.norm(action))
            self._episode_metrics['action_norm_sum'] += action_norm
            self._episode_metrics['reward_sum'] += reward
            if terminated:
                self._episode_metrics['success'] = True
        
        # Log episode metrics when episode ends (non-intrusive - after all core logic)
        if (terminated or truncated) and hasattr(self, '_episode_metrics') and self._episode_metrics['step_count'] > 0:
            self._log_episode_metrics()

        # Return obs as float32
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"terminated": bool(terminated)}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                # keep default (non-passive) for generic use; mac viewer will be launched externally
                self.viewer = mujoco.viewer.launch(self.model, self.data)
            self.viewer.sync()

    def close(self):
        # Log final episode metrics before closing
        if hasattr(self, '_episode_metrics') and self._episode_metrics['step_count'] > 0:
            self._log_episode_metrics()
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None