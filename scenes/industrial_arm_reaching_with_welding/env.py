import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import csv


class Z1WeldingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, model_path: str | None = None, metrics_csv_path: str | None = None, disable_logging: bool = False):
        """
        Welding environment with torch alignment tracking.
        - new optional `model_path`. If None, use the scene's default z1scene.xml
        - new optional `metrics_csv_path`. If None, defaults to logs/episode_metrics.csv
        - new optional `disable_logging`. If True, disables CSV logging (default: False)
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

        # Observation: qpos + qvel + ee_pos (3) + ball_pos (3) + torch_dir (3) + seam_dir (3)
        n_obs = self.model.nq + self.model.nv + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        # Initialize CSV logging (enabled by default unless explicitly disabled)
        self.disable_logging = disable_logging
        if not disable_logging:
            if metrics_csv_path is None:
                # Default to logs/episode_metrics_welding.csv relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                metrics_csv_path = os.path.join(project_root, "logs", "episode_metrics_welding.csv")
            self.metrics_csv_path = metrics_csv_path
        else:
            self.metrics_csv_path = None
        self.episode_count = 0
        self._csv_initialized = False

    # ---------- helpers ----------
    def _get_torch_site_id(self):
        """Get torch site ID, with fallback to eetip if torch site doesn't exist."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torchtip")
        except:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")

    def _get_seam_body_id(self):
        """Get seam/target body ID, with fallback to ball if seam doesn't exist."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seam")
        except:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    def _get_obs(self) -> np.ndarray:
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = self._get_seam_body_id()
        ball_pos = self.data.xpos[ball_id]
        
        # Torch direction (from torch site orientation)
        torch_id = self._get_torch_site_id()
        torch_xmat = self.data.site_xmat[torch_id].reshape(3, 3)
        torch_dir = torch_xmat[:, 0]  # Forward direction of torch
        
        # Seam direction (normalized direction along seam, default to Z-up if not specified)
        # In real welding, this would come from the seam geometry
        seam_dir = np.array([0.0, 0.0, 1.0])  # Default: vertical seam
        
        # Ensure float32 to avoid MPS float64 issues
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, ball_pos, torch_dir, seam_dir]).astype(np.float32)

    def _set_ball_random_pos(self):
        x = np.random.uniform(0.2, 0.3)
        y = np.random.uniform(0.2, 0.3) * np.random.choice([1, -1])
        z = 0.05
        seam_id = self._get_seam_body_id()
        self.model.body(seam_id).pos[:] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    def _compute_torch_alignment(self):
        """
        Compute torch alignment metric: angle between torch direction and target/seam direction.
        Returns cosine similarity (1.0 = perfect alignment, -1.0 = opposite).
        """
        torch_id = self._get_torch_site_id()
        torch_xmat = self.data.site_xmat[torch_id].reshape(3, 3)
        torch_dir = torch_xmat[:, 0]  # Forward direction of torch
        
        # Target direction: from torch tip to target/seam
        torch_pos = self.data.site_xpos[torch_id]
        seam_id = self._get_seam_body_id()
        seam_pos = self.data.xpos[seam_id]
        target_dir = seam_pos - torch_pos
        target_norm = np.linalg.norm(target_dir)
        if target_norm > 0:
            target_dir = target_dir / target_norm
        else:
            target_dir = np.array([0.0, 0.0, -1.0])  # Default downward
        
        # Cosine of angle between torch direction and target direction
        alignment = float(np.dot(torch_dir, target_dir))
        return alignment

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
                               'torch_alignment_avg', 'torch_alignment_final', 'action_norm_avg', 
                               'reward_sum', 'success'])
        
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
        torch_alignment_avg = metrics['torch_alignment_sum'] / steps
        action_norm_avg = metrics['action_norm_sum'] / steps
        
        # Write to CSV
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                distance_avg,
                metrics['distance_final'],
                orientation_avg,
                torch_alignment_avg,
                metrics['torch_alignment_final'],
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
            'torch_alignment_sum': 0.0,
            'torch_alignment_final': 0.0,
            'action_norm_sum': 0.0,
            'reward_sum': 0.0,
            'success': False
        }
        
        # home keyframe if present
        self.data.qpos[:] = self.model.key("home").qpos
        self._set_ball_random_pos()
        mujoco.mj_forward(self.model, self.data)

        # initialize previous distance
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ee_pos = self.data.site_xpos[ee_id]
        ball_id = self._get_seam_body_id()
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
        ball_id = self._get_seam_body_id()
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

        # Torch alignment metric (welding-specific)
        torch_alignment = self._compute_torch_alignment()
        
        # Include torch alignment in reward (optional, can be tuned)
        torch_alignment_reward = 0.3 * torch_alignment

        action_penalty = float(0.05 * np.sum(np.square(action)))

        reward = 3.0 * dense_reward + 1.5 * progress + 0.45 * orientation_reward + torch_alignment_reward - action_penalty

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
            self._episode_metrics['torch_alignment_sum'] += torch_alignment
            self._episode_metrics['torch_alignment_final'] = torch_alignment
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

