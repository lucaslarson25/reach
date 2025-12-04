"""
Integration of gesture recognition with MuJoCo simulation.
Renders camera frames from MuJoCo, processes through gesture pipeline,
and maps gestures to robot control actions.
"""

import numpy as np
import mujoco
import cv2
from collections import deque
from pathlib import Path
import sys
import os

# Add paths for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

from scenes.industrial_arm_reaching.env import Z1ReachEnv
from training.infer_from_keypoints import GestureInferenceFromKeypoints
import mediapipe as mp


class GestureControlledRobot:
    """
    Integrates gesture recognition with MuJoCo robot control.
    Renders camera frames, processes through MediaPipe and classifier,
    and maps gestures to robot actions.
    """
    
    def __init__(self, model_path='scenes/image_recognition/models/gesture_classifier.pth',
                 camera_name='head_camera', camera_width=640, camera_height=480):
        """
        Initialize gesture-controlled robot.
        
        Args:
            model_path: Path to trained gesture classifier
            camera_name: Name of camera in MuJoCo model
            camera_width: Camera image width
            camera_height: Camera image height
        """
        # Initialize environment
        self.env = Z1ReachEnv(render_mode=None, disable_logging=True)
        
        # Initialize gesture inference
        try:
            self.gesture_inference = GestureInferenceFromKeypoints(model_path)
        except FileNotFoundError:
            print(f"Warning: Model not found at {model_path}")
            print("Gesture recognition will be disabled. Please train a model first.")
            self.gesture_inference = None
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera setup
        self.camera_name = camera_name
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Try to find camera in model
        try:
            self.camera_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if self.camera_id < 0:
                print(f"Warning: Camera '{camera_name}' not found. Using default camera (0).")
                self.camera_id = 0
        except:
            # If no cameras exist, we'll need to create a default view
            if self.env.model.ncam == 0:
                print("Warning: No cameras found in model. Will use fixed camera view.")
                self.camera_id = None
            else:
                self.camera_id = 0
        
        # Create renderer for camera
        self.renderer = mujoco.Renderer(self.env.model, height=camera_height, width=camera_width)
        
        # Gesture buffer for sequence processing
        self.gesture_buffer = deque(maxlen=30)  # Store last 30 frames
        self.min_frames_for_prediction = 10
        
        # Gesture-to-action mapping
        self.gesture_actions = {
            'handshake': 'approach',
            'fist_bump': 'retreat',
            'high_five': 'execute'
        }
        
        # Current state
        self.current_gesture = None
        self.current_action = None
        self.gesture_confidence = 0.0
        
        # Accuracy tracking
        self.gesture_predictions = []
        self.gesture_ground_truth = []  # For evaluation if provided
        
    def render_camera_frame(self):
        """Render a camera frame from MuJoCo simulation."""
        # Update renderer with current state
        if self.camera_id is not None:
            self.renderer.update_scene(self.env.data, camera_id=self.camera_id)
        else:
            # Use default view if no camera
            self.renderer.update_scene(self.env.data)
            # Set a reasonable camera view
            self.renderer.scene.camera.lookat[:] = [0.25, 0, 0.1]  # Look at workspace
            self.renderer.scene.camera.distance = 1.0
            self.renderer.scene.camera.azimuth = 135
            self.renderer.scene.camera.elevation = -20
        
        # Render to image
        image = self.renderer.render()
        
        # Convert from RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def extract_keypoints_from_image(self, image):
        """Extract MediaPipe keypoints from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Hand keypoints
        hand_results = self.hands.process(image_rgb)
        hand_keypoints = []
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_kp = []
                for landmark in hand_landmarks.landmark:
                    hand_kp.extend([landmark.x, landmark.y, landmark.z])
                hand_keypoints.append(hand_kp)
        
        while len(hand_keypoints) < 2:
            hand_keypoints.append([0.0] * 63)
        
        # Pose keypoints
        pose_results = self.pose.process(image_rgb)
        pose_keypoints = []
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            arm_indices = [11, 13, 15, 12, 14, 16]
            
            for idx in arm_indices:
                landmark = landmarks[idx]
                pose_keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        if len(pose_keypoints) == 0:
            pose_keypoints = [0.0] * 18
        
        # Combine features (144 total)
        feature_vector = []
        feature_vector.extend(hand_keypoints[0])
        feature_vector.extend(hand_keypoints[1])
        feature_vector.extend(pose_keypoints)
        
        return np.array(feature_vector, dtype=np.float32), hand_results, pose_results
    
    def process_gesture(self, image):
        """
        Process image through gesture recognition pipeline.
        
        Returns:
            tuple: (gesture_name, confidence, action_name) or (None, 0.0, None)
        """
        if self.gesture_inference is None:
            return None, 0.0, None
        
        # Extract keypoints
        keypoints, hand_results, pose_results = self.extract_keypoints_from_image(image)
        
        # Add to buffer
        self.gesture_buffer.append(keypoints)
        
        # Predict if buffer has enough frames
        if len(self.gesture_buffer) >= self.min_frames_for_prediction:
            try:
                # Get probabilities
                probabilities = self.gesture_inference.predict_proba(list(self.gesture_buffer))
                gesture_name, confidence = self.gesture_inference.predict(list(self.gesture_buffer))
                
                # Only accept if confidence is high enough
                if confidence >= 0.6:  # Threshold for reliable detection
                    action_name = self.gesture_actions.get(gesture_name, None)
                    self.current_gesture = gesture_name
                    self.current_action = action_name
                    self.gesture_confidence = confidence
                    
                    # Track for accuracy
                    self.gesture_predictions.append((gesture_name, confidence))
                    
                    return gesture_name, confidence, action_name
            except Exception as e:
                print(f"Error in gesture prediction: {e}")
        
        return None, 0.0, None
    
    def gesture_to_action(self, gesture_name):
        """
        Map gesture to robot action.
        
        Args:
            gesture_name: Detected gesture ('handshake', 'fist_bump', 'high_five')
        
        Returns:
            str: Action name ('approach', 'retreat', 'execute')
        """
        return self.gesture_actions.get(gesture_name, None)
    
    def execute_action(self, action_name, obs):
        """
        Execute robot action based on gesture.
        
        Args:
            action_name: Action to execute ('approach', 'retreat', 'execute')
            obs: Current observation from environment
        
        Returns:
            np.ndarray: Action vector for robot
        """
        n_act = self.env.action_space.shape[0]
        action = np.zeros(n_act, dtype=np.float32)
        
        if action_name == 'approach':
            # Move toward target (ball)
            # Extract end effector and ball positions from observation
            # obs = [qpos, qvel, ee_pos(3), ball_pos(3)]
            ee_pos = obs[-6:-3]
            ball_pos = obs[-3:]
            direction = ball_pos - ee_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0.01:
                direction = direction / direction_norm
                # Map direction to joint actions (simplified)
                # Use first few actuators to move in direction
                for i in range(min(3, n_act)):
                    action[i] = direction[i] * 0.5
        
        elif action_name == 'retreat':
            # Move away from target
            ee_pos = obs[-6:-3]
            ball_pos = obs[-3:]
            direction = ee_pos - ball_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0.01:
                direction = direction / direction_norm
                for i in range(min(3, n_act)):
                    action[i] = direction[i] * 0.3
        
        elif action_name == 'execute':
            # Execute reaching task (move toward target with higher intensity)
            ee_pos = obs[-6:-3]
            ball_pos = obs[-3:]
            direction = ball_pos - ee_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0.01:
                direction = direction / direction_norm
                for i in range(min(n_act, 6)):
                    action[i] = direction[i % 3] * 0.7
        
        return action
    
    def get_accuracy(self):
        """Calculate gesture recognition accuracy if ground truth is available."""
        if len(self.gesture_predictions) == 0:
            return None
        
        # If ground truth provided, calculate accuracy
        if len(self.gesture_ground_truth) > 0 and len(self.gesture_ground_truth) == len(self.gesture_predictions):
            correct = sum(1 for (pred, _), truth in zip(self.gesture_predictions, self.gesture_ground_truth) 
                         if pred == truth)
            return correct / len(self.gesture_predictions)
        
        # Otherwise return average confidence as proxy
        if self.gesture_predictions:
            avg_confidence = np.mean([conf for _, conf in self.gesture_predictions])
            return avg_confidence
        
        return None
    
    def reset(self):
        """Reset environment and buffers."""
        obs, _ = self.env.reset()
        self.gesture_buffer.clear()
        self.current_gesture = None
        self.current_action = None
        self.gesture_confidence = 0.0
        return obs
    
    def step(self, use_gesture_control=True):
        """
        Step simulation with optional gesture control.
        
        Args:
            use_gesture_control: If True, use gesture to control robot
        
        Returns:
            tuple: (obs, reward, terminated, truncated, info, gesture_info)
        """
        # Render camera frame
        image = self.render_camera_frame()
        
        # Process gesture
        gesture_name, confidence, action_name = self.process_gesture(image)
        
        # Get action
        obs = self.env._get_obs()
        
        if use_gesture_control and action_name is not None:
            action = self.execute_action(action_name, obs)
        else:
            # Default: no action (or could use a policy)
            action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        gesture_info = {
            'gesture': gesture_name,
            'confidence': confidence,
            'action': action_name
        }
        
        return obs, reward, terminated, truncated, info, gesture_info
    
    def close(self):
        """Close environment and cleanup."""
        self.env.close()
        if hasattr(self, 'renderer'):
            del self.renderer

