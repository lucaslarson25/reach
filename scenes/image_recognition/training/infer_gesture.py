"""
Inference script for gesture recognition.
Loads trained model and performs real-time gesture classification.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
import csv
import time
from datetime import datetime
from collections import deque

# Import model
import sys
import os
# Add parent directory to path to import models
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.gesture_classifier import create_model


class GestureInference:
    """Real-time gesture inference using MediaPipe and trained model."""
    
    def __init__(self, model_path, device='auto'):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model_type = checkpoint.get('model_type', 'lstm')
        model_kwargs = checkpoint.get('model_kwargs', {})
        self.model = create_model(model_type, **model_kwargs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load scaler and other metadata
        self.scaler = checkpoint.get('scaler')
        self.gesture_to_idx = checkpoint.get('gesture_to_idx', {
            'handshake': 0, 'fist_bump': 1, 'high_five': 2
        })
        self.idx_to_gesture = {v: k for k, v in self.gesture_to_idx.items()}
        self.max_length = checkpoint.get('max_length')
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        
        # Buffer for sequence
        self.sequence_buffer = deque(maxlen=self.max_length if self.max_length else 120)
        self.prediction_history = deque(maxlen=10)
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)  # Track last 30 frames
        self.last_time = time.time()
        
        # CSV logging
        self.csv_log = None
        self.csv_writer = None
        self.last_logged_gesture = None
        self.last_logged_time = 0
        self.min_log_interval = 0.5  # Log same gesture max once per 0.5 seconds
        
    def extract_keypoints(self, image):
        """Extract hand and arm pose keypoints from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = {}
        
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
        
        keypoints['hands'] = hand_keypoints[:2]
        
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
        
        keypoints['pose'] = pose_keypoints
        
        # Combine features
        feature_vector = []
        feature_vector.extend(keypoints['hands'][0])
        feature_vector.extend(keypoints['hands'][1])
        feature_vector.extend(keypoints['pose'])
        
        return np.array(feature_vector, dtype=np.float32), hand_results, pose_results
    
    def predict(self, sequence):
        """Predict gesture from sequence."""
        if len(sequence) < 10:  # Need minimum frames
            return None, None
        
        # Normalize
        if self.scaler is not None:
            sequence_normalized = []
            for frame in sequence:
                frame_normalized = self.scaler.transform(frame.reshape(1, -1))
                sequence_normalized.append(frame_normalized[0])
            sequence = np.array(sequence_normalized)
        else:
            sequence = np.array(sequence)
        
        # Pad or truncate if needed
        if self.max_length is not None:
            if len(sequence) < self.max_length:
                pad_length = self.max_length - len(sequence)
                padding = np.zeros((pad_length, sequence.shape[-1]))
                sequence = np.vstack([sequence, padding])
            else:
                sequence = sequence[:self.max_length]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([len(sequence)]).to(self.device)
        
        # Predict
        with torch.no_grad():
            if hasattr(self.model, 'lstm') and self.max_length is None:
                outputs = self.model(sequence_tensor, length_tensor)
            else:
                outputs = self.model(sequence_tensor)
            
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_score = confidence.item()
            probabilities_np = probabilities.cpu().numpy()[0]
        
        return predicted_idx, confidence_score, probabilities_np
    
    def draw_landmarks(self, image, hand_results, pose_results):
        """Draw MediaPipe landmarks on image."""
        annotated_image = image.copy()
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        return annotated_image
    
    def setup_csv_logging(self, log_file=None):
        """Setup CSV logging for gesture events."""
        if log_file is None:
            log_dir = Path('logs/gesture_events')
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gesture_events_{timestamp}.csv"
        
        self.csv_log = log_file
        self.csv_file = open(log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'gesture', 'confidence', 'fps'])
        print(f"Logging gesture events to: {log_file}")
    
    def log_gesture_event(self, gesture_name, confidence, fps):
        """Log gesture event to CSV if it's a new detection."""
        current_time = time.time()
        
        # Only log if it's a different gesture or enough time has passed
        if (gesture_name != self.last_logged_gesture or 
            current_time - self.last_logged_time >= self.min_log_interval):
            
            if self.csv_writer:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.csv_writer.writerow([timestamp, gesture_name, f"{confidence:.4f}", f"{fps:.2f}"])
                self.csv_file.flush()  # Ensure immediate write
            
            self.last_logged_gesture = gesture_name
            self.last_logged_time = current_time
    
    def calculate_fps(self):
        """Calculate and return current FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_counter.append(fps)
            avg_fps = np.mean(self.fps_counter) if self.fps_counter else fps
            return avg_fps
        return 0.0
    
    def run(self, log_file=None):
        """Run real-time inference."""
        # Setup CSV logging
        self.setup_csv_logging(log_file)
        
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\nGesture Recognition Inference")
        print("Press 'q' to quit")
        print("Press 'c' to clear buffer")
        print(f"Logging to: {self.csv_log}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Extract keypoints
                keypoints, hand_results, pose_results = self.extract_keypoints(frame)
                
                # Add to buffer
                self.sequence_buffer.append(keypoints)
                
                # Predict if buffer has enough frames
                predicted_gesture = None
                predicted_confidence = 0.0
                
                if len(self.sequence_buffer) >= 10:
                    predicted_idx, confidence, probabilities = self.predict(list(self.sequence_buffer))
                    
                    if predicted_idx is not None and confidence > 0.5:
                        gesture_name = self.idx_to_gesture.get(predicted_idx, 'unknown')
                        predicted_gesture = gesture_name
                        predicted_confidence = confidence
                        self.prediction_history.append((gesture_name, confidence))
                        
                        # Log gesture event to CSV
                        self.log_gesture_event(gesture_name, confidence, fps)
                        
                        # Draw prediction
                        cv2.putText(frame, f"Gesture: {gesture_name}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No gesture detected", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Collecting frames... ({len(self.sequence_buffer)})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw landmarks
                annotated = self.draw_landmarks(frame, hand_results, pose_results)
                
                # Display FPS
                fps_color = (0, 255, 0) if fps >= 15 else (0, 0, 255)  # Green if ≥15 FPS, red otherwise
                cv2.putText(annotated, f"FPS: {fps:.1f}", 
                          (annotated.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                
                # Show prediction history
                if len(self.prediction_history) > 0:
                    recent = list(self.prediction_history)[-3:]
                    y_offset = 100
                    for gesture, conf in recent:
                        cv2.putText(annotated, f"{gesture}: {conf:.2f}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
                
                cv2.imshow('Gesture Recognition', annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.sequence_buffer.clear()
                    self.prediction_history.clear()
                    print("Buffer cleared")
        
        finally:
            if self.csv_file:
                self.csv_file.close()
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession complete. Logged events saved to: {self.csv_log}")


def main():
    parser = argparse.ArgumentParser(description='Gesture recognition inference')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (default: gesture_classifier.pth or best_gesture_classifier.pt)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to CSV log file (default: auto-generated in logs/gesture_events/)')
    
    args = parser.parse_args()
    
    # Try to find model if not specified
    if args.model_path is None:
        model_dir = Path('scenes/image_recognition/models')
        pth_path = model_dir / 'gesture_classifier.pth'
        pt_path = model_dir / 'best_gesture_classifier.pt'
        
        if pth_path.exists():
            args.model_path = str(pth_path)
        elif pt_path.exists():
            args.model_path = str(pt_path)
        else:
            print("Error: No model found. Please train a model first using train_gesture_classifier.py")
            return
    
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train_gesture_classifier.py")
        return
    
    inference = GestureInference(args.model_path, args.device)
    inference.run(log_file=args.log_file)


if __name__ == '__main__':
    main()

