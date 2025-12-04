"""
Data collection script for gesture recognition.
Uses MediaPipe to extract hand/arm pose keypoints and saves sequences for training.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# Gesture classes
GESTURES = ['handshake', 'fist_bump', 'high_five']

class GestureDataCollector:
    def __init__(self, data_dir='scenes/image_recognition/data', min_samples_per_gesture=100):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples_per_gesture = min_samples_per_gesture
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Hand and pose detectors
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
        
        self.current_sequence = []
    
    def get_sample_count(self, gesture_class):
        """Get current sample count for a gesture class."""
        gesture_dir = self.data_dir / gesture_class
        if not gesture_dir.exists():
            return 0
        return len(list(gesture_dir.glob('*.json')))
        self.recording = False
        
    def extract_keypoints(self, image):
        """Extract hand and arm pose keypoints from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = {}
        
        # Hand keypoints (up to 2 hands)
        hand_results = self.hands.process(image_rgb)
        hand_keypoints = []
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Extract 21 hand landmarks (x, y, z for each)
                hand_kp = []
                for landmark in hand_landmarks.landmark:
                    hand_kp.extend([landmark.x, landmark.y, landmark.z])
                hand_keypoints.append(hand_kp)
        
        # Pad to 2 hands (42 landmarks * 3 coords = 126 features per hand)
        while len(hand_keypoints) < 2:
            hand_keypoints.append([0.0] * 63)  # 21 landmarks * 3 coords
        
        keypoints['hands'] = hand_keypoints[:2]  # Keep only first 2 hands
        
        # Pose keypoints (arm-related: shoulders, elbows, wrists)
        pose_results = self.pose.process(image_rgb)
        pose_keypoints = []
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Extract arm-related keypoints
            # Left arm: shoulder (11), elbow (13), wrist (15)
            # Right arm: shoulder (12), elbow (14), wrist (16)
            arm_indices = [11, 13, 15, 12, 14, 16]  # MediaPipe pose indices
            
            for idx in arm_indices:
                landmark = landmarks[idx]
                pose_keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # If no pose detected, pad with zeros
        if len(pose_keypoints) == 0:
            pose_keypoints = [0.0] * 18  # 6 keypoints * 3 coords
        
        keypoints['pose'] = pose_keypoints
        
        # Combine all keypoints into a single feature vector
        # Format: [hand1 (63), hand2 (63), pose_arms (18)] = 144 features
        feature_vector = []
        feature_vector.extend(keypoints['hands'][0])  # First hand
        feature_vector.extend(keypoints['hands'][1])  # Second hand
        feature_vector.extend(keypoints['pose'])      # Arm pose
        
        return np.array(feature_vector, dtype=np.float32), hand_results, pose_results
    
    def draw_landmarks(self, image, hand_results, pose_results):
        """Draw MediaPipe landmarks on image for visualization."""
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
    
    def save_sequence(self, gesture_class, sequence):
        """Save a gesture sequence to file."""
        gesture_dir = self.data_dir / gesture_class
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = gesture_dir / f"{gesture_class}_{timestamp}.json"
        
        data = {
            'gesture': gesture_class,
            'sequence': sequence.tolist(),
            'num_frames': len(sequence),
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved sequence: {filename} ({len(sequence)} frames)")
        return filename
    
    def collect_gesture(self, gesture_class, min_frames=30, max_frames=120):
        """Collect a single gesture sequence."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        print(f"\nCollecting '{gesture_class}' gesture")
        print("Press SPACE to start/stop recording")
        print("Press 'q' to quit")
        print("Press 's' to save current sequence")
        
        sequence = []
        recording = False
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract keypoints
                keypoints, hand_results, pose_results = self.extract_keypoints(frame)
                
                # Draw landmarks
                annotated = self.draw_landmarks(frame, hand_results, pose_results)
                
                # Recording logic
                if recording:
                    sequence.append(keypoints)
                    frame_count += 1
                    cv2.putText(annotated, f"Recording... ({frame_count} frames)", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if frame_count >= max_frames:
                        recording = False
                        print(f"Auto-stopped at {max_frames} frames")
                else:
                    cv2.putText(annotated, "Press SPACE to record", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display gesture class and sample count
                sample_count = self.get_sample_count(gesture_class)
                cv2.putText(annotated, f"Gesture: {gesture_class}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated, f"Samples: {sample_count}/{self.min_samples_per_gesture}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (0, 255, 0) if sample_count >= self.min_samples_per_gesture else (0, 165, 255), 2)
                
                # Display sequence length if exists
                if len(sequence) > 0:
                    cv2.putText(annotated, f"Sequence: {len(sequence)} frames", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Gesture Collection', annotated)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Spacebar
                    if not recording:
                        recording = True
                        sequence = []
                        frame_count = 0
                        print("Recording started...")
                    else:
                        recording = False
                        print(f"Recording stopped. Collected {len(sequence)} frames")
                elif key == ord('s'):  # Save
                    if len(sequence) >= min_frames:
                        sequence_array = np.array(sequence)
                        self.save_sequence(gesture_class, sequence_array)
                        sequence = []
                        frame_count = 0
                        sample_count = self.get_sample_count(gesture_class)
                        print(f"Sequence saved! Total samples for {gesture_class}: {sample_count}")
                        if sample_count >= self.min_samples_per_gesture:
                            print(f"✓ {gesture_class} has reached minimum of {self.min_samples_per_gesture} samples!")
                    else:
                        print(f"Sequence too short! Need at least {min_frames} frames, got {len(sequence)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return sequence


def main():
    """Main function for data collection."""
    min_samples = 100
    collector = GestureDataCollector(min_samples_per_gesture=min_samples)
    
    print("=" * 60)
    print("Gesture Data Collection System")
    print("=" * 60)
    print(f"\nTarget: ≥{min_samples} samples per gesture (≥{min_samples * 3} total)")
    print("\nAvailable gestures:")
    for i, gesture in enumerate(GESTURES, 1):
        count = collector.get_sample_count(gesture)
        status = "✓" if count >= min_samples else f"({count}/{min_samples})"
        print(f"  {i}. {gesture} {status}")
    print("\nPress 'q' to quit at any time")
    
    while True:
        print("\n" + "=" * 60)
        print("Select gesture to collect:")
        for i, gesture in enumerate(GESTURES, 1):
            print(f"  {i}. {gesture}")
        print("  0. Quit")
        
        try:
            choice = input("\nEnter choice (1-3, or 0 to quit): ").strip()
            
            if choice == '0':
                break
            
            gesture_idx = int(choice) - 1
            if 0 <= gesture_idx < len(GESTURES):
                gesture_class = GESTURES[gesture_idx]
                collector.collect_gesture(gesture_class)
                
                # Show updated counts
                print("\n" + "=" * 60)
                print("Current sample counts:")
                total = 0
                for gesture in GESTURES:
                    count = collector.get_sample_count(gesture)
                    total += count
                    status = "✓" if count >= min_samples else ""
                    print(f"  {gesture}: {count} {status}")
                print(f"  Total: {total} samples")
                if total >= min_samples * 3:
                    print(f"\n✓ Collection complete! You have ≥{min_samples * 3} total samples.")
            else:
                print("Invalid choice!")
        
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break
    
    # Final summary
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("\nFinal sample counts:")
    total = 0
    for gesture in GESTURES:
        count = collector.get_sample_count(gesture)
        total += count
        status = "✓" if count >= min_samples else f"(need {min_samples - count} more)"
        print(f"  {gesture}: {count} {status}")
    print(f"  Total: {total} samples")
    if total >= min_samples * 3:
        print(f"\n✓ Ready for training! You have ≥{min_samples * 3} total samples.")
    else:
        print(f"\n⚠ Need at least {min_samples * 3} total samples for training.")


if __name__ == '__main__':
    main()

