"""
Standalone inference function that takes MediaPipe keypoints and outputs gesture class probabilities.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import models
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.gesture_classifier import create_model


class GestureInferenceFromKeypoints:
    """
    Inference class that takes MediaPipe keypoints and outputs gesture probabilities.
    """
    
    def __init__(self, model_path='scenes/image_recognition/models/gesture_classifier.pth', device='auto'):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to trained model (.pth file)
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model checkpoint
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model_type = checkpoint.get('model_type', 'lstm')
        model_kwargs = checkpoint.get('model_kwargs', {})
        self.model = create_model(model_type, **model_kwargs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load metadata
        self.scaler = checkpoint.get('scaler')
        self.gesture_to_idx = checkpoint.get('gesture_to_idx', {
            'handshake': 0, 'fist_bump': 1, 'high_five': 2
        })
        self.idx_to_gesture = {v: k for k, v in self.gesture_to_idx.items()}
        self.max_length = checkpoint.get('max_length')
    
    def predict_proba(self, keypoint_sequence):
        """
        Predict gesture class probabilities from a sequence of keypoints.
        
        Args:
            keypoint_sequence: List or numpy array of keypoint vectors.
                              Each vector should be 144 features:
                              [hand1 (63), hand2 (63), pose_arms (18)]
                              Shape: (sequence_length, 144) or list of (144,) arrays
        
        Returns:
            dict: Dictionary with gesture names as keys and probabilities as values
                  Example: {'handshake': 0.85, 'fist_bump': 0.10, 'high_five': 0.05}
        """
        # Convert to numpy array if needed
        if isinstance(keypoint_sequence, list):
            sequence = np.array(keypoint_sequence, dtype=np.float32)
        else:
            sequence = np.array(keypoint_sequence, dtype=np.float32)
        
        # Ensure correct shape
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)  # Single frame -> (1, 144)
        
        if sequence.shape[-1] != 144:
            raise ValueError(f"Expected 144 features per frame, got {sequence.shape[-1]}")
        
        # Normalize if scaler available
        if self.scaler is not None:
            sequence_normalized = []
            for frame in sequence:
                frame_normalized = self.scaler.transform(frame.reshape(1, -1))
                sequence_normalized.append(frame_normalized[0])
            sequence = np.array(sequence_normalized)
        
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
            probabilities_np = probabilities.cpu().numpy()[0]
        
        # Convert to dictionary
        prob_dict = {}
        for idx, gesture_name in self.idx_to_gesture.items():
            prob_dict[gesture_name] = float(probabilities_np[idx])
        
        return prob_dict
    
    def predict(self, keypoint_sequence):
        """
        Predict the most likely gesture class from a sequence of keypoints.
        
        Args:
            keypoint_sequence: List or numpy array of keypoint vectors (same as predict_proba)
        
        Returns:
            tuple: (gesture_name, confidence_score)
                   Example: ('handshake', 0.85)
        """
        prob_dict = self.predict_proba(keypoint_sequence)
        gesture_name = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[gesture_name]
        return gesture_name, confidence


def infer_from_keypoints(keypoint_sequence, model_path='scenes/image_recognition/models/gesture_classifier.pth'):
    """
    Convenience function for inference from keypoints.
    
    Args:
        keypoint_sequence: List or numpy array of keypoint vectors.
                          Each vector should be 144 features from MediaPipe.
        model_path: Path to trained model
    
    Returns:
        dict: Dictionary with gesture names as keys and probabilities as values
    """
    inference = GestureInferenceFromKeypoints(model_path)
    return inference.predict_proba(keypoint_sequence)


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference from MediaPipe keypoints')
    parser.add_argument('--model-path', type=str,
                        default='scenes/image_recognition/models/gesture_classifier.pth',
                        help='Path to trained model')
    parser.add_argument('--keypoints-file', type=str, default=None,
                        help='JSON file containing keypoint sequence (for testing)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = GestureInferenceFromKeypoints(args.model_path)
    
    if args.keypoints_file:
        # Load keypoints from file (for testing)
        import json
        with open(args.keypoints_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'sequence' in data:
                sequence = np.array(data['sequence'], dtype=np.float32)
            else:
                # Assume data is directly the sequence
                sequence = np.array(data, dtype=np.float32)
        
        # Predict
        probabilities = inference.predict_proba(sequence)
        gesture, confidence = inference.predict(sequence)
        
        print(f"\nPredicted gesture: {gesture} (confidence: {confidence:.4f})")
        print("\nAll probabilities:")
        for gesture_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {gesture_name}: {prob:.4f}")
    else:
        print("Gesture Inference from Keypoints")
        print("=" * 60)
        print("\nUsage:")
        print("  from infer_from_keypoints import infer_from_keypoints")
        print("  ")
        print("  # keypoint_sequence should be shape (N, 144) where N is sequence length")
        print("  # Each frame has 144 features: [hand1(63), hand2(63), pose_arms(18)]")
        print("  probabilities = infer_from_keypoints(keypoint_sequence)")
        print("  ")
        print("Or use the class directly:")
        print("  inference = GestureInferenceFromKeypoints(model_path)")
        print("  probabilities = inference.predict_proba(keypoint_sequence)")
        print("  gesture, confidence = inference.predict(keypoint_sequence)")

