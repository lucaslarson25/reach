# Gesture-Controlled Robot Integration

Integration of gesture recognition pipeline with MuJoCo simulation for robot control.

## Overview

This module integrates the gesture recognition system with the MuJoCo reaching environment, allowing the robot to respond to hand gestures detected from camera feeds.

## Features

- **MuJoCo Camera Rendering**: Renders camera frames from MuJoCo simulation
- **Gesture Processing**: Processes images through MediaPipe and PyTorch classifier
- **Gesture-to-Action Mapping**: Maps detected gestures to robot control actions
- **Real-time Control**: Robot responds to gestures in real-time
- **Accuracy Tracking**: Monitors gesture recognition accuracy (target: ≥80%)

## Gesture-to-Action Mapping

- **handshake** → `approach`: Robot moves toward target
- **fist_bump** → `retreat`: Robot moves away from target  
- **high_five** → `execute`: Robot executes reaching task (moves toward target with higher intensity)

## Usage

### Demo Script

Run the complete pipeline demo:

```bash
python scenes/image_recognition/integration/demo_gesture_control.py \
    --model-path scenes/image_recognition/models/gesture_classifier.pth \
    --duration 60 \
    --show-camera \
    --min-confidence 0.6
```

**Arguments:**
- `--model-path`: Path to trained gesture classifier
- `--camera-name`: Name of camera in MuJoCo model (default: 'head_camera')
- `--duration`: Demo duration in seconds (default: 60)
- `--show-camera`: Display camera feed with gesture overlay
- `--min-confidence`: Minimum confidence threshold (default: 0.6)
- `--use-webcam`: Use webcam for gesture input (default: True)

**Controls:**
- Press `q` to quit
- Press `r` to reset environment

### Programmatic Usage

```python
from scenes.image_recognition.integration.gesture_controlled_robot import GestureControlledRobot

# Initialize
robot = GestureControlledRobot(
    model_path='scenes/image_recognition/models/gesture_classifier.pth',
    camera_name='head_camera'
)

# Reset environment
obs = robot.reset()

# Step with gesture control
obs, reward, terminated, truncated, info, gesture_info = robot.step(use_gesture_control=True)

# Access gesture information
gesture_name = gesture_info['gesture']  # 'handshake', 'fist_bump', 'high_five'
confidence = gesture_info['confidence']  # 0.0 to 1.0
action_name = gesture_info['action']     # 'approach', 'retreat', 'execute'

# Cleanup
robot.close()
```

## Architecture

### GestureControlledRobot Class

Main integration class that:
1. Renders camera frames from MuJoCo simulation
2. Extracts MediaPipe keypoints from images
3. Processes keypoint sequences through gesture classifier
4. Maps gestures to robot actions
5. Executes actions in the simulation

### Camera Rendering

- Uses MuJoCo's `Renderer` to capture camera views
- Supports named cameras or default view
- Renders at configurable resolution (default: 640x480)

### Action Execution

Actions are mapped to robot joint movements:
- **approach**: Moves end effector toward target ball
- **retreat**: Moves end effector away from target
- **execute**: Aggressive movement toward target for task completion

## Accuracy Requirements

The system tracks gesture recognition accuracy with a target of **≥80%**.

Accuracy is calculated as:
- Average confidence of detected gestures
- Ratio of high-confidence detections (≥0.7)

For best results:
- Ensure good lighting conditions
- Make clear, distinct gestures
- Maintain consistent distance from camera
- Allow sufficient frames (10+) for sequence processing

## File Structure

```
scenes/image_recognition/integration/
├── __init__.py
├── README.md
├── gesture_controlled_robot.py    # Main integration class
└── demo_gesture_control.py        # Demo script
```

## Requirements

- Trained gesture classifier model (`gesture_classifier.pth`)
- MediaPipe installed
- MuJoCo environment with camera (or default view)
- Webcam (optional, for practical demo)

## Troubleshooting

**No gestures detected:**
- Check camera is working
- Ensure sufficient lighting
- Verify model is loaded correctly
- Lower `--min-confidence` threshold

**Low accuracy:**
- Collect more diverse training data
- Ensure good lighting in demo environment
- Make clear, distinct gestures
- Check model was trained with ≥85% test accuracy

**Robot not responding:**
- Check gesture is being detected (confidence > threshold)
- Verify action mapping is correct
- Check environment is running correctly

