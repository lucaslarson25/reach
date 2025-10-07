"""
Vision Module
=============

This module integrates YOLO-based object detection with the robotic
arm simulation and control.

Key components:
- yolo_detector.py: YOLOv8 interface for object detection
- camera.py: Camera simulation in MuJoCo
- visual_processing.py: Image preprocessing and postprocessing
- integration.py: Integrating vision with RL (visual observations)

Vision system responsibilities:
- Detect objects in the environment (cups, utensils, etc.)
- Provide object locations as observations to RL agent
- Enable vision-based policies
- Support sim-to-real transfer with realistic camera simulation
"""

# Future exports:
# from .yolo_detector import YOLODetector
# from .camera import Camera
# from .visual_processing import preprocess_image, extract_features

__all__ = [
    # Will be populated with vision classes
]

