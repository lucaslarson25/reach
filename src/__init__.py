"""
REACH - Reinforcement Learning for Assistive Robotic Arm
=========================================================

This package contains the core simulation, training, and control components
for the REACH capstone project.

Main modules:
- simulation: MuJoCo environment definitions for the robotic arm
- agents: RL agent implementations (PPO, SAC)
- vision: YOLO-based object detection and visual processing
- control: Robot control interfaces and policies
- utils: Shared utilities, configurations, and helper functions
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "REACH Development Team"
__email__ = "tjd352@nau.edu"

# When we add core classes, we can expose them here for easier imports
# Example:
# from .simulation import ArmEnvironment
# from .agents import PPOAgent, SACAgent
# from .vision import YOLODetector

# For now, just define the package structure
__all__ = [
    # Will be populated as we implement core classes
]

