"""
Task Environments
=================

This module contains specific task environments that inherit from the
base ArmEnvironment.

Each task has:
- Specific reward function
- Task-specific reset logic (object placement, target generation)
- Termination conditions
- Observation augmentations (e.g., target location)

Planned tasks:
- reaching.py: Reach to target positions in 3D space
- grasping.py: Grasp objects of various shapes
- lifting.py: Lift and move objects
- daily_tasks.py: Complex tasks like brushing teeth, eating
"""

# Future exports:
# from .reaching import ReachingTask
# from .grasping import GraspingTask
# from .lifting import LiftingTask

__all__ = [
    # Will be populated with task environment classes
]

