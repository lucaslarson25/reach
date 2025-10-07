"""
Data Management Module
======================

This module handles data storage, retrieval, and management for the REACH project.

Key responsibilities:
- Store training trajectories and episodes
- Log experimental results
- Save/load trained models with metadata
- Track hyperparameter configurations
- Store evaluation metrics
- Manage datasets for offline RL (if needed)

Database options:
- SQLite: Lightweight, file-based, good for local development
- PostgreSQL: More robust, good for shared team database on server

This allows team members to:
- Compare different experiments
- Track which hyperparameters work best
- Replay successful trajectories
- Share results across team
"""

# Future exports:
# from .database import Database
# from .trajectory_logger import TrajectoryLogger
# from .experiment_tracker import ExperimentTracker

__all__ = [
    # Will be populated with data management classes
]

