"""
Deployment and Integration Module
==================================

This module handles deploying trained policies to real hardware and
integrating all components together.

Key responsibilities:
- Load trained models for deployment
- Interface with real robotic hardware
- Handle sim-to-real transfer
- Implement safety monitoring
- Coordinate perception, control, and actuation
- Real-time performance optimization

Components:
- policy_loader.py: Load and prepare trained policies
- hardware_interface.py: Communicate with physical arm
- sim_to_real.py: Domain adaptation and transfer
- safety_monitor.py: Real-time safety checks
- integration.py: Coordinate all modules
"""

# Future exports:
# from .policy_loader import PolicyLoader
# from .hardware_interface import HardwareInterface
# from .safety_monitor import SafetyMonitor

__all__ = [
    # Will be populated with deployment classes
]

