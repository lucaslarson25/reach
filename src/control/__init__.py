"""
Control Module
==============

This module handles robot control interfaces and policies.

Components:
- policy.py: Neural network policies for control
- controllers.py: Low-level controllers (PID, impedance control)
- sim_to_real.py: Utilities for transferring policies to real hardware
- safety.py: Safety constraints and monitoring

The control module bridges the gap between:
- RL policies (high-level commands)
- Robot actuators (low-level control)
- Real hardware (when ready for deployment)
"""

# Future exports:
# from .policy import NeuralPolicy
# from .controllers import PIDController, ImpedanceController
# from .safety import SafetyMonitor

__all__ = [
    # Will be populated with control classes
]

