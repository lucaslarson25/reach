"""
Hardware Interface
==================

Interfaces with the physical robotic arm hardware.

Purpose:
- Send motor commands to real arm
- Read sensor data from real arm
- Handle communication protocols
- Ensure safe operation
- Map simulation commands to real hardware

Safety is CRITICAL here - the real arm can cause injury or damage!
Always include:
- Emergency stop capability
- Joint limit enforcement
- Force/torque limits
- Velocity limits
- Collision detection
"""

# import serial  # For serial communication with arm
# import time
# import numpy as np
# from typing import Optional, Dict
#
#
# class HardwareInterface:
#     """
#     Interface to physical robotic arm.
#     
#     SAFETY FIRST: This code controls real hardware that can cause harm.
#     - Always test in simulation first
#     - Implement emergency stops
#     - Enforce joint limits strictly
#     - Monitor forces and velocities
#     """
#     
#     def __init__(self, port: str = "/dev/ttyUSB0", config: Optional[Dict] = None):
#         """
#         Initialize hardware interface.
#         
#         Args:
#             port: Serial port for arm communication
#             config: Hardware configuration (joint limits, calibration, etc.)
#         """
#         # TODO: Open serial connection
#         # TODO: Load hardware configuration
#         # TODO: Initialize safety limits
#         # TODO: Calibrate arm (homing sequence)
#         # TODO: Setup emergency stop handler
#         pass
#     
#     def send_command(self, joint_positions: np.ndarray, 
#                     check_limits: bool = True) -> bool:
#         """
#         Send position command to arm.
#         
#         Args:
#             joint_positions: Desired joint positions (radians)
#             check_limits: Verify command is safe before sending
#         
#         Returns:
#             success: True if command sent successfully
#         """
#         # SAFETY CHECK: Verify command is within limits
#         # if check_limits and not self._check_safe(joint_positions):
#         #     print("WARNING: Command outside safe limits, not sending!")
#         #     return False
#         
#         # TODO: Convert to hardware units
#         # TODO: Format command message
#         # TODO: Send via serial
#         # TODO: Wait for acknowledgment
#         # TODO: Return success status
#         pass
#     
#     def read_sensors(self) -> Dict:
#         """
#         Read current state from arm sensors.
#         
#         Returns:
#             sensor_data: Dict with joint positions, velocities, forces, etc.
#         """
#         # TODO: Request sensor data from arm
#         # TODO: Parse response
#         # TODO: Convert to standard units
#         # TODO: Return sensor dict
#         pass
#     
#     def emergency_stop(self):
#         """
#         EMERGENCY STOP - halt all motion immediately.
#         
#         This should:
#         - Cut power to motors
#         - Set all velocities to zero
#         - Log the emergency stop
#         - Require manual reset before resuming
#         """
#         # TODO: Send emergency stop command
#         # TODO: Log emergency stop event
#         # TODO: Set flag requiring manual reset
#         pass
#     
#     def _check_safe(self, joint_positions: np.ndarray) -> bool:
#         """
#         Verify command is safe to execute.
#         
#         Checks:
#         - Joint positions within limits
#         - Change from current position is reasonable
#         - Resulting forces/torques are safe
#         
#         Returns:
#             safe: True if command is safe
#         """
#         # TODO: Check joint limits
#         # TODO: Check rate of change
#         # TODO: Predict forces/torques
#         # TODO: Return safety verdict
#         pass
#     
#     def calibrate(self):
#         """
#         Calibrate arm (find home position).
#         
#         Typically done at startup to establish zero positions.
#         """
#         # TODO: Move to limit switches
#         # TODO: Set zero positions
#         # TODO: Verify calibration
#         pass
#     
#     def close(self):
#         """
#         Safely shut down hardware interface.
#         """
#         # TODO: Move to safe position
#         # TODO: Disable motors
#         # TODO: Close serial connection
#         pass

