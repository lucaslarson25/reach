# System Design and Limitations

## Architecture Overview
- **Simulation Core:** MuJoCo 3.x for physics and rendering.
- **RL Framework:** Stable-Baselines3 (PPO) with Gymnasium environments.
- **Scene Configuration:** YAML files select environment class, model XML, policy path, and run settings.
- **Assets:** MuJoCo XML + mesh files stored per scene in `scenes/<scene_name>/models/`.
- **Rendering Pipeline:** `renders/render_demo*.py` loads config, policy, and runs rollouts in a MuJoCo viewer.

## Key Assumptions
- The robot model XML is self-contained and valid for MuJoCo.
- Training uses a fixed action space that directly drives MuJoCo actuators.
- Policies are trained and evaluated from the repo root so relative paths resolve correctly.
- macOS users must use `mjpython` for viewer rendering.

## Current Limitations
- **Contact realism:** Foot-ground and arm-table contacts are approximations and may not match real-world dynamics.
- **Actuator modeling:** Simple position actuators and rewards can under-model motor limits, backlash, and latency.
- **Domain gap:** Visual appearance, friction, and joint compliance are not calibrated to a real robot.
- **Evaluation scope:** Rollouts validate task-level behavior but not robustness to disturbances.
- **Sensor model:** No realistic sensor noise or delays are modeled by default.

## Sim-to-Real Risks
- Mismatch in joint limits, damping, and torque limits can cause unstable motions.
- Surface friction and contact compliance differ from real environments.
- Policies may overfit to the simulator’s deterministic physics without noise.

## Next Steps Toward Physical Deployment
- Add sensor noise, actuator latency, and friction randomization.
- Calibrate model parameters with real robot measurements.
- Add safety constraints to limit joint velocities and accelerations.
- Validate policies on simplified real-world tasks before full deployment.
- Implement data logging pipelines for real-world rollouts to refine models.
