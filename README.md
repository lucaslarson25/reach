# REACH Capstone Project

REACH is a Northern Arizona University capstone project for robotics
simulation, reinforcement learning, rendering, gesture recognition, and AINex
humanoid experimentation. The project combines MuJoCo physics, Gymnasium
environments, Stable-Baselines3 PPO policies, MediaPipe/PyTorch gesture tools,
AINex action-group assets, a static project website, and Monsoon HPC scripts.

For installation instructions, see [SETUP.md](SETUP.md). For commands and
workflows, see [USAGE.md](USAGE.md).

## Current Main Workflows

- Train and run YAML-driven arm-only reaching policies for many MuJoCo
  Menagerie arms through `scripts/train.py` and `scripts/run.py`.
- Render legacy trained PPO policies through config-driven render scripts in
  `renders/`.
- Train and evaluate AINex standing, reaching, and walk-to-ball tasks.
- Replay AINex hardware action groups from CSV files in MuJoCo.
- Run two gesture-recognition paths:
  - the project gesture-classifier pipeline under `scenes/image_recognition/`;
  - the MediaPipe `.task` webcam runners under `gesture_recognition/`.
- Run smoke tests locally or on NAU Monsoon.
- Serve or open the static website in `website/`.

## Repository Layout

```text
reach/
|-- README.md
|-- SETUP.md
|-- USAGE.md
|-- requirements.txt
|-- requirements-hpc.txt
|-- pyproject.toml
|-- eval_model.py
|-- assets/
|   |-- ainex/
|   `-- action_groups/
|-- cluster/
|   |-- test_monsoon.sh
|   |-- train_monsoon.sh
|   |-- train_arms_monsoon.sh
|   `-- run_arms_background.sh
|-- config/
|   |-- arms.yaml
|   |-- arms_loader.py
|   |-- render_run.yaml
|   |-- ainex_render.yaml
|   |-- ainex_reach.yaml
|   |-- ainex_walk_to_ball.yaml
|   `-- render_loader.py
|-- documentation/
|   |-- system_design.md
|   |-- monsoon_setup.md
|   |-- installer_plan.md
|   |-- demos/
|   |-- headshots/
|   |-- logos/
|   `-- *.pdf / *.docx / *.pptx
|-- gesture_recognition/
|-- renders/
|-- scenes/
|   |-- arms/
|   |-- ainex_soccer/
|   |-- industrial_arm_reaching/
|   |-- industrial_arm_reaching_with_welding/
|   |-- image_recognition/
|   |-- cartpole/
|   |-- humanoid/
|   `-- legs/
|-- scripts/
|-- tests/
|-- tools/
|-- website/
`-- logs/
```

## Primary Arm Reach System

The primary workflow is `scenes/arms/`, a YAML-driven arm reach environment.
Only the arm XML is supplied; `scenes/arms/scene_compose.py` composes the arm
with a floor and target ball at runtime.

Default commands:

```bash
python scripts/train.py --arm-id panda
mjpython scripts/run.py --arm-id panda
```

`config/arms.yaml` controls the default arm, ball mode, PPO settings, reward
style, policy directory, and run settings. CLI flags and environment variables
override the YAML. Policies are saved by default to:

```text
policies/ppo_arms_<arm_id>_mac_<steps_in_k>k.zip
```

Registered or discoverable arms include `arm_2link`, `panda`, `fr3`, `ur5e`,
`ur10e`, `iiwa14`, `xarm7`, `sawyer`, `lite6`, `vx300s`, `wx250s`, `aloha`,
`unitree_z1`, and `z1`.

## AINex Assets and Action Groups

The repository includes AINex humanoid assets synced from the AINex Soccer
project:

- `assets/ainex/`: MJCF/URDF files and robot meshes.
- `assets/action_groups/raw/`: original `.d6a` action groups.
- `assets/action_groups/csv/`: exported servo/timing CSV action groups.
- `tools/`: viewers, action-group replay, sequence playback, and converters.
- `scenes/ainex_soccer/`: Gymnasium environments, training scripts, and PPO
  policies.

Useful commands:

```bash
mjpython tools/view_ainex_stable.py
mjpython tools/replay_actiongroup.py assets/action_groups/csv/wave.csv
mjpython tools/run_sequence.py
```

## Legacy and Supporting Systems

- `renders/` contains config-driven renderers for trained PPO policies and
  MuJoCo model inspection.
- `scenes/industrial_arm_reaching/` contains the older Z1 reaching workflow.
- `scenes/image_recognition/` contains a MediaPipe keypoint collection,
  PyTorch classifier training, inference, and gesture-controlled robot demo.
- `gesture_recognition/` contains direct MediaPipe Gesture Recognizer `.task`
  webcam runners and AINex/ROS-oriented trigger scripts.
- `cluster/` and `documentation/monsoon_setup.md` document NAU Monsoon training.
- `tests/smoke_test.py` provides a quick headless setup sanity check.

## Team

- Taylor Davis: team lead, coding, architecture, integration.
- Victor Rodriguez: coding, recording, documentation, model development.
- Clayton Ramsey: coding, architecture, environment structure, testing.
- Lucas Larson: coding, version control management, repository operations.

## Sponsors

- Dr. Zach Lerner, Ph.D., Associate Professor, Mechanical Engineering, NAU.
- Prof. Carlo R. da Cunha, Ph.D., Assistant Professor, Electrical Engineering,
  NAU.

## License

This project was developed for academic purposes as part of the NAU Computer
Science Capstone Program. All rights reserved by the REACH development team and
Northern Arizona University.
