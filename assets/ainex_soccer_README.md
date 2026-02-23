# AINex Soccer – MuJoCo Simulation & Action Groups

AINex assets are synced from [tjdavis51/ainex_soccer](https://github.com/tjdavis51/ainex_soccer).
This document describes how the REACH project integrates the AINex humanoid robot.

## Structure

- **assets/ainex/**  
  Canonical robot meshes, MJCF, and URDF files (from ainex_soccer)
- **assets/action_groups/raw/**  
  Original hardware action group (.d6a SQLite databases)
- **assets/action_groups/csv/**  
  CSV exports of action groups (Servo1–Servo22 + timing)
- **scenes/ainex_soccer/**  
  REACH scene: reach task env, training scripts, policies
- **tools/**  
  Conversion utilities, viewer, and action-group playback:
  - `view_ainex_stable.py` – pose the base model (no physics)
  - `replay_actiongroup.py` – replay a single CSV action group
  - `run_sequence.py` – run a choreographed sequence of actions
  - `actiongroup_engine.py` – playback engine (used by replay/run_sequence)

## Quick Start

View the base AINex model (no physics, pose via sliders):

```bash
pip install mujoco numpy
mjpython tools/view_ainex_stable.py
```

Train the reach task (with action groups as reference for policy decisions):

```bash
python -m scenes.ainex_soccer.training.ainex_reach_train --timesteps 500000

# With custom blend (0=policy only, 1=action group only)
python -m scenes.ainex_soccer.training.ainex_reach_train --timesteps 500000 --action-group-blend 0.3

# Without action groups (pure policy)
python -m scenes.ainex_soccer.training.ainex_reach_train --timesteps 500000 --action-groups ""
```

Replay action groups (uses `ainex_physics.xml` with full physics):

```bash
mjpython tools/replay_actiongroup.py                              # default: forward_one_step.csv
mjpython tools/replay_actiongroup.py assets/action_groups/csv/wave.csv

mjpython tools/run_sequence.py                                    # choreographed walk/turn/wave sequence
```

## Updating from ainex_soccer

To pull latest assets from the upstream repo:

```bash
git clone https://github.com/tjdavis51/ainex_soccer.git /tmp/ainex_soccer
cp -r /tmp/ainex_soccer/assets/ainex/* assets/ainex/
cp -r /tmp/ainex_soccer/assets/action_groups/csv/* assets/action_groups/csv/
```
