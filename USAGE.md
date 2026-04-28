# Usage

Run commands from the repository root with your virtual environment active.
Use `mjpython` instead of `python` for MuJoCo viewer commands on macOS.

## Quick Sanity Check

```bash
python -m tests.smoke_test
```

This headless test verifies core imports, `AINexReachEnv`, and a tiny PPO learn
step.

## Primary Arm Reach Workflow

The main workflow is `scenes/arms/`, driven by `config/arms.yaml` and the
wrappers in `scripts/`.

### Train

```bash
python scripts/train.py
python scripts/train.py --arm-id panda
python scripts/train.py --arm-id ur5e --steps 500000
python scripts/train.py --arm-id aloha --per-arm-policies
```

Useful flags:

- `--config config/arms.yaml`
- `--arm-id <arm_id>`
- `--steps <timesteps>`
- `--ball-mode shared` or `--ball-mode per_arm`
- `--per-arm-policies`

Environment variables also work:

```bash
ARM_ID=aloha TOTAL_STEPS=1000000 python scripts/train.py
USE_MPS=1 python scripts/train.py --arm-id panda
MODEL_PATH=scenes/industrial_arm_reaching/models/z1scene.xml python scripts/train.py --arm-id z1
```

Default output:

```text
policies/ppo_arms_<arm_id>_mac_<steps_in_k>k.zip
```

For multi-arm per-arm policies:

```text
policies/ppo_arms_<arm_id>_arm<i>_mac_<steps_in_k>k.zip
```

### Run With Viewer

macOS:

```bash
mjpython scripts/run.py --arm-id panda
mjpython scripts/run.py --arm-id ur5e --steps 10000
mjpython scripts/run.py --arm-id aloha --per-arm-policies
```

Windows / Linux:

```bash
python scripts/run.py --arm-id panda
python scripts/run.py --arm-id ur5e --model policies/ppo_arms_ur5e_mac_500k.zip
```

Useful flags:

- `--model <policy.zip>` to override the derived policy path.
- `--steps <count>` to change run length.
- `--debug` to print observation/action info.
- `--stochastic` to sample nondeterministic actions.

### Evaluate Without Viewer

```bash
python -m scenes.arms.training.eval_model --model policies/ppo_arms_panda_mac_300k.zip --arm-id panda --episodes 10
```

### Supported Arm IDs

The current registry and discovery system supports:

```text
arm_2link, panda, fr3, ur5e, ur10e, iiwa14, xarm7, sawyer,
lite6, vx300s, wx250s, aloha, unitree_z1, z1
```

To add another arm, place its MJCF under
`scenes/arms/models/arms/<arm_id>/` and use a main XML named `arm.xml`,
`<arm_id>.xml`, or `scene.xml`. Add a registry entry in
`scenes/arms/arm_registry.py` only when auto-discovery is not enough.

## `config/arms.yaml`

Important settings:

- `scene.arm_id`: default arm to train or run.
- `scene.model_path`: optional full scene XML override.
- `scene.ball_mode`: `shared` or `per_arm`.
- `scene.per_arm_policies`: train/load separate policies for multi-arm models.
- `train.total_steps`: default PPO timesteps.
- `train.policy_dir`: output policy directory.
- `train.reward_style`: `z1` or `arms`.
- `train.reach_min_mode`: `auto` or `registry`.
- `train.ee_priority_scale`: scale distal joints more heavily.
- `train.device`: explicit `cpu`, `cuda`, `mps`, or null for automatic local
  default.
- `run.policy_path`: explicit policy path override.
- `run.steps`, `run.deterministic`, `run.debug`, `run.stochastic`.

CLI flags override YAML, and environment variables override YAML where
implemented.

## Legacy Renderers

Use these for config-driven PPO rendering outside the primary `scenes/arms/`
workflow.

### Render a Policy

Windows / Linux:

```powershell
python renders\render_demo.py --config config\render_run.yaml
python renders\render_demo.py --config config\ainex_render.yaml
python renders\render_demo.py --config config\ainex_reach.yaml
python renders\render_demo.py --config config\ainex_walk_to_ball.yaml
```

macOS:

```bash
mjpython renders/render_demo_mac.py --config config/render_run.yaml
mjpython renders/render_demo_mac.py --config config/ainex_render.yaml
mjpython renders/render_demo_mac.py --config config/ainex_reach.yaml
mjpython renders/render_demo_mac.py --config config/ainex_walk_to_ball.yaml
```

### View a Model Without a Policy

Windows / Linux:

```powershell
python renders\render_model.py --config config\render_run.yaml
python renders\render_model.py --model scenes\industrial_arm_reaching\models\z1scene.xml
```

macOS:

```bash
mjpython renders/render_model_mac.py --config config/render_run.yaml
mjpython renders/render_model_mac.py --model assets/ainex/ainex_stable.xml
```

### Render Config Schema

```yaml
scene:
  env_class: scenes.industrial_arm_reaching.env:Z1ReachEnv
  model_xml: scenes/industrial_arm_reaching/models/z1scene.xml

policy:
  path: scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip

run:
  episodes: 10
  max_seconds_per_ep: 30.0
  deterministic: true
  trace: true
```

`trace: true` saves trajectory CSV/PNG files under `logs/trajectories/` when
the environment exposes an end-effector position.

## AINex Training and Evaluation

### Standing

```bash
python scenes/ainex_soccer/training/ainex_train.py --seed 42 --timesteps 300000 --num-envs 8
```

Output:

- `scenes/ainex_soccer/policies/ppo_ainex_stand.zip`
- `logs/ainex_stand/monitor.csv`
- `logs/ainex_stand/episode_metrics.csv`

### Arm Reaching

```bash
python scenes/ainex_soccer/training/ainex_reach_train.py --seed 42 --timesteps 1500000 --num-envs 8
python scenes/ainex_soccer/training/ainex_reach_train.py --timesteps 500000 --action-group-blend 0.3
python scenes/ainex_soccer/training/ainex_reach_train.py --timesteps 500000 --action-groups ""
```

Output:

- `scenes/ainex_soccer/policies/ppo_ainex_reach.zip`
- `logs/ainex_reach/monitor.csv`
- `logs/ainex_reach/episode_metrics.csv`

Evaluate with rendering and trajectories:

```bash
mjpython scenes/ainex_soccer/training/ainex_reach_eval.py --config config/ainex_reach.yaml --episodes 5 --max-seconds 20 --deterministic
```

### Walk To Ball / Whole Body

```bash
python scenes/ainex_soccer/training/ainex_walk_to_ball_train.py --seed 42 --timesteps 1000000 --num-envs 8
python scenes/ainex_soccer/training/ainex_reach_wholebody_train.py --seed 42 --timesteps 2000000 --num-envs 8
```

Both save the walk-to-ball policy path:

```text
scenes/ainex_soccer/policies/ppo_ainex_walk_to_ball.zip
```

Evaluate:

```bash
mjpython scenes/ainex_soccer/training/ainex_walk_to_ball_eval.py --config config/ainex_walk_to_ball.yaml --episodes 5 --max-seconds 25 --deterministic
```

## AINex Action Groups and Tools

View the stable AINex model with control sliders:

```bash
mjpython tools/view_ainex_stable.py
```

Replay one action group:

```bash
mjpython tools/replay_actiongroup.py
mjpython tools/replay_actiongroup.py assets/action_groups/csv/wave.csv
```

Run the choreographed sequence in `tools/run_sequence.py`:

```bash
mjpython tools/run_sequence.py
```

Export `.d6a` action-group files to CSV:

```bash
python tools/extract_actiongroup_to_csv.py --help
```

## Industrial Arm Reaching Legacy Workflow

The older Z1 workflow remains in `scenes/industrial_arm_reaching/`.

Train:

```bash
python scenes/industrial_arm_reaching/training/arm_train.py
python scenes/industrial_arm_reaching/training/arm_train_mac.py
```

Evaluate:

```bash
python eval_model.py --model scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip --arm-id z1
python scenes/industrial_arm_reaching/training/eval_model.py
```

For new arm work, prefer the primary `scenes/arms/` workflow.

## Gesture Recognition: Project Classifier Pipeline

This pipeline lives under `scenes/image_recognition/` and uses MediaPipe
keypoints plus a PyTorch LSTM/CNN classifier.

Collect data:

```bash
python scenes/image_recognition/training/collect_gesture_data.py
```

Train:

```bash
python scenes/image_recognition/training/train_gesture_classifier.py --data-dir scenes/image_recognition/data --model-type lstm --epochs 50 --batch-size 16 --lr 0.001
```

Run webcam inference:

```bash
python scenes/image_recognition/training/infer_gesture.py --model-path scenes/image_recognition/models/gesture_classifier.pth --log-file logs/gesture_events/session.csv
```

Run the gesture-controlled robot demo:

```bash
python scenes/image_recognition/integration/demo_gesture_control.py --model-path scenes/image_recognition/models/gesture_classifier.pth --duration 60 --show-camera --min-confidence 0.6
```

Supported classes are `handshake`, `fist_bump`, and `high_five`.

## Gesture Recognition: MediaPipe `.task` Webcam Runners

The top-level `gesture_recognition/` directory runs MediaPipe Gesture
Recognizer `.task` models directly.

Basic webcam recognition:

```bash
python gesture_recognition/gesture_webcam.py --model gesture_recognition/FISTBUMP_HIGHFIVE_HANDSHAKE.task --camera 0 --mirror --min-score 0.6 --print-on-change
```

Options include `--width`, `--height`, `--fps`, `--camera`, `--mirror`,
`--min-score`, `--print-on-change`, and `--print-every`.

AINex snapshot/trigger scripts:

```bash
python gesture_recognition/ainex_gesture_recognition.py --model gesture_recognition/FISTBUMP_HIGHFIVE_HANDSHAKE.task --show
python gesture_recognition/ainex_gesture_rec_with_res.py --model gesture_recognition/FISTBUMP_HIGHFIVE_HANDSHAKE.task --dry-run
python gesture_recognition/min_run.py --model gesture_recognition/FISTBUMP_HIGHFIVE_HANDSHAKE.task --dry-run
```

The trigger scripts include persistence and cooldown options such as
`--trigger-score`, `--persist-sec`, and `--cooldown-sec`. They are intended for
AINex/ROS integration; use `--dry-run` until the target ROS container is known
to be correct.

## Monsoon / HPC Training

Full instructions are in `documentation/monsoon_setup.md`.

Smoke test:

```bash
./cluster/test_monsoon.sh
sbatch cluster/test_monsoon.sh
```

Arm training through SLURM:

```bash
ARM_ID=panda TIMESTEPS=5000000 sbatch cluster/train_arms_monsoon.sh
```

Background training without SLURM:

```bash
ARM_ID=aloha TIMESTEPS=5000000 EMAIL=you@nau.edu nohup ./cluster/run_arms_background.sh > logs/arms_train.log 2>&1 &
tail -f logs/arms_train.log
```

The Monsoon scripts commit changed `policies/ppo_arms_*.zip` files and push to
`origin monsoon` after successful training.

## Cartpole, Humanoid, and Legs

- `scenes/cartpole/training/` contains Jupyter notebooks.
- `scenes/humanoid/` is a placeholder for full-body humanoid scenes.
- `scenes/legs/` is a placeholder for legs-only scenes.

Start JupyterLab:

```bash
jupyter lab
```

## Static Website

Open `website/index.html` directly or serve the directory:

```bash
python -m http.server 8000 -d website
```

Then visit `http://localhost:8000`.
