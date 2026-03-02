# Arms: arm-only reach

**You upload only the arm.** The scene (floor + ball + arm) is composed at load time. Training and ball sampling adapt to the arm's length and DOF.

## MuJoCo Menagerie arms included

All arms from the [MuJoCo Model Gallery (Arms)](https://mujoco.readthedocs.io/en/stable/models.html) are included under `models/arms/`:

| arm_id       | Model                    |
|-------------|---------------------------|
| arm_2link   | 2-link demo arm           |
| panda       | Franka Emika Panda        |
| fr3         | Franka FR3                |
| ur5e        | Universal Robots UR5e     |
| ur10e       | Universal Robots UR10e    |
| iiwa14      | KUKA LBR iiwa14           |
| xarm7       | UFACTORY xArm7            |
| sawyer      | Rethink Robotics Sawyer   |
| lite6       | UFactory Lite 6           |
| vx300s      | Trossen ViperX 300 6DOF   |
| wx250s      | Trossen WidowX 250 6DOF   |
| aloha       | ALOHA 2 (dual arm)        |
| unitree_z1, z1 | Unitree Z1             |

Source: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).

**Note:** `arm_2link` and `z1` have fallbacks. For `z1`, if `models/arms/z1/` is missing, the env uses `scene_path_fallback` to a pre-built scene in `scenes/industrial_arm_reaching/models/`.

## Layout

- `models/floor.xml`, `models/ball.xml` – shared scene pieces (do not edit for per-arm changes).
- `models/arms/<arm_id>/` – one folder per arm (e.g. `arm.xml`, `<arm_id>.xml`, or `scene.xml` plus any `assets/`). Composed scene is written as `_composed.xml` inside each arm folder so asset paths resolve correctly.
- Registry in `arm_registry.py`: `arm_id` → `arm_path`, `ee_site_name`, `reach_min`/`reach_max`, optional `scene_path_fallback`.

## Config (YAML)

Train and run are driven by **`config/arms.yaml`**:

- **scene.arm_id** – which arm (e.g. `panda`, `ur5e`, `aloha`). See table above.
- **scene.model_path** – optional; `null` = use composed scene from registry/discovery.
- **scene.ball_mode** – `shared` (1 ball, all arms) or `per_arm` (N balls for N arms).
- **scene.per_arm_policies** – `true` = train/run separate policy per arm (multi-arm only).
- **train.total_steps**, **train.policy_dir**, PPO options.
- **train.reward_time_penalty** – per-step penalty (minimum-time).
- **train.reward_smoothness** – jerk penalty for smoother motion.
- **run.policy_path** – `null` = derive from `scene.arm_id` + `train.total_steps`.
- **run.steps**, **run.deterministic**, **run.debug**, **run.stochastic**.

Env vars **ARM_ID**, **TOTAL_STEPS**, **MODEL_PATH**, **USE_MPS** override the YAML when set. CLI flags take precedence.

## Train / run

**Default workflow:** one way to train and run for all arms. CLI tweaks without editing YAML.

From project root:

```bash
# Train (default: panda, 300k steps)
python scripts/train.py

# Run in viewer (default: panda). macOS: use mjpython
mjpython scripts/run.py
```

**CLI overrides** (no YAML edits needed):

```bash
# Train different arm / steps / ball mode
python scripts/train.py --arm-id aloha
python scripts/train.py --arm-id ur5e --steps 500000
python scripts/train.py --arm-id aloha --per-arm-policies

# Run different arm / policy / steps
mjpython scripts/run.py --arm-id aloha
mjpython scripts/run.py --arm-id ur5e --model policies/ppo_arms_ur5e_mac_500k.zip
mjpython scripts/run.py --arm-id aloha --per-arm-policies
```

**Alternative** (explicit config or env vars):

```bash
ARM_ID=panda python -m scenes.arms.training.arm_train_mac --config config/arms.yaml
python -m scenes.arms.training.arm_train_mac --config config/arms.yaml --arm-id aloha
mjpython -m scenes.arms.training.run_simulation --config config/arms.yaml --arm-id aloha
```

**Eval** (metrics only):

```bash
python -m scenes.arms.training.eval_model --model policies/ppo_arms_panda_mac_300k.zip --arm-id panda
```

**macOS:** The MuJoCo passive viewer requires `mjpython` (not `python`) to run the simulation script.

**If the arm doesn't move in the viewer:** set `run.debug: true` in config or pass `--debug`; if action norm is ~0, retrain with more `train.total_steps` or try `run.stochastic: true`.

## Adding another arm

### Auto-discovery (recommended)

1. Copy the arm from Menagerie (or your own XML + assets) into `models/arms/<arm_id>/`.
2. Name the main XML `arm.xml`, `<arm_id>.xml`, or `scene.xml`.
3. Use `--arm-id <arm_id>` or set `scene.arm_id` in config, then train and run.

EE sites and reach are inferred from the MJCF.

### Manual registry

For manual override or fallback scenes, add an entry in `arm_registry.py`:

```python
"panda": ArmConfig(
    name="Franka Panda",
    arm_path="arms/panda/panda.xml",
    ee_site_name="hand",
    reach_min=0.15,
    reach_max=0.85,
    home_keyframe_name="home",
),
```

If the arm XML is not under `models/arms/` yet, set `scene_path_fallback` to a full scene XML path so the env still loads (e.g. a pre-built scene elsewhere).
