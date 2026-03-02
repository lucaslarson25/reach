# Multi-Arm Reach Training

The reach env supports **multiple arms** with different lengths and degrees of freedom. Ball targets are sampled **within reach** of the current arm; the **end-effector** is always the tip of the arm (resolved from the model); and **smooth motion** is handled by the same actuator logic for any DOF.

## Supported arms (registry)

- **z1** – Unitree Z1 (6 DOF), scene: `models/z1scene.xml`
- **arm_2link** – 2-link demo arm, scene: `models/arm_scene.xml`

## Adding a MuJoCo Menagerie arm

[Model Gallery (Arms)](https://mujoco.readthedocs.io/en/stable/models.html) lists Menagerie arms (Franka Panda, UR5e, xArm7, etc.). To add one:

### 1. Get the arm model

Clone or copy the arm from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), e.g.:

- `franka_emika_panda/` → copy `panda.xml` (or `panda_nohand.xml`) and any `assets/` it needs
- Put them under e.g. `scenes/industrial_arm_reaching/models/menagerie/franka_emika_panda/`

### 2. Create a scene XML

Create a scene that includes **floor**, **ball**, and the **arm** (same pattern as `z1scene.xml`):

```xml
<mujoco model="panda_reach_scene">
  <include file="floor.xml"/>
  <include file="ball.xml"/>
  <include file="menagerie/franka_emika_panda/panda.xml"/>
</mujoco>
```

- The arm XML should define the robot under `<worldbody>` (Menagerie arm files do this).
- The **ball** must be a body named `"ball"` (see `ball.xml`). The env moves this body each reset.

### 3. Register the arm

In `arm_registry.py`, add an entry to `ARM_REGISTRY`:

```python
"panda": ArmConfig(
    name="Franka Emika Panda",
    scene_path=_scene_path("panda_scene.xml"),  # or path to your scene
    ee_site_name="hand",   # Menagerie Panda uses "hand"; check the arm XML for <site name="...">
    reach_min=0.15,
    reach_max=0.85,
    home_keyframe_name="home",  # optional; if missing, env uses qpos0
),
```

- **ee_site_name**: The MuJoCo `<site>` at the tip of the arm. If you omit it, the env tries common names (`eetip`, `hand`, `attachment`, `tool0`, `ee`, etc.) or the first site in the model.
- **reach_min / reach_max**: Ball is sampled in a horizontal disk with radius in `[reach_min, reach_max]`. If `reach_max` is `None`, it is **computed** from the model by sampling random joint configs (slower first run).
- **home_keyframe_name**: Keyframe used at reset. If the model has no keyframe with this name, the env uses `qpos0`.

### 4. Train

```bash
# By arm ID (from registry)
ARM_ID=panda python -m scenes.industrial_arm_reaching.training.arm_train_mac

# Or by scene path
MODEL_PATH=scenes/industrial_arm_reaching/models/panda_scene.xml python -m scenes.industrial_arm_reaching.training.arm_train_mac
```

Observation space is always `(qpos + qvel + ee_pos(3) + ball_pos(3))`; action space is `model.nu`. So **different DOF is handled automatically**; the policy size adapts to the loaded model.

## Summary

| Concern | How it’s handled |
|--------|-------------------|
| Different arm lengths | Ball sampled in `[reach_min, reach_max]` (from registry or computed). |
| Different DOF | `action_space` and `observation_space` built from loaded model (`nq`, `nv`, `nu`). |
| End-effector at tip | Resolved by `ee_site_name` (registry or auto-detect from common site names). |
| Smooth motion | Same scaling and smoothing of actions into actuator ranges for any arm. |
