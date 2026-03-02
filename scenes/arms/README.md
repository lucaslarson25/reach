# Arms: arm-only reach

**You upload only the arm.** The scene (floor + ball + arm) is composed at load time. Training and ball sampling adapt to the arm’s length and DOF.

## Layout

- `models/floor.xml`, `models/ball.xml` – shared scene pieces (do not edit for per-arm changes).
- `models/arms/<arm_id>/` – put each arm here (e.g. `arm.xml` or `panda.xml` plus any `assets/`).
- Registry in `arm_registry.py`: `arm_id` → `arm_path`, `ee_site_name`, optional `reach_min`/`reach_max`, optional `scene_path_fallback`.

## Adding an arm

1. **Upload the arm**  
   Put the arm XML (and its assets) under `models/arms/<arm_id>/`, e.g.  
   `models/arms/panda/panda.xml` and `models/arms/panda/assets/` if needed.

2. **Register it**  
   In `arm_registry.py`, add:
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
   If the arm XML is not under `models/arms/` yet, you can set `scene_path_fallback` to a full scene XML path so the env still loads (e.g. a pre-built scene elsewhere).

3. **Train**  
   From project root:
   ```bash
   ARM_ID=arm_2link python -m scenes.arms.training.arm_train_mac
   ARM_ID=panda python -m scenes.arms.training.arm_train_mac
   ```

## Built-in arms

- **arm_2link** – 2-link arm; arm XML lives at `models/arms/arm_2link/arm.xml`. No extra assets.
- **z1** – Unitree Z1. If `models/arms/z1/z1_arm.xml` (and assets) are not present, the env uses `scene_path_fallback` to the pre-built scene in `scenes/industrial_arm_reaching/models/z1scene.xml`.

## Run from project root

- Training: `python -m scenes.arms.training.arm_train_mac` or `arm_train`
- Eval: `python -m scenes.arms.training.eval_model --model <path.zip> --arm-id <id>`
