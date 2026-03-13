# Adding a New Arm

This guide helps you train the ball-reaching policy on **any arm** you add to the project. The system uses per-arm overrides to handle different robot geometries and failure modes automatically.

## Quick Start

1. Place your arm MJCF in `scenes/arms/models/arms/<arm_id>/`.
2. Add the arm to the registry (or use discovery).
3. Train: `python scripts/train.py --arm-id <arm_id> --steps 300000`
4. If you see issues, add entries to `config/arm_overrides.yaml`.

## Per-Arm Overrides

Edit `config/arm_overrides.yaml` to fix common problems. Add a section keyed by `arm_id`:

```yaml
my_arm:
  reach_max_cap: 0.5        # Limit ball distance (m). Lower = easier.
  reach_min_mode: registry  # Use registry reach_min (closer balls)
  initial_pose: random      # Randomize start pose each episode
  ball_mode: per_arm        # For bimanual: each arm has its own ball
  joint_limit_margin_penalty: 0.02   # Penalize joints near limits
```

### When to Use Each Override

| Problem | Override | Typical Value |
|--------|----------|---------------|
| Arm moves erratically (circles, close-then-far) | `reach_max_cap` | `0.5` (tighter ball band) |
| Arms start "connected" (bimanual) | `initial_pose` | `random` |
| Arms interfere reaching same ball | `ball_mode` | `per_arm` |
| Arm folds into itself / self-collision | `joint_limit_margin_penalty` | `0.015`–`0.03` |
| Balls too far, policy gives up | `reach_max_cap` | Lower than registry `reach_max` |
| Need a different home pose | `initial_keyframe` | Keyframe name in your XML |

## Troubleshooting

### Arm doesn’t move toward the ball

- Train longer (e.g. 500k steps).
- Lower `reach_max_cap` in `arm_overrides.yaml` for easier targets.
- Ensure `ee_priority_scale: false` in `config/arms.yaml` (let the policy learn coordination).

### Bimanual arms start with grippers together

- Add `initial_pose: random` for your arm in `arm_overrides.yaml`.
- Optionally use `ball_mode: per_arm` so each arm has its own target ball.

### Arm folds into itself (e.g. Z1)

- Add `joint_limit_margin_penalty: 0.02` for your arm.
- Or switch to the industrial Z1 scene: in `arms.yaml` set  
  `model_path: "scenes/industrial_arm_reaching/models/z1scene.xml"` and `arm_id: z1` for that run.

### Policy completes training but barely moves at runtime

- Check NumPy compatibility (policy saved/loaded with the same NumPy version).
- Retrain with more steps.

## Override Keys Reference

**Scene / reset:**
- `initial_pose`: `"home"` (default) or `"random"`
- `initial_keyframe`: keyframe name for reset (if different from `home`)
- `ball_mode`: `"shared"` or `"per_arm"`
- `model_path`: full scene XML path (optional)

**Train / ball placement:**
- `reach_min_mode`: `"auto"` or `"registry"`
- `reach_max_cap`: max ball distance in meters
- `joint_limit_margin_penalty`: penalty when joints are near limits (float or omit to disable)

## See Also

- `config/arm_overrides.yaml` – existing overrides for ur5e, aloha, z1
- `config/arms.yaml` – global defaults
- `scenes/arms/arm_registry.py` – arm definitions and reach limits
