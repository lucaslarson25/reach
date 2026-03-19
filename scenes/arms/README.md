# Arms (Arm-Only Reach)

YAML-driven arm reach with auto-discovery and MuJoCo Menagerie arms. See main [README.md](../../README.md#arm-reach-training-primary-workflow) for full documentation.

## Train and run

```bash
python scripts/train.py --arm-id <arm_id>   # e.g. z1, panda, aloha
mjpython scripts/run.py --arm-id <arm_id>
```

## Observation and reward

- **Observation** (per arm): `qpos`, `qvel`, `ee_pos`, `ball_pos`, **distance to ball**, **unit direction EE→ball**, **delta_distance** (getting closer / moving away). Explicit distance and direction help the policy choose where to move and to correct when moving away.
- **Reward**: dense (closer = more), progress (reward for decreasing distance), orientation (EE toward ball), small move-away penalty. Success bonus when within 5 cm.
- **EE-priority scaling** (`ee_priority_scale: true`): actions are scaled so joints closer to the end-effector have higher gain, reducing getting stuck.

## Adding Custom Arms

To add a new arm, either register it in `arm_registry.py` or place its MJCF in `models/arms/<arm_id>/`. Auto-discovery will infer DOF, reach bounds, and EE sites.

**MJCF requirements** (for good training):

- **End-effector site at the arm tip** – A site named `eetip`, `hand`, `gripper`, `attachment_site`, or similar at the end-effector. The reward uses this site for distance and orientation.
- **Ball placement** – With `reach_min_mode: "auto"` (default), the inner workspace boundary is inferred from the arm model by sampling joint configs (10th percentile of horizontal EE distances). Ball targets always fall within the reachable region. Use `reach_min_mode: "registry"` to override with `reach_min_fraction` / `reach_min_floor`.
