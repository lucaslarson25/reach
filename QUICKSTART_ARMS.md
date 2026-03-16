# Arm reach: upload, train, run

Minimal steps to get an arm training and running in the simulator.

## 1. Setup (one time)

From the project root (`reach/`):

**macOS / Linux** — optional one-command setup:

```bash
chmod +x scripts/setup_arms.sh
./scripts/setup_arms.sh
# Then activate the venv:  source .venv/bin/activate
```

**Or do it manually:**

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .            # optional: gives you  train  and  run  commands
```

## 2. Train

Default arm is **Panda** (300k steps, a few minutes).

**If you ran `pip install -e .`:**

```bash
train panda                 # default arm
train ur5e                  # other arm
train ur5e 500000           # arm + steps
```

**Otherwise:**

```bash
python scripts/train.py
python scripts/train.py --arm-id ur5e --steps 500000
```

Policy is saved to `policies/ppo_arms_<arm_id>_mac_300k.zip` (or the step count you used).

## 3. Run the simulation

Opens the MuJoCo viewer with the trained arm reaching for a ball.

**If you ran `pip install -e .`:**

```bash
run panda                   # default arm
run ur5e                    # other arm
run ur5e 10000              # arm + max steps in viewer
```

On macOS, `run` uses `mjpython` for the viewer automatically.

**Otherwise:**

```bash
mjpython scripts/run.py --arm-id panda    # macOS
python scripts/run.py --arm-id panda     # Windows/Linux
```

Close the viewer window to exit.

---

## Cheat sheet

| Goal              | Command (after `pip install -e .`) |
|-------------------|-------------------------------------|
| Train (Panda)     | `train panda` |
| Train other arm   | `train ur5e` or `train ur5e 500000` |
| Run (any arm)     | `run panda` or `run ur5e 10000` |

Without install: `python scripts/train.py --arm-id <id>`, `mjpython scripts/run.py --arm-id <id>` (or `python` on Windows/Linux).

See `config/arms.yaml` and the main [README](README.md) for more options.

---

## Upload another arm

To add **your own** arm and then train and run it:

1. **Upload** – Put the arm MJCF in:
   ```text
   scenes/arms/models/arms/<arm_id>/
   ```
   Name the main file `arm.xml`, `<arm_id>.xml`, or `scene.xml`. Include an end-effector `<site>` (e.g. named `eetip`, `hand`, `tool0`, `attachment_site`).

2. **Train** – Same as above with your id:
   ```bash
   python scripts/train.py --arm-id <arm_id>
   ```

3. **Run** – Same as above:
   ```bash
   mjpython scripts/run.py --arm-id <arm_id>   # macOS
   python scripts/run.py --arm-id <arm_id>      # Windows/Linux
   ```

**Optional:** To set EE site name, reach limits, or home keyframe, add an entry in `scenes/arms/arm_registry.py`. If the arm behaves poorly (erratic, folding), add a section in `config/arm_overrides.yaml`.

**Full pipeline:** [documentation/adding_new_arm.md](documentation/adding_new_arm.md)
