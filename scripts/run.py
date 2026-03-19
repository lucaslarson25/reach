#!/usr/bin/env python3
"""
Default entry point to run trained arm policy with viewer. Run from project root.

  mjpython scripts/run.py                    # run panda (default)
  mjpython scripts/run.py --arm-id aloha     # run ALOHA
  mjpython scripts/run.py --arm-id ur5e --steps 10000

macOS: use mjpython (MuJoCo passive viewer). CLI overrides config/arms.yaml.
"""
import os
import sys

# Run from project root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

if __name__ == "__main__":
    from scenes.arms.training.run_simulation import main
    main()
