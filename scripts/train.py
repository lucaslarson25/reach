#!/usr/bin/env python3
"""
Default entry point to train arm reach policies. Run from project root.

  python scripts/train.py                    # train panda (default)
  python scripts/train.py --arm-id aloha     # train ALOHA
  python scripts/train.py --arm-id ur5e --steps 500000
  python scripts/train.py --arm-id aloha --per-arm-policies

CLI overrides config/arms.yaml. See --help for all options.
"""
import os
import sys

# Run from project root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

if __name__ == "__main__":
    from scenes.arms.training.arm_train_mac import main
    main()
