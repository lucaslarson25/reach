"""
Simple CLI so users can run:  train <arm_id> [steps]   and   run <arm_id> [steps]
Install with:  pip install -e .   then  train panda   or  run panda 5000
"""

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_arm_and_steps(argv: list[str], default_arm: str = "panda"):
    """Parse argv into (arm_id, steps_int_or_none). First pos = arm_id, second if digits = steps."""
    args = [a for a in argv if not a.startswith("-")]
    arm_id = args[0] if args else default_arm
    steps = None
    if len(args) > 1 and args[1].isdigit():
        steps = args[1]
    return arm_id, steps


def _has_help(argv: list[str]) -> bool:
    return "--help" in argv or "-h" in argv


def train_main() -> None:
    """Entry point: train <arm_id> [steps]"""
    repo = _repo_root()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, str(repo))

    argv = sys.argv[1:]
    if _has_help(argv):
        sys.argv = ["train", "--help"]
        from scenes.arms.training.arm_train_mac import main
        main()
        return
    arm_id, steps = _parse_arm_and_steps(argv)
    new_argv = ["train", "--arm-id", arm_id]
    if steps is not None:
        new_argv += ["--steps", steps]
    sys.argv = new_argv

    from scenes.arms.training.arm_train_mac import main
    main()


def run_main() -> None:
    """Entry point: run <arm_id> [steps]. On macOS uses mjpython for the viewer."""
    repo = _repo_root()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, str(repo))

    argv = sys.argv[1:]
    if _has_help(argv):
        sys.argv = ["run", "--help"]
        from scenes.arms.training.run_simulation import main
        main()
        return
    arm_id, steps = _parse_arm_and_steps(argv)
    run_argv = ["--arm-id", arm_id]
    if steps is not None:
        run_argv += ["--steps", steps]

    # On macOS use mjpython for MuJoCo viewer; otherwise python
    use_mjpython = sys.platform == "darwin"
    if use_mjpython:
        import shutil
        py = shutil.which("mjpython") or shutil.which("python") or "python"
    else:
        py = sys.executable

    # Run as: mjpython -m scenes.arms.training.run_simulation --arm-id X [--steps Y]
    module = "scenes.arms.training.run_simulation"
    cmd = [py, "-m", module] + run_argv
    os.execvp(py, cmd)
