#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Send PiperX teaching/master input role command.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--configure-gripper-params", action="store_true")
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend="mock")
    if args.left_can:
        config.left_can = args.left_can
    if args.right_can:
        config.right_can = args.right_can
    env = DualPiperXEnv(config)
    try:
        if args.configure_gripper_params:
            env.configure_gripper_teaching_pendants()
        env.set_teaching_input_role()
        print("Sent teaching/master input role command to both arms.")
        print("If the SDK/firmware requires it, power-cycle the arms before collecting.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
