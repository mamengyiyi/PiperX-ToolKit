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
    parser = argparse.ArgumentParser(description="Send PiperX motion/follower output role command.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--reset-after-teaching", action="store_true", help="Send SDK reset command after teaching mode.")
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend="mock")
    config.enable_on_connect = False
    if args.left_can:
        config.left_can = args.left_can
    if args.right_can:
        config.right_can = args.right_can
    env = DualPiperXEnv(config)
    try:
        env.set_motion_output_role()
        if args.reset_after_teaching:
            env.reset_after_teaching()
        print("Sent motion/follower output role command to both arms.")
        print("If the SDK/firmware requires it, power-cycle the arms before deployment.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

