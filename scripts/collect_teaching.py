#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.collect import Collector
from piperx_toolkit.teleop import TeachingPendantTeleop
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bimanual PiperX teaching/master-input demonstrations.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--dataset", "-d", default="datasets/demo.zarr")
    parser.add_argument("--episodes", "-n", type=int, default=1)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--task", default="")
    parser.add_argument("--action-shift-frames", type=int, default=1)
    parser.add_argument("--configure-roles", action="store_true", help="Send teaching/master role command before collecting.")
    parser.add_argument("--configure-gripper-params", action="store_true", help="Configure gripper/teaching-pendant params before collecting.")
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend=args.camera_backend)
    if args.left_can:
        config.left_can = args.left_can
    if args.right_can:
        config.right_can = args.right_can
    env = DualPiperXEnv(config)
    teleop = TeachingPendantTeleop(
        env,
        configure_roles=args.configure_roles,
        configure_gripper_params=args.configure_gripper_params,
    )
    try:
        collector = Collector(
            env=env,
            dataset_path=args.dataset,
            num_episodes=args.episodes,
            hz=args.hz,
            task=args.task,
            action_shift_frames=args.action_shift_frames,
            teleop_source=teleop,
        )
        collector.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
