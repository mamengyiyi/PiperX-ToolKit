#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.deploy.policy_runner import ActionLimiter, PolicyRunner, load_policy
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a policy to bimanual PiperX arms.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    parser.add_argument("--policy", required=True, help=".npy replay sequence or torch policy")
    parser.add_argument("--action-mode", default="absolute_joint", choices=["absolute_joint", "absolute_eef", "smooth_eef", "delta_eef"])
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-teaching-mode", action="store_true", help="Skip ctrl_mode guard. Use only for mock/debug.")
    parser.add_argument("--lowpass-alpha", type=float, default=0.7)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.15)
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend=args.camera_backend)
    config.enable_on_connect = True
    env = DualPiperXEnv(config)
    try:
        policy = load_policy(args.policy)
        limiter = ActionLimiter(max_joint_delta_rad=args.max_joint_delta_rad, lowpass_alpha=args.lowpass_alpha)
        runner = PolicyRunner(
            env,
            policy=policy,
            action_mode=args.action_mode,
            hz=args.hz,
            guard_motion_mode=not args.allow_teaching_mode,
            limiter=limiter,
        )
        runner.run(max_steps=args.max_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
