#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Read arms and cameras for a short fixed duration.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="mock", choices=["sdk", "mock"])
    parser.add_argument("--camera-backend", default="mock", choices=["mock", "opencv"])
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend=args.camera_backend)
    if args.left_can:
        config.left_can = args.left_can
    if args.right_can:
        config.right_can = args.right_can
    env = DualPiperXEnv(config)
    count = 0
    t0 = time.time()
    try:
        last_obs = {}
        while time.time() - t0 < args.duration:
            last_obs = env.get_observation()
            count += 1
        elapsed = time.time() - t0
        print(f"Read {count} observations in {elapsed:.2f}s ({count / max(elapsed, 1e-6):.1f} Hz)")
        for key, value in sorted(last_obs.items()):
            print(f"{key:24s} shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
