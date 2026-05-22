#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.collect import Collector
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one short fixed-duration episode.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="mock", choices=["sdk", "mock"])
    parser.add_argument("--camera-backend", default="mock", choices=["mock", "opencv"])
    parser.add_argument("--dataset", default="datasets/smoke.zarr")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend=args.camera_backend)
    env = DualPiperXEnv(config)
    try:
        collector = Collector(env, dataset_path=args.dataset, hz=args.hz, task="smoke test")
        stats = collector.collect_fixed_duration(args.duration)
        print(f"Saved smoke episode: {stats.steps} steps, {stats.fps:.1f} FPS -> {args.dataset}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
