#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump real Piper SDK message structure for adapter tuning.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--out", default="sdk_msgs.json")
    args = parser.parse_args()
    setup_logging()

    config = load_env_config(args.config, backend=args.backend, camera_backend="mock")
    if args.left_can:
        config.left_can = args.left_can
    if args.right_can:
        config.right_can = args.right_can
    env = DualPiperXEnv(config)
    try:
        snapshot = env.diagnostics_snapshot()
        Path(args.out).write_text(json.dumps(snapshot, indent=2, ensure_ascii=True))
        print(f"Wrote {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
