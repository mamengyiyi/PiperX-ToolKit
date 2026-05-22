#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.convert import convert_zarr_to_lerobot, dry_run


def split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PiperX ToolKit Zarr dataset to LeRobot v3.")
    parser.add_argument("--zarr", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default=None)
    parser.add_argument("--state", default="left_joint_pos,right_joint_pos")
    parser.add_argument("--action", default="action_left,action_right")
    parser.add_argument("--cameras", default="front,left_wrist,right_wrist")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.zarr, max_episodes=args.episodes)
        return
    if args.output is None or args.repo_id is None:
        raise SystemExit("--output and --repo-id are required unless --dry-run is set")
    convert_zarr_to_lerobot(
        zarr_path=args.zarr,
        output_dir=args.output,
        repo_id=args.repo_id,
        state_keys=split_csv(args.state),
        action_keys=split_csv(args.action),
        camera_names=split_csv(args.cameras),
        fps=args.fps,
        task=args.task,
        max_episodes=args.episodes,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
