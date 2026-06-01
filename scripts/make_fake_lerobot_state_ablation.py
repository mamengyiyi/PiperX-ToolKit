#!/usr/bin/env python3
"""Generate tiny LeRobot datasets with and without observation.state.

This creates two deterministic PiperX-shaped fake datasets:

* <repo-prefix>_with_state: has observation.state, action, and three RGB views.
* <repo-prefix>_no_state: has action and three RGB views, but no observation.state.

Use them as a direct OpenPI ablation: the same training config should accept the
with-state dataset and fail on the no-state dataset when the config repacks
`observation.state`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CAMERAS = ("front", "left_wrist", "right_wrist")


def _import_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _repo_path(root: Path, repo_id: str) -> Path:
    return root / repo_id


def _fake_image(ep: int, frame: int, cam_index: int, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    base = (ep * 37 + frame * 11 + cam_index * 53) % 256
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = (xx + base) % 256
    rgb[..., 1] = (yy + base * 2) % 256
    rgb[..., 2] = ((xx // 2 + yy // 3 + base * 3) % 256).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _fake_state(ep: int, frame: int, state_dim: int) -> np.ndarray:
    values = np.linspace(-0.2, 0.2, state_dim, dtype=np.float32)
    return values + np.float32(ep * 0.05 + frame * 0.01)


def _fake_action(ep: int, frame: int, action_dim: int) -> np.ndarray:
    values = np.linspace(0.1, -0.1, action_dim, dtype=np.float32)
    return values + np.float32(ep * 0.03 - frame * 0.005)


def _make_features(has_state: bool, action_dim: int, state_dim: int, image_size: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
    }
    if has_state:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _create_one_dataset(
    *,
    root: Path,
    repo_id: str,
    has_state: bool,
    overwrite: bool,
    episodes: int,
    frames_per_episode: int,
    fps: int,
    image_size: int,
    state_dim: int,
    action_dim: int,
    task: str,
) -> None:
    LeRobotDataset = _import_lerobot_dataset()
    dataset_root = _repo_path(root, repo_id)
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dataset_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    features = _make_features(has_state, action_dim, state_dim, image_size)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="piperx_bimanual_fake",
        features=features,
        root=dataset_root,
        use_videos=False,
        image_writer_threads=1,
    )

    for ep in range(episodes):
        for frame_idx in range(frames_per_episode):
            frame: dict[str, Any] = {
                "action": _fake_action(ep, frame_idx, action_dim),
                "task": task,
            }
            if has_state:
                frame["observation.state"] = _fake_state(ep, frame_idx, state_dim)
            for cam_index, cam in enumerate(CAMERAS):
                frame[f"observation.images.{cam}"] = _fake_image(ep, frame_idx, cam_index, image_size)
            dataset.add_frame(frame)
        dataset.save_episode()

    state_text = "with observation.state" if has_state else "WITHOUT observation.state"
    print(f"Created {repo_id} ({state_text}) at {dataset_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny with-state/no-state LeRobot datasets for OpenPI ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=None,
        help="LeRobot root. Defaults to $HF_LEROBOT_HOME or ./lerobot_datasets.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="mamengyiyi/piperx_fake_state_ablation",
        help="Output repo prefix. Suffixes _with_state and _no_state are appended.",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes per fake dataset.")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=12,
        help="Frames per episode. Keep >= action_horizon for OpenPI smoke tests.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--task", default="piperx fake state ablation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    import os

    args = parse_args()
    root = Path(args.root or os.environ.get("HF_LEROBOT_HOME", "lerobot_datasets")).expanduser().resolve()
    with_state_repo = f"{args.repo_prefix}_with_state"
    no_state_repo = f"{args.repo_prefix}_no_state"

    common = dict(
        root=root,
        overwrite=args.overwrite,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        fps=args.fps,
        image_size=args.image_size,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        task=args.task,
    )
    _create_one_dataset(repo_id=with_state_repo, has_state=True, **common)
    _create_one_dataset(repo_id=no_state_repo, has_state=False, **common)

    print("\nUse these repo_ids for the OpenPI comparison:")
    print(f"  with state: {with_state_repo}")
    print(f"  no state  : {no_state_repo}")


if __name__ == "__main__":
    main()
