from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from piperx_toolkit.collect.schema import episode_ranges

IMAGE_PREFIX = "rgb_"
EXCLUDE_NUMERIC = {"timestamp", "episode"}


def _open_zarr(path: str) -> tuple[Any, Any]:
    import zarr

    root = zarr.open(path, mode="r")
    return root["data"], root["meta"]


def _meta_config(meta: Any) -> dict[str, Any]:
    raw = meta.attrs.get("config", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def discover(data: Any) -> tuple[list[str], list[str]]:
    numeric = []
    cameras = []
    for key in sorted(data.keys()):
        if key.startswith(IMAGE_PREFIX):
            cameras.append(key.removeprefix(IMAGE_PREFIX))
        elif key not in EXCLUDE_NUMERIC:
            numeric.append(key)
    return numeric, cameras


def dry_run(zarr_path: str, max_episodes: int | None = None) -> None:
    data, meta = _open_zarr(zarr_path)
    ranges = episode_ranges(data, meta, max_episodes=max_episodes)
    numeric, cameras = discover(data)
    print(f"Zarr: {zarr_path}")
    print(f"Episodes: {len(ranges)}")
    print(f"Frames: {sum(end - start for start, end in ranges)}")
    print(f"Config: {_meta_config(meta)}")
    print("\nNumeric arrays:")
    for key in numeric:
        arr = data[key]
        print(f"  {key:24s} shape={arr.shape} dtype={arr.dtype}")
    print("\nCameras:")
    for cam in cameras:
        arr = data[f"rgb_{cam}"]
        print(f"  {cam:24s} shape={arr.shape} dtype={arr.dtype}")


def _dim(data: Any, keys: list[str]) -> int:
    total = 0
    for key in keys:
        shape = data[key].shape
        total += int(shape[1]) if len(shape) > 1 else 1
    return total


def _read_concat(data: Any, keys: list[str], start: int, end: int) -> np.ndarray:
    arrays = []
    for key in keys:
        arr = data[key][start:end].astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)


def convert_zarr_to_lerobot(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    camera_names: list[str] | None = None,
    fps: int = 30,
    task: str | None = None,
    robot_type: str = "piperx_bimanual",
    max_episodes: int | None = None,
    use_videos: bool = False,
    overwrite: bool = False,
) -> Any:
    from PIL import Image

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    data, meta = _open_zarr(zarr_path)
    numeric, cameras = discover(data)
    state_keys = state_keys or ["left_joint_pos", "right_joint_pos"]
    action_keys = action_keys or ["action_left", "action_right"]
    camera_names = camera_names or cameras
    task = task or _meta_config(meta).get("task") or Path(zarr_path).stem

    for key in state_keys + action_keys:
        if key not in data:
            raise KeyError(f"Missing Zarr array: {key}. Available numeric arrays: {numeric}")
    for cam in camera_names:
        if f"rgb_{cam}" not in data:
            raise KeyError(f"Missing camera rgb_{cam}. Available cameras: {cameras}")

    ranges = episode_ranges(data, meta, max_episodes=max_episodes)
    state_dim = _dim(data, state_keys)
    action_dim = _dim(data, action_keys)
    first_cam = camera_names[0]
    _, channels, height, width = data[f"rgb_{first_cam}"].shape
    image_shape = (height, width, channels)

    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass overwrite=True or --overwrite.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output,
        use_videos=use_videos,
        image_writer_threads=4,
    )

    for ep_idx, (start, end) in enumerate(ranges):
        state_batch = _read_concat(data, state_keys, start, end)
        action_batch = _read_concat(data, action_keys, start, end)
        camera_batches = {cam: data[f"rgb_{cam}"][start:end] for cam in camera_names}

        for i in range(end - start):
            frame: dict[str, Any] = {
                "observation.state": state_batch[i],
                "action": action_batch[i],
                "task": task,
            }
            for cam in camera_names:
                img = camera_batches[cam][i].transpose(1, 2, 0)
                frame[f"observation.images.{cam}"] = img if use_videos else Image.fromarray(img)
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"Converted episode {ep_idx}: {end - start} frames")

    return dataset

