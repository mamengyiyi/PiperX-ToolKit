from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_CAMERAS = ("front", "left_wrist", "right_wrist")
STATE_KEYS = (
    "left_eef_pos",
    "left_joint_pos",
    "left_joint_qvel",
    "left_joint_effort",
    "right_eef_pos",
    "right_joint_pos",
    "right_joint_qvel",
    "right_joint_effort",
)
ACTION_KEYS = ("action_left", "action_right")
META_KEYS = ("timestamp", "episode")


@dataclass
class ZarrSchemaConfig:
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    image_size: tuple[int, int] = (640, 480)
    task: str = ""
    hz: float = 30.0
    action_mode: str = "absolute_joint"
    action_shift_frames: int = 1


def _compressor() -> Any:
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        return None


def _require_array(group: Any, name: str, shape_tail: tuple[int, ...], dtype: Any, chunks: tuple[int, ...] | None = None) -> Any:
    if name in group:
        return group[name]
    compressor = _compressor()
    kwargs: dict[str, Any] = {
        "shape": (0, *shape_tail),
        "dtype": dtype,
        "chunks": chunks or (1, *shape_tail),
    }
    if compressor is not None:
        kwargs["compressor"] = compressor
    return group.create_dataset(name, **kwargs)


def open_or_create_dataset(path: str, config: ZarrSchemaConfig) -> tuple[Any, Any, int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "zarr is required for data collection. Install robot/data dependencies with "
            "`pip install -r requirements-robot.txt` or `pip install -e '.[data]'`."
        ) from exc

    exists = os.path.exists(path)
    try:
        root = zarr.open(path, mode="a", zarr_format=2)
    except TypeError:
        root = zarr.open(path, mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")
    ensure_schema(data, meta, config)
    meta.attrs["config"] = json.dumps(
        {
            "task": config.task,
            "hz": config.hz,
            "cameras": list(config.cameras),
            "image_size": list(config.image_size),
            "action_mode": config.action_mode,
            "action_from": "future_joint_pos",
            "action_shift_frames": config.action_shift_frames,
            "robot_type": "piperx_bimanual",
        },
        ensure_ascii=True,
    )
    start_episode = 0
    if exists and "episode" in data and len(data["episode"]) > 0:
        start_episode = int(np.max(data["episode"][:])) + 1
    return data, meta, start_episode


def ensure_schema(data: Any, meta: Any, config: ZarrSchemaConfig) -> None:
    width, height = config.image_size
    image_shape = (3, height, width)
    for cam in config.cameras:
        _require_array(data, f"rgb_{cam}", image_shape, np.uint8, chunks=(1, *image_shape))

    for key in STATE_KEYS:
        _require_array(data, key, (7,), np.float32, chunks=(1024, 7))
    for key in ACTION_KEYS:
        _require_array(data, key, (7,), np.float32, chunks=(1024, 7))

    _require_array(data, "timestamp", (), np.float64, chunks=(4096,))
    _require_array(data, "episode", (), np.uint32, chunks=(4096,))
    if "episode_ends" not in meta:
        meta.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(1024,))


def append_array(array: Any, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.shape[0] == 0:
        return
    old = int(array.shape[0])
    new_shape = (old + values.shape[0], *array.shape[1:])
    array.resize(new_shape)
    array[old : old + values.shape[0]] = values


def append_episode(data: Any, meta: Any, episode: dict[str, np.ndarray]) -> None:
    for key, values in episode.items():
        append_array(data[key], values)
    recompute_episode_ends(data, meta)


def recompute_episode_ends(data: Any, meta: Any) -> None:
    if "episode" not in data or len(data["episode"]) == 0:
        return
    ep = data["episode"][:]
    ends = []
    running = 0
    for episode_id in np.unique(ep):
        running += int(np.sum(ep == episode_id))
        ends.append(running)
    arr = meta["episode_ends"]
    arr.resize((len(ends),))
    arr[:] = np.asarray(ends, dtype=np.uint32)


def episode_ranges(data: Any, meta: Any, max_episodes: int | None = None) -> list[tuple[int, int]]:
    if "episode_ends" in meta and len(meta["episode_ends"]) > 0:
        ends = list(map(int, meta["episode_ends"][:]))
    else:
        ep = data["episode"][:]
        ends = []
        running = 0
        for episode_id in np.unique(ep):
            running += int(np.sum(ep == episode_id))
            ends.append(running)
    if max_episodes is not None:
        ends = ends[:max_episodes]
    ranges = []
    start = 0
    for end in ends:
        ranges.append((start, end))
        start = end
    return ranges
