from __future__ import annotations

from typing import Any

import numpy as np


def compressor() -> Any:
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        return None


def require_array(
    group: Any,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: Any,
    chunks: tuple[int, ...] | None = None,
) -> Any:
    if name in group:
        return group[name]
    kwargs: dict[str, Any] = {
        "shape": (0, *shape_tail),
        "dtype": dtype,
        "chunks": chunks or (1, *shape_tail),
    }
    comp = compressor()
    if comp is not None:
        kwargs["compressor"] = comp
    return group.create_dataset(name, **kwargs)


def append_array(array: Any, values: np.ndarray) -> None:
    old = int(array.shape[0])
    array.resize((old + values.shape[0], *array.shape[1:]))
    array[old : old + values.shape[0]] = values


def recompute_episode_ends(data: Any, meta: Any) -> None:
    ep = data["episode"][:]
    ends = []
    running = 0
    for episode_id in np.unique(ep):
        running += int(np.sum(ep == episode_id))
        ends.append(running)
    arr = meta["episode_ends"]
    arr.resize((len(ends),))
    arr[:] = np.asarray(ends, dtype=np.uint32)


def append_episode(data: Any, meta: Any, episode: dict[str, np.ndarray]) -> None:
    for key, values in episode.items():
        append_array(data[key], values)
    recompute_episode_ends(data, meta)


def next_episode_id(data: Any) -> int:
    if "episode" in data and len(data["episode"]) > 0:
        return int(np.max(data["episode"][:])) + 1
    return 0
