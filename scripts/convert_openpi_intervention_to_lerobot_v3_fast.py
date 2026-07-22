#!/usr/bin/env python3
"""Convert PiperX OpenPI intervention Zarr data into two LeRobot v3 views.

The full view preserves every rollout frame. The intervention view contains one
LeRobot episode for each contiguous human-intervention run. Both views use the
same standard training fields as the existing PiperX converter and add metadata
needed for DAgger/IWR processing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CAMERA_DEFAULTS = ("front", "left_wrist", "right_wrist")
TOWEL_TYPE_IDS = {"square": 0, "small_rectangle": 1, "large_rectangle": 2}
CONTROL_POLICY = 0
CONTROL_INTERVENTION = 1
UNKNOWN_SUCCESS = 255
SCHEMA_VERSION = 1

VECTOR_KEYS = (
    "observation.state",
    "action",
    "piperx.policy_action",
    "piperx.human_action",
    "piperx.executed_action",
)
SCALAR_KEYS = (
    "piperx.control_source",
    "piperx.intervention_mask",
    "piperx.episode_success",
    "piperx.policy_action_valid",
    "piperx.human_action_valid",
    "piperx.original_episode_index",
    "piperx.original_frame_index",
    "piperx.intervention_segment_index",
    "piperx.source_id",
    "piperx.towel_type_id",
)

REQUIRED_VECTOR_ARRAYS = (
    "left_joint_pos",
    "right_joint_pos",
    "policy_action_left",
    "policy_action_right",
    "expert_action_left",
    "expert_action_right",
    "executed_action_left",
    "executed_action_right",
)
REQUIRED_SCALAR_ARRAYS = (
    "intervention_mask",
    "control_source",
    "episode_success",
    "episode",
    "timestamp",
)


@dataclasses.dataclass(frozen=True)
class EpisodeRange:
    output_episode_index: int
    original_episode_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class SourceAudit:
    episode_ranges: tuple[EpisodeRange, ...]
    total_source_frames: int
    selected_frames: int
    intervention_frames: int
    intervention_segments: int
    success_episodes: int
    failure_episodes: int
    unknown_episodes: int
    image_shape: tuple[int, int, int]


@dataclasses.dataclass
class EpisodeBatch:
    state: np.ndarray
    action: np.ndarray
    policy_action: np.ndarray
    human_action: np.ndarray
    executed_action: np.ndarray
    control_source: np.ndarray
    intervention_mask: np.ndarray
    episode_success: np.ndarray
    policy_action_valid: np.ndarray
    human_action_valid: np.ndarray
    original_episode_index: np.ndarray
    original_frame_index: np.ndarray
    intervention_segment_index: np.ndarray
    source_id: np.ndarray
    towel_type_id: np.ndarray
    images: dict[str, Sequence[Any] | np.ndarray]

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _open_zarr(path: Path):
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required") from exc
    try:
        return zarr.open_group(str(path), mode="r")
    except AttributeError:
        return zarr.open(str(path), mode="r")


def _read_meta_config(meta_group: Any) -> dict[str, Any]:
    raw = meta_group.attrs.get("config")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _require_shape(data: Any, key: str, expected: tuple[int | None, ...], total_frames: int) -> None:
    if key not in data:
        raise KeyError(f"Missing required Zarr array: data/{key}")
    shape = tuple(int(x) for x in data[key].shape)
    if not shape or shape[0] != total_frames:
        raise ValueError(f"data/{key} length mismatch: shape={shape}, expected first dimension {total_frames}")
    if len(shape) != len(expected):
        raise ValueError(f"data/{key} rank mismatch: shape={shape}, expected={expected}")
    for actual, wanted in zip(shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            raise ValueError(f"data/{key} shape mismatch: shape={shape}, expected={expected}")


def _episode_ranges(data: Any, meta: Any, max_episodes: int | None) -> tuple[EpisodeRange, ...]:
    if "episode_ends" not in meta:
        raise KeyError("Missing required Zarr array: meta/episode_ends")
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0:
        raise ValueError("meta/episode_ends must be a non-empty 1D array")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("meta/episode_ends must be strictly increasing and positive")

    total_frames = int(data["intervention_mask"].shape[0])
    if int(ends[-1]) != total_frames:
        raise ValueError(f"meta/episode_ends[-1]={int(ends[-1])} does not match data length {total_frames}")

    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--episodes must be positive")
        ends = ends[:max_episodes]

    starts = np.concatenate([np.array([0], dtype=np.int64), ends[:-1]])
    ranges: list[EpisodeRange] = []
    for output_ep, (start, end) in enumerate(zip(starts, ends, strict=True)):
        episode_values = np.asarray(data["episode"][int(start) : int(end)], dtype=np.int64)
        if episode_values.size == 0 or np.any(episode_values != episode_values[0]):
            raise ValueError(f"data/episode is not constant inside episode range {output_ep}")
        ranges.append(
            EpisodeRange(
                output_episode_index=output_ep,
                original_episode_index=int(episode_values[0]),
                start=int(start),
                end=int(end),
            )
        )
    return tuple(ranges)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"Intervention mask must be 1D, got {mask.shape}")
    if not mask.any():
        return []
    starts = np.flatnonzero(mask & np.concatenate([np.array([True]), ~mask[:-1]]))
    ends = np.flatnonzero(mask & np.concatenate([~mask[1:], np.array([True])])) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def audit_source(
    zarr_path: Path,
    camera_names: Sequence[str],
    max_episodes: int | None = None,
) -> SourceAudit:
    root = _open_zarr(zarr_path)
    if "data" not in root or "meta" not in root:
        raise KeyError("Zarr root must contain data and meta groups")
    data = root["data"]
    meta = root["meta"]

    if "intervention_mask" not in data:
        raise KeyError("Missing required Zarr array: data/intervention_mask")
    total_frames = int(data["intervention_mask"].shape[0])
    for key in REQUIRED_VECTOR_ARRAYS:
        _require_shape(data, key, (None, 7), total_frames)
    for key in REQUIRED_SCALAR_ARRAYS:
        _require_shape(data, key, (None,), total_frames)

    image_shape: tuple[int, int, int] | None = None
    for camera in camera_names:
        key = f"rgb_{camera}"
        _require_shape(data, key, (None, 3, None, None), total_frames)
        _, channels, height, width = tuple(int(x) for x in data[key].shape)
        current_shape = (height, width, channels)
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(f"Camera image shapes differ: {image_shape} vs {key}={current_shape}")
    if image_shape is None:
        raise ValueError("At least one camera is required")

    ranges = _episode_ranges(data, meta, max_episodes)
    intervention_frames = 0
    intervention_segments = 0
    success_count = 0
    failure_count = 0
    unknown_count = 0

    for episode in ranges:
        start, end = episode.start, episode.end
        mask = np.asarray(data["intervention_mask"][start:end], dtype=np.uint8)
        control = np.asarray(data["control_source"][start:end], dtype=np.int64)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid intervention_mask value in source episode {episode.original_episode_index}")
        if not np.isin(control, [CONTROL_POLICY, CONTROL_INTERVENTION]).all():
            values = sorted(set(int(x) for x in np.unique(control)))
            raise ValueError(f"Invalid control_source values {values} in source episode {episode.original_episode_index}")
        if not np.array_equal(mask.astype(bool), control == CONTROL_INTERVENTION):
            raise ValueError(f"intervention_mask/control_source mismatch in source episode {episode.original_episode_index}")

        success = np.asarray(data["episode_success"][start:end], dtype=np.int64)
        if success.size == 0 or np.any(success != success[0]):
            raise ValueError(f"episode_success is not constant in source episode {episode.original_episode_index}")
        label = int(success[0])
        if label == 1:
            success_count += 1
        elif label == 0:
            failure_count += 1
        elif label == UNKNOWN_SUCCESS:
            unknown_count += 1
        else:
            raise ValueError(f"Invalid episode_success={label} in source episode {episode.original_episode_index}")

        state = _concat_pair(data, "left_joint_pos", "right_joint_pos", start, end)
        executed = _concat_pair(data, "executed_action_left", "executed_action_right", start, end)
        policy = _concat_pair(data, "policy_action_left", "policy_action_right", start, end)
        human = _concat_pair(data, "expert_action_left", "expert_action_right", start, end)
        if not np.isfinite(state).all():
            raise ValueError(f"Non-finite state in source episode {episode.original_episode_index}")
        if not np.isfinite(executed).all():
            raise ValueError(f"Non-finite executed action in source episode {episode.original_episode_index}")
        policy_mask = control == CONTROL_POLICY
        intervention_mask = control == CONTROL_INTERVENTION
        if policy_mask.any() and not np.isfinite(policy[policy_mask]).all():
            raise ValueError(f"Non-finite policy action on policy frames in source episode {episode.original_episode_index}")
        if intervention_mask.any() and not np.isfinite(human[intervention_mask]).all():
            raise ValueError(f"Non-finite expert action on intervention frames in source episode {episode.original_episode_index}")
        if intervention_mask.any() and not np.allclose(
            human[intervention_mask], executed[intervention_mask], rtol=0.0, atol=1e-6
        ):
            max_delta = float(np.max(np.abs(human[intervention_mask] - executed[intervention_mask])))
            raise ValueError(
                f"expert_action differs from executed_action on intervention frames in source episode "
                f"{episode.original_episode_index}; max delta={max_delta}"
            )

        runs = _contiguous_runs(intervention_mask)
        intervention_frames += int(intervention_mask.sum())
        intervention_segments += len(runs)

    return SourceAudit(
        episode_ranges=ranges,
        total_source_frames=total_frames,
        selected_frames=sum(episode.length for episode in ranges),
        intervention_frames=intervention_frames,
        intervention_segments=intervention_segments,
        success_episodes=success_count,
        failure_episodes=failure_count,
        unknown_episodes=unknown_count,
        image_shape=image_shape,
    )


def _concat_pair(data: Any, left_key: str, right_key: str, start: int, end: int) -> np.ndarray:
    left = np.asarray(data[left_key][start:end], dtype=np.float32)
    right = np.asarray(data[right_key][start:end], dtype=np.float32)
    return np.concatenate([left, right], axis=1)


def _filled_with_valid_mask(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values).all(axis=1)
    filled = np.where(np.isfinite(values), values, np.float32(0.0)).astype(np.float32, copy=False)
    return filled, valid.astype(np.int64)


def _segment_ids(intervention_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    runs = _contiguous_runs(intervention_mask)
    ids = np.full(len(intervention_mask), -1, dtype=np.int64)
    for segment_index, (start, end) in enumerate(runs):
        ids[start:end] = segment_index
    return ids, runs


def _prepare_images(
    data: Any,
    camera_names: Sequence[str],
    start: int,
    end: int,
    use_videos: bool,
    image_writer_threads: int,
) -> dict[str, Sequence[Any] | np.ndarray]:
    camera_arrays = {
        camera: np.asarray(data[f"rgb_{camera}"][start:end], dtype=np.uint8).transpose(0, 2, 3, 1)
        for camera in camera_names
    }
    if use_videos:
        return camera_arrays

    from PIL import Image

    workers = max(1, min(image_writer_threads, len(camera_names) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            camera: executor.submit(lambda array: [Image.fromarray(frame) for frame in array], array)
            for camera, array in camera_arrays.items()
        }
        return {camera: future.result() for camera, future in futures.items()}


def _load_episode_batch(
    data: Any,
    episode: EpisodeRange,
    camera_names: Sequence[str],
    source_id: int,
    towel_type_id: int,
    use_videos: bool,
    image_writer_threads: int,
) -> tuple[EpisodeBatch, list[tuple[int, int]]]:
    start, end = episode.start, episode.end
    state = _concat_pair(data, "left_joint_pos", "right_joint_pos", start, end)
    executed = _concat_pair(data, "executed_action_left", "executed_action_right", start, end)
    policy_raw = _concat_pair(data, "policy_action_left", "policy_action_right", start, end)
    human_raw = _concat_pair(data, "expert_action_left", "expert_action_right", start, end)
    policy, policy_valid = _filled_with_valid_mask(policy_raw)
    human, human_valid = _filled_with_valid_mask(human_raw)
    control = np.asarray(data["control_source"][start:end], dtype=np.int64)
    intervention = np.asarray(data["intervention_mask"][start:end], dtype=np.int64)
    segment_ids, runs = _segment_ids(intervention.astype(bool))
    success = np.asarray(data["episode_success"][start:end], dtype=np.int64)
    length = end - start

    batch = EpisodeBatch(
        state=state,
        action=executed.copy(),
        policy_action=policy,
        human_action=human,
        executed_action=executed,
        control_source=control,
        intervention_mask=intervention,
        episode_success=success,
        policy_action_valid=policy_valid,
        human_action_valid=human_valid,
        original_episode_index=np.full(length, episode.original_episode_index, dtype=np.int64),
        original_frame_index=np.arange(length, dtype=np.int64),
        intervention_segment_index=segment_ids,
        source_id=np.full(length, source_id, dtype=np.int64),
        towel_type_id=np.full(length, towel_type_id, dtype=np.int64),
        images=_prepare_images(data, camera_names, start, end, use_videos, image_writer_threads),
    )
    return batch, runs


def _feature_schema(image_shape: tuple[int, int, int], camera_names: Sequence[str], use_videos: bool) -> dict[str, dict]:
    features: dict[str, dict] = {
        key: {"dtype": "float32", "shape": (14,), "names": [key.split(".")[-1]]} for key in VECTOR_KEYS
    }
    features.update(
        {key: {"dtype": "int64", "shape": (1,), "names": [key.split(".")[-1]]} for key in SCALAR_KEYS}
    )
    image_dtype = "video" if use_videos else "image"
    for camera in camera_names:
        features[f"observation.images.{camera}"] = {
            "dtype": image_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    return features


def _frame_from_batch(batch: EpisodeBatch, index: int, camera_names: Sequence[str], task: str) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": batch.state[index],
        "action": batch.action[index],
        "piperx.policy_action": batch.policy_action[index],
        "piperx.human_action": batch.human_action[index],
        "piperx.executed_action": batch.executed_action[index],
        "piperx.control_source": np.asarray([batch.control_source[index]], dtype=np.int64),
        "piperx.intervention_mask": np.asarray([batch.intervention_mask[index]], dtype=np.int64),
        "piperx.episode_success": np.asarray([batch.episode_success[index]], dtype=np.int64),
        "piperx.policy_action_valid": np.asarray([batch.policy_action_valid[index]], dtype=np.int64),
        "piperx.human_action_valid": np.asarray([batch.human_action_valid[index]], dtype=np.int64),
        "piperx.original_episode_index": np.asarray([batch.original_episode_index[index]], dtype=np.int64),
        "piperx.original_frame_index": np.asarray([batch.original_frame_index[index]], dtype=np.int64),
        "piperx.intervention_segment_index": np.asarray(
            [batch.intervention_segment_index[index]], dtype=np.int64
        ),
        "piperx.source_id": np.asarray([batch.source_id[index]], dtype=np.int64),
        "piperx.towel_type_id": np.asarray([batch.towel_type_id[index]], dtype=np.int64),
        "task": task,
    }
    for camera in camera_names:
        frame[f"observation.images.{camera}"] = batch.images[camera][index]
    return frame


def _create_dataset(
    root: Path,
    repo_id: str,
    features: dict[str, dict],
    fps: int,
    robot_type: str,
    use_videos: bool,
    streaming_encoding: bool,
    image_writer_threads: int,
):
    LeRobotDataset = _load_lerobot_dataset_class()
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=root,
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        streaming_encoding=streaming_encoding if use_videos else False,
    )


def _safe_finalize(dataset: Any) -> None:
    try:
        dataset.finalize()
    except Exception:
        pass


def _remove_intermediate_images(root: Path) -> None:
    image_dir = root / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir, ignore_errors=True)


def _conversion_manifest(
    *,
    view: str,
    zarr_path: Path,
    final_output: Path,
    repo_id: str,
    task: str,
    fps: int,
    source_id: int,
    towel_type: str,
    camera_names: Sequence[str],
    audit: SourceAudit,
    mappings: list[dict[str, int]],
) -> dict[str, Any]:
    if view == "full":
        output_episodes = len(audit.episode_ranges)
        output_frames = audit.selected_frames
    else:
        output_episodes = audit.intervention_segments
        output_frames = audit.intervention_frames
    return {
        "schema_version": SCHEMA_VERSION,
        "view": view,
        "source_zarr": str(zarr_path.resolve()),
        "output": str(final_output.resolve()),
        "repo_id": repo_id,
        "task": task,
        "fps": fps,
        "source_id": source_id,
        "towel_type": towel_type,
        "towel_type_id": TOWEL_TYPE_IDS[towel_type],
        "standard_fields": {
            "observation.state": "left_joint_pos + right_joint_pos",
            "action": "executed_action_left + executed_action_right",
            "images": [f"observation.images.{camera}" for camera in camera_names],
        },
        "missing_action_policy": "replace non-finite policy/expert values with zero and preserve validity masks",
        "source_statistics": {
            "selected_episodes": len(audit.episode_ranges),
            "selected_frames": audit.selected_frames,
            "intervention_frames": audit.intervention_frames,
            "intervention_segments": audit.intervention_segments,
            "success_episodes": audit.success_episodes,
            "failure_episodes": audit.failure_episodes,
            "unknown_episodes": audit.unknown_episodes,
        },
        "output_statistics": {"episodes": output_episodes, "frames": output_frames},
        "episode_mapping": mappings,
    }


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "piperx_conversion.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_output(root: Path, repo_id: str, expected_episodes: int, expected_frames: int, samples: int) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise ValueError(f"{root}: total_episodes mismatch")
    if int(info.get("total_frames", -1)) != expected_frames:
        raise ValueError(f"{root}: total_frames mismatch")
    required = set(VECTOR_KEYS) | set(SCALAR_KEYS)
    missing = required - set(info.get("features", {}))
    if missing:
        raise ValueError(f"{root}: missing features {sorted(missing)}")
    if expected_frames == 0 or samples <= 0:
        return

    LeRobotDataset = _load_lerobot_dataset_class()
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if len(dataset) != expected_frames:
        raise ValueError(f"{root}: reader length {len(dataset)} != {expected_frames}")
    indices = np.linspace(0, expected_frames - 1, min(samples, expected_frames), dtype=np.int64)
    for index in np.unique(indices):
        item = dataset[int(index)]
        action = np.asarray(item["action"], dtype=np.float32)
        executed = np.asarray(item["piperx.executed_action"], dtype=np.float32)
        if action.shape != (14,) or not np.array_equal(action, executed):
            raise ValueError(f"{root}: action/executed_action mismatch at frame {int(index)}")
        if not np.isfinite(np.asarray(item["observation.state"], dtype=np.float32)).all():
            raise ValueError(f"{root}: non-finite state at frame {int(index)}")
        for feature in info["features"]:
            if feature.startswith("observation.images."):
                image = np.asarray(item[feature])
                if image.ndim != 3:
                    raise ValueError(f"{root}: bad decoded image shape for {feature}: {image.shape}")


def _temp_path(final_path: Path, label: str) -> Path:
    return final_path.with_name(f".{final_path.name}.tmp-{label}-{os.getpid()}-{int(time.time())}")


def _backup_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.old-{os.getpid()}-{int(time.time())}")


def _publish_pair(staged: Sequence[Path], finals: Sequence[Path], overwrite: bool) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for final in finals:
            if final.exists():
                if not overwrite:
                    raise FileExistsError(f"Output exists: {final}. Pass --overwrite to replace it.")
                backup = _backup_path(final)
                final.rename(backup)
                backups.append((backup, final))
        for stage, final in zip(staged, finals, strict=True):
            stage.rename(final)
            published.append((final, stage))
    except Exception:
        for final, stage in reversed(published):
            if final.exists():
                final.rename(stage)
        for backup, final in reversed(backups):
            if backup.exists() and not final.exists():
                backup.rename(final)
        raise
    for backup, _ in backups:
        shutil.rmtree(backup)


def convert_dual_views(
    *,
    zarr_path: Path,
    output_path: Path,
    repo_id: str,
    towel_type: str,
    source_id: int,
    task: str,
    fps: int = 30,
    robot_type: str = "piperx_bimanual",
    camera_names: Sequence[str] = CAMERA_DEFAULTS,
    max_episodes: int | None = None,
    use_videos: bool = True,
    streaming_encoding: bool = True,
    image_writer_threads: int = 8,
    overwrite: bool = False,
    validate_samples: int = 16,
) -> tuple[Path, Path]:
    audit = audit_source(zarr_path, camera_names, max_episodes)
    intervention_output = output_path.with_name(f"{output_path.name}_intervention")
    intervention_repo_id = f"{repo_id}_intervention"
    finals = (output_path.resolve(), intervention_output.resolve())
    if finals[0] == finals[1]:
        raise ValueError("Full and intervention output paths must differ")
    for final in finals:
        if final.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {final}. Pass --overwrite to replace it.")
        final.parent.mkdir(parents=True, exist_ok=True)

    staged = (_temp_path(finals[0], "full"), _temp_path(finals[1], "intervention"))
    for path in staged:
        if path.exists():
            shutil.rmtree(path)

    root = _open_zarr(zarr_path)
    data = root["data"]
    features = _feature_schema(audit.image_shape, camera_names, use_videos)
    full_dataset = None
    intervention_dataset = None
    full_mappings: list[dict[str, int]] = []
    intervention_mappings: list[dict[str, int]] = []
    intervention_output_episode = 0
    started = time.monotonic()

    try:
        full_dataset = _create_dataset(
            staged[0], repo_id, features, fps, robot_type, use_videos, streaming_encoding, image_writer_threads
        )
        intervention_dataset = _create_dataset(
            staged[1],
            intervention_repo_id,
            features,
            fps,
            robot_type,
            use_videos,
            streaming_encoding,
            image_writer_threads,
        )

        for episode in audit.episode_ranges:
            batch, runs = _load_episode_batch(
                data,
                episode,
                camera_names,
                source_id,
                TOWEL_TYPE_IDS[towel_type],
                use_videos,
                image_writer_threads,
            )
            for index in range(batch.length):
                full_dataset.add_frame(_frame_from_batch(batch, index, camera_names, task))
            full_dataset.save_episode()
            full_mappings.append(
                {
                    "output_episode_index": episode.output_episode_index,
                    "original_episode_index": episode.original_episode_index,
                    "original_start_frame": 0,
                    "original_end_frame": episode.length,
                }
            )

            for segment_index, (segment_start, segment_end) in enumerate(runs):
                for index in range(segment_start, segment_end):
                    intervention_dataset.add_frame(_frame_from_batch(batch, index, camera_names, task))
                intervention_dataset.save_episode()
                intervention_mappings.append(
                    {
                        "output_episode_index": intervention_output_episode,
                        "original_episode_index": episode.original_episode_index,
                        "intervention_segment_index": segment_index,
                        "original_start_frame": segment_start,
                        "original_end_frame": segment_end,
                    }
                )
                intervention_output_episode += 1
            print(
                f"Episode {episode.output_episode_index}: frames={episode.length} "
                f"intervention_segments={len(runs)}"
            )

        full_dataset.finalize()
        intervention_dataset.finalize()
        full_dataset = None
        intervention_dataset = None
        if use_videos:
            _remove_intermediate_images(staged[0])
            _remove_intermediate_images(staged[1])

        _write_manifest(
            staged[0],
            _conversion_manifest(
                view="full",
                zarr_path=zarr_path,
                final_output=finals[0],
                repo_id=repo_id,
                task=task,
                fps=fps,
                source_id=source_id,
                towel_type=towel_type,
                camera_names=camera_names,
                audit=audit,
                mappings=full_mappings,
            ),
        )
        _write_manifest(
            staged[1],
            _conversion_manifest(
                view="intervention",
                zarr_path=zarr_path,
                final_output=finals[1],
                repo_id=intervention_repo_id,
                task=task,
                fps=fps,
                source_id=source_id,
                towel_type=towel_type,
                camera_names=camera_names,
                audit=audit,
                mappings=intervention_mappings,
            ),
        )
        _validate_output(
            staged[0], repo_id, len(audit.episode_ranges), audit.selected_frames, validate_samples
        )
        _validate_output(
            staged[1],
            intervention_repo_id,
            audit.intervention_segments,
            audit.intervention_frames,
            validate_samples,
        )
        _publish_pair(staged, finals, overwrite)
    except Exception:
        if full_dataset is not None:
            _safe_finalize(full_dataset)
        if intervention_dataset is not None:
            _safe_finalize(intervention_dataset)
        for path in staged:
            shutil.rmtree(path, ignore_errors=True)
        raise

    elapsed = time.monotonic() - started
    print(f"Published full dataset: {finals[0]}")
    print(f"Published intervention dataset: {finals[1]}")
    print(
        f"Converted {audit.selected_frames} full frames and {audit.intervention_frames} intervention frames "
        f"in {elapsed:.1f}s"
    )
    return finals


def _print_audit(zarr_path: Path, audit: SourceAudit, camera_names: Sequence[str]) -> None:
    print(f"Zarr: {zarr_path.resolve()}")
    print(f"Episodes: {len(audit.episode_ranges)}")
    print(f"Full frames: {audit.selected_frames}")
    print(f"Intervention frames: {audit.intervention_frames}")
    print(f"Intervention segments: {audit.intervention_segments}")
    print(
        f"Labels: success={audit.success_episodes} failure={audit.failure_episodes} "
        f"unknown={audit.unknown_episodes}"
    )
    print(f"Cameras: {','.join(camera_names)} shape={audit.image_shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenPI intervention Zarr data to full and intervention LeRobot v3 datasets.")
    parser.add_argument("--zarr", "-i", required=True, help="Input PiperX intervention Zarr path")
    parser.add_argument("--output", "-o", default=None, help="Full LeRobot v3 output path")
    parser.add_argument("--repo-id", default=None, help="Full LeRobot repo id")
    parser.add_argument("--towel-type", required=True, choices=sorted(TOWEL_TYPE_IDS))
    parser.add_argument("--source-id", required=True, type=int)
    parser.add_argument("--task", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="piperx_bimanual")
    parser.add_argument("--episodes", type=int, default=None, help="Convert only the first N source episodes")
    parser.add_argument("--cameras", default=",".join(CAMERA_DEFAULTS))
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of videos")
    parser.add_argument("--no-streaming", action="store_true", help="Disable LeRobot streaming video encoding")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--validate-samples", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Audit input without writing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    if not camera_names:
        raise ValueError("--cameras must contain at least one camera")
    audit = audit_source(zarr_path, camera_names, args.episodes)
    _print_audit(zarr_path, audit, camera_names)
    if args.dry_run:
        return
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")
    if args.repo_id is None:
        raise ValueError("--repo-id is required unless --dry-run is used")

    meta_config = _read_meta_config(_open_zarr(zarr_path)["meta"])
    task = args.task or str(meta_config.get("task") or meta_config.get("prompt") or zarr_path.stem)
    convert_dual_views(
        zarr_path=zarr_path,
        output_path=Path(args.output).expanduser().resolve(),
        repo_id=args.repo_id,
        towel_type=args.towel_type,
        source_id=args.source_id,
        task=task,
        fps=args.fps,
        robot_type=args.robot_type,
        camera_names=camera_names,
        max_episodes=args.episodes,
        use_videos=not args.no_videos,
        streaming_encoding=not args.no_streaming,
        image_writer_threads=args.image_writer_threads,
        overwrite=args.overwrite,
        validate_samples=args.validate_samples,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
