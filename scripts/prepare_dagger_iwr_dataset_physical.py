#!/usr/bin/env python3
"""Physically remove unusable DAgger frames and publish LeRobot v2.1 datasets.

Unlike ``prepare_dagger_iwr_dataset.py``, this cleaner does not retain rejected
frames with a zero train mask.  Every output frame is trainable, and each
contiguous, single-controller run is written as an independent LeRobot episode.

The full output is the training dataset.  The optional intervention output is a
reference view projected from the full output by immutable source-frame keys;
it is never appended to the full output because those human frames already
exist there.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import prepare_dagger_iwr_dataset as annotated
import prepare_intervention_dataset as legacy


CONTROL_POLICY = annotated.CONTROL_POLICY
CONTROL_INTERVENTION = annotated.CONTROL_INTERVENTION
UNKNOWN_SUCCESS = annotated.UNKNOWN_SUCCESS
DEFAULT_LEROBOT_HOME = annotated.DEFAULT_LEROBOT_HOME
SCHEMA_VERSION = 1
INTERVENTION_SUFFIX = "_intervention"

REMOVED_MASK_COLUMNS = (
    "piperx.train_mask",
    "piperx.stall_mask",
    "piperx.switch_mask",
)
IWR_FEATURES = {
    "piperx.sample_weight": {
        "dtype": "float32",
        "shape": [1],
        "names": ["sample_weight"],
    },
    "piperx.indicator": {"dtype": "int64", "shape": [1], "names": ["indicator"]},
}


@dataclasses.dataclass(frozen=True)
class PhysicalSegment:
    source_root: Path
    source_episode_index: int
    start: int
    end: int
    control_source: int
    tasks: tuple[str, ...]
    success: int
    source_id: int
    original_episode_index: int
    original_start_frame: int
    original_end_frame: int

    @property
    def length(self) -> int:
        return int(self.end - self.start)


@dataclasses.dataclass(frozen=True)
class PhysicalScan:
    segments: tuple[PhysicalSegment, ...]
    source_infos: tuple[dict[str, Any], ...]
    source_frames: int
    candidate_frames: int
    retained_frames: int
    policy_frames: int
    intervention_frames: int
    context_frames_removed: int
    short_run_frames_removed: int
    stall_frames: int
    switch_frames: int
    source_episodes: int
    success_source_episodes: int
    failure_source_episodes: int
    unknown_source_episodes: int


def _runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    yield from legacy._runs(np.asarray(mask, dtype=bool))


def physical_segment_bounds(
    train_mask: np.ndarray,
    control_source: np.ndarray,
    min_segment_frames: int,
) -> list[tuple[int, int, int]]:
    """Return trainable contiguous runs, split at controller boundaries."""
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    control = np.asarray(control_source, dtype=np.int64).reshape(-1)
    if train.shape != control.shape:
        raise ValueError(f"train/control length mismatch: {train.shape} != {control.shape}")
    if min_segment_frames <= 0:
        raise ValueError("min_segment_frames must be positive")
    if not np.isin(control, [CONTROL_POLICY, CONTROL_INTERVENTION]).all():
        raise ValueError("control_source contains unsupported values")

    result: list[tuple[int, int, int]] = []
    for controller in (CONTROL_POLICY, CONTROL_INTERVENTION):
        for start, end in _runs(train & (control == controller)):
            if end - start >= min_segment_frames:
                result.append((start, end, controller))
    result.sort(key=lambda item: item[0])
    return result


def _segment_from_bounds(
    *,
    root: Path,
    episode_index: int,
    df: pd.DataFrame,
    start: int,
    end: int,
    controller: int,
    tasks: tuple[str, ...],
    success: int,
) -> PhysicalSegment:
    source_ids = annotated._scalar_array(df, "piperx.source_id")
    original_episodes = annotated._scalar_array(df, "piperx.original_episode_index")
    original_frames = annotated._scalar_array(df, "piperx.original_frame_index")
    if np.any(source_ids[start:end] != source_ids[start]):
        raise ValueError(f"source_id changes inside {root} episode {episode_index}")
    if np.any(original_episodes[start:end] != original_episodes[start]):
        raise ValueError(f"original_episode_index changes inside {root} episode {episode_index}")
    selected_original = original_frames[start:end]
    if len(selected_original) > 1 and not np.all(np.diff(selected_original) == 1):
        raise ValueError(f"original frame indices are not contiguous in {root} episode {episode_index}")
    return PhysicalSegment(
        source_root=root,
        source_episode_index=episode_index,
        start=start,
        end=end,
        control_source=controller,
        tasks=tasks,
        success=success,
        source_id=int(source_ids[start]),
        original_episode_index=int(original_episodes[start]),
        original_start_frame=int(original_frames[start]),
        original_end_frame=int(original_frames[end - 1]) + 1,
    )


def scan_full_sources(
    roots: Sequence[Path],
    config: annotated.IwrConfig,
    *,
    min_segment_frames: int,
    max_episodes_per_source: int | None = None,
    drop_failed_episodes: bool = False,
    drop_unknown_episodes: bool = False,
) -> PhysicalScan:
    infos = annotated._validate_source_infos(roots)
    segments: list[PhysicalSegment] = []
    counters = {
        "source_frames": 0,
        "candidate_frames": 0,
        "retained_frames": 0,
        "policy_frames": 0,
        "intervention_frames": 0,
        "context_frames_removed": 0,
        "short_run_frames_removed": 0,
        "stall_frames": 0,
        "switch_frames": 0,
        "source_episodes": 0,
        "success_source_episodes": 0,
        "failure_source_episodes": 0,
        "unknown_source_episodes": 0,
    }

    for root, info in zip(roots, infos, strict=True):
        tasks = legacy._read_tasks_jsonl(root)
        rows = legacy._read_episodes_jsonl(root)
        if max_episodes_per_source is not None:
            rows = rows[:max_episodes_per_source]
        for row in tqdm(rows, desc=f"scan full {root.name}"):
            episode_index = int(row["episode_index"])
            df = legacy._read_lerobot_episode(root, info, episode_index)
            validated = annotated.validate_episode(df, root, episode_index)
            success = annotated._episode_success_from_row(row, int(validated["success"]))
            if success == 0 and drop_failed_episodes:
                continue
            if success == UNKNOWN_SUCCESS and drop_unknown_episodes:
                continue

            masks = annotated.compute_masks(df, validated, config)
            bounds = physical_segment_bounds(masks["train"], masks["control"], min_segment_frames)
            retained = sum(end - start for start, end, _ in bounds)
            candidate = int(np.sum(masks["train"]))
            episode_tasks = legacy._episode_task_names(row, tasks)
            for start, end, controller in bounds:
                segment = _segment_from_bounds(
                    root=root,
                    episode_index=episode_index,
                    df=df,
                    start=start,
                    end=end,
                    controller=controller,
                    tasks=episode_tasks,
                    success=success,
                )
                segments.append(segment)
                if controller == CONTROL_POLICY:
                    counters["policy_frames"] += segment.length
                else:
                    counters["intervention_frames"] += segment.length

            counters["source_frames"] += len(df)
            counters["candidate_frames"] += candidate
            counters["retained_frames"] += retained
            counters["context_frames_removed"] += len(df) - candidate
            counters["short_run_frames_removed"] += candidate - retained
            counters["stall_frames"] += int(np.sum(masks["stall"]))
            counters["switch_frames"] += int(np.sum(masks["switch"]))
            counters["source_episodes"] += 1
            if success == 1:
                counters["success_source_episodes"] += 1
            elif success == 0:
                counters["failure_source_episodes"] += 1
            else:
                counters["unknown_source_episodes"] += 1

    if not segments:
        raise ValueError("No physical segments remain after cleaning")
    return PhysicalScan(segments=tuple(segments), source_infos=infos, **counters)


def _frame_identity(df: pd.DataFrame, index: int) -> tuple[int, int, int]:
    return (
        int(annotated._scalar_array(df, "piperx.source_id")[index]),
        int(annotated._scalar_array(df, "piperx.original_episode_index")[index]),
        int(annotated._scalar_array(df, "piperx.original_frame_index")[index]),
    )


def retained_intervention_actions(scan: PhysicalScan) -> dict[tuple[int, int, int], bytes]:
    retained: dict[tuple[int, int, int], bytes] = {}
    info_cache: dict[Path, dict[str, Any]] = {}
    frame_cache: dict[tuple[Path, int], pd.DataFrame] = {}
    for segment in scan.segments:
        if segment.control_source != CONTROL_INTERVENTION:
            continue
        info = info_cache.setdefault(
            segment.source_root,
            legacy._read_json(segment.source_root / "meta" / "info.json"),
        )
        cache_key = (segment.source_root, segment.source_episode_index)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = legacy._read_lerobot_episode(
                segment.source_root, info, segment.source_episode_index
            )
        df = frame_cache[cache_key]
        executed = annotated._vector_array(df, "piperx.executed_action")
        source_ids, original_episodes, original_frames = annotated._frame_key_arrays(df)
        for index in range(segment.start, segment.end):
            key = (int(source_ids[index]), int(original_episodes[index]), int(original_frames[index]))
            if key in retained:
                raise ValueError(f"Duplicate retained intervention frame {key}")
            retained[key] = executed[index].tobytes()
    return retained


def scan_intervention_projection(
    roots: Sequence[Path],
    retained_actions: dict[tuple[int, int, int], bytes],
    *,
    min_segment_frames: int,
) -> tuple[PhysicalScan, dict[str, int]]:
    infos = annotated._validate_source_infos(roots)
    segments: list[PhysicalSegment] = []
    found: set[tuple[int, int, int]] = set()
    source_frames = 0
    source_episodes = 0
    success_episodes = failure_episodes = unknown_episodes = 0

    for root, info in zip(roots, infos, strict=True):
        tasks = legacy._read_tasks_jsonl(root)
        for row in tqdm(legacy._read_episodes_jsonl(root), desc=f"project intervention {root.name}"):
            episode_index = int(row["episode_index"])
            df = legacy._read_lerobot_episode(root, info, episode_index)
            validated = annotated.validate_episode(df, root, episode_index)
            control = np.asarray(validated["control"], dtype=np.int64)
            if not np.all(control == CONTROL_INTERVENTION):
                raise ValueError(f"Intervention reference contains policy frames: {root} episode {episode_index}")
            success = annotated._episode_success_from_row(row, int(validated["success"]))
            source_ids, original_episodes, original_frames = annotated._frame_key_arrays(df)
            executed = annotated._vector_array(df, "piperx.executed_action")
            selected = np.zeros(len(df), dtype=bool)
            for index in range(len(df)):
                key = (int(source_ids[index]), int(original_episodes[index]), int(original_frames[index]))
                expected = retained_actions.get(key)
                if expected is None:
                    continue
                if expected != executed[index].tobytes():
                    raise ValueError(f"Intervention reference action mismatch at {key}")
                if key in found:
                    raise ValueError(f"Duplicate intervention reference frame {key}")
                found.add(key)
                selected[index] = True

            episode_tasks = legacy._episode_task_names(row, tasks)
            for start, end in _runs(selected):
                if end - start < min_segment_frames:
                    raise ValueError(
                        "A retained full-view intervention run became shorter than the configured minimum "
                        f"in reference episode {episode_index}: [{start}, {end})"
                    )
                segments.append(
                    _segment_from_bounds(
                        root=root,
                        episode_index=episode_index,
                        df=df,
                        start=start,
                        end=end,
                        controller=CONTROL_INTERVENTION,
                        tasks=episode_tasks,
                        success=success,
                    )
                )

            source_frames += len(df)
            source_episodes += 1
            if success == 1:
                success_episodes += 1
            elif success == 0:
                failure_episodes += 1
            else:
                unknown_episodes += 1

    missing = sorted(set(retained_actions) - found)
    if missing:
        raise ValueError(
            f"Intervention projection is missing {len(missing)} retained full-view frames; first={missing[0]}"
        )
    retained_frames = sum(segment.length for segment in segments)
    if retained_frames != len(retained_actions):
        raise ValueError(f"Projection count mismatch: {retained_frames} != {len(retained_actions)}")
    scan = PhysicalScan(
        segments=tuple(segments),
        source_infos=infos,
        source_frames=source_frames,
        candidate_frames=len(retained_actions),
        retained_frames=retained_frames,
        policy_frames=0,
        intervention_frames=retained_frames,
        context_frames_removed=source_frames - retained_frames,
        short_run_frames_removed=0,
        stall_frames=0,
        switch_frames=0,
        source_episodes=source_episodes,
        success_source_episodes=success_episodes,
        failure_source_episodes=failure_episodes,
        unknown_source_episodes=unknown_episodes,
    )
    return scan, {
        "retained_full_intervention_frames": len(retained_actions),
        "projected_reference_frames": retained_frames,
        "missing": 0,
        "action_mismatches": 0,
    }


def resolve_intervention_weight(scan: PhysicalScan, requested: str, policy_weight: float) -> float:
    if requested != "auto":
        value = float(requested)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("--intervention-weight must be positive or 'auto'")
        return value
    if scan.intervention_frames <= 0:
        raise ValueError("Cannot compute an automatic weight without retained intervention frames")
    return float(policy_weight * max(1.0, scan.policy_frames / scan.intervention_frames))


def prepare_output_dataframe(
    source_df: pd.DataFrame,
    segment: PhysicalSegment,
    *,
    output_episode_index: int,
    global_frame_index: int,
    task_index: int,
    fps: float,
    policy_weight: float,
    intervention_weight: float,
) -> pd.DataFrame:
    out = source_df.iloc[segment.start : segment.end].copy()
    out.drop(columns=[key for key in REMOVED_MASK_COLUMNS if key in out.columns], inplace=True)
    n = len(out)
    out["timestamp"] = np.arange(n, dtype=np.float32) / np.float32(fps)
    out["frame_index"] = np.arange(n, dtype=np.int64)
    out["episode_index"] = np.full(n, output_episode_index, dtype=np.int64)
    out["index"] = np.arange(global_frame_index, global_frame_index + n, dtype=np.int64)
    out["task_index"] = np.full(n, task_index, dtype=np.int64)
    weight = intervention_weight if segment.control_source == CONTROL_INTERVENTION else policy_weight
    out["piperx.sample_weight"] = np.full(n, weight, dtype=np.float32)
    out["piperx.indicator"] = np.full(
        n, 1 if segment.control_source == CONTROL_INTERVENTION else 0, dtype=np.int64
    )
    return out


def _augment_info(
    source_info: dict[str, Any],
    *,
    repo_id: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
) -> dict[str, Any]:
    out = dict(source_info)
    features = dict(out.get("features", {}))
    for key in REMOVED_MASK_COLUMNS:
        features.pop(key, None)
    features.update(IWR_FEATURES)
    out.update(
        {
            "repo_id": repo_id,
            "codebase_version": "v2.1",
            "features": features,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_videos,
            "total_chunks": 1,
            "chunks_size": max(1000, total_episodes),
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    )
    return out


def _find_ffmpeg() -> str:
    return legacy._find_ffmpeg()


def _find_ffprobe(ffmpeg: str) -> str | None:
    candidate = Path(ffmpeg).with_name("ffprobe")
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("ffprobe")
    if found:
        return found
    return None


def _count_video_frames(path: Path, ffprobe: str | None) -> int:
    if ffprobe is not None:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(probe.stdout.strip())

    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _extract_exact_clip(
    ffmpeg: str,
    ffprobe: str | None,
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_frame / fps:.9f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.9f}",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        actual = _count_video_frames(temporary, ffprobe)
        if actual != frame_count:
            raise ValueError(f"Video frame count mismatch for {source}: {actual} != {frame_count}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.tmp-{os.getpid()}-{int(time.time())}")


def _backup_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.old-{os.getpid()}-{int(time.time())}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(legacy._jsonify(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_work_dataset(
    scan: PhysicalScan,
    *,
    final_root: Path,
    repo_id: str,
    dataset_view: str,
    policy_weight: float,
    intervention_weight: float,
    cleaning_config: annotated.IwrConfig,
    min_segment_frames: int,
    reference_audit: dict[str, int] | None,
    skip_videos: bool,
    video_workers: int,
) -> Path:
    work_root = _temporary_path(final_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    (work_root / "meta").mkdir(parents=True)
    (work_root / "data" / "chunk-000").mkdir(parents=True)

    video_keys = legacy._video_keys_from_info(scan.source_infos[0])
    task_names: list[str] = []
    for segment in scan.segments:
        for task in segment.tasks:
            if task not in task_names:
                task_names.append(task)
    if not task_names:
        task_names = [""]
    with (work_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for task_index, task in enumerate(task_names):
            file.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")

    output_info = _augment_info(
        scan.source_infos[0],
        repo_id=repo_id,
        total_episodes=len(scan.segments),
        total_frames=scan.retained_frames,
        total_tasks=len(task_names),
        total_videos=0 if skip_videos else len(scan.segments) * len(video_keys),
    )
    feature_keys = list(output_info["features"])
    video_jobs: list[tuple[Path, Path, int, int, float]] = []
    manifest: list[dict[str, Any]] = []
    info_cache: dict[Path, dict[str, Any]] = {}
    frame_cache: dict[tuple[Path, int], pd.DataFrame] = {}
    global_frame = 0

    with (work_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, (
        work_root / "meta" / "episodes_stats.jsonl"
    ).open("w", encoding="utf-8") as stats_file:
        for output_episode_index, segment in enumerate(tqdm(scan.segments, desc=f"write {dataset_view} parquet")):
            source_info = info_cache.setdefault(
                segment.source_root,
                legacy._read_json(segment.source_root / "meta" / "info.json"),
            )
            cache_key = (segment.source_root, segment.source_episode_index)
            if cache_key not in frame_cache:
                frame_cache[cache_key] = legacy._read_lerobot_episode(
                    segment.source_root, source_info, segment.source_episode_index
                )
            source_df = frame_cache[cache_key]
            task = next((item for item in segment.tasks if item), "")
            output_df = prepare_output_dataframe(
                source_df,
                segment,
                output_episode_index=output_episode_index,
                global_frame_index=global_frame,
                task_index=task_names.index(task),
                fps=float(source_info["fps"]),
                policy_weight=policy_weight,
                intervention_weight=intervention_weight,
            )
            parquet_path = work_root / "data" / "chunk-000" / f"episode_{output_episode_index:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(output_df, preserve_index=False), parquet_path)
            episode_file.write(
                json.dumps(
                    {
                        "episode_index": output_episode_index,
                        "tasks": list(segment.tasks),
                        "length": segment.length,
                        "episode_success": segment.success,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats_file.write(
                json.dumps(
                    {
                        "episode_index": output_episode_index,
                        "stats": legacy._episode_stats(output_df, feature_keys),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest.append(
                {
                    "output_episode_index": output_episode_index,
                    "source_root": str(segment.source_root),
                    "source_episode_index": segment.source_episode_index,
                    "source_start": segment.start,
                    "source_end": segment.end,
                    "length": segment.length,
                    "control_source": segment.control_source,
                    "source_id": segment.source_id,
                    "original_episode_index": segment.original_episode_index,
                    "original_start_frame": segment.original_start_frame,
                    "original_end_frame": segment.original_end_frame,
                    "success": segment.success,
                    "sample_weight": (
                        intervention_weight
                        if segment.control_source == CONTROL_INTERVENTION
                        else policy_weight
                    ),
                }
            )
            if not skip_videos:
                for video_key in video_keys:
                    source_video = legacy._find_lerobot_video(
                        segment.source_root,
                        source_info,
                        segment.source_episode_index,
                        video_key,
                    )
                    destination = (
                        work_root
                        / "videos"
                        / "chunk-000"
                        / video_key
                        / f"episode_{output_episode_index:06d}.mp4"
                    )
                    video_jobs.append(
                        (source_video, destination, segment.start, segment.length, float(source_info["fps"]))
                    )
            global_frame += segment.length

    _write_json(work_root / "meta" / "info.json", output_info)
    (work_root / "meta" / "piperx_iwr_physical_segment_manifest.jsonl").write_text(
        "".join(json.dumps(legacy._jsonify(item), ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "physical_dagger_iwr_cleaning",
        "dataset_view": dataset_view,
        "source_roots": [str(path) for path in sorted({segment.source_root for segment in scan.segments})],
        "output_root": str(final_root),
        "repo_id": repo_id,
        "frames_are_physically_removed": True,
        "train_mask_is_written": False,
        "segments_are_split_at_control_source_changes": True,
        "intervention_reference_is_appended": False,
        "action_horizon": cleaning_config.action_horizon,
        "stall_min_frames": cleaning_config.stall_min_frames,
        "stall_state_delta_eps": cleaning_config.stall_state_delta_eps,
        "stall_action_delta_eps": cleaning_config.stall_action_delta_eps,
        "drop_before_switch_frames": cleaning_config.drop_before_switch_frames,
        "drop_after_switch_frames": cleaning_config.drop_after_switch_frames,
        "require_future_clean": cleaning_config.require_future_clean,
        "min_segment_frames": min_segment_frames,
        "policy_weight": policy_weight,
        "intervention_weight": intervention_weight,
        "statistics": {
            field.name: getattr(scan, field.name)
            for field in dataclasses.fields(PhysicalScan)
            if field.name not in {"segments", "source_infos"}
        }
        | {"output_episodes": len(scan.segments)},
        "intervention_reference_audit": reference_audit,
    }
    _write_json(work_root / "meta" / "piperx_iwr_physical_cleaning.json", report)

    if not skip_videos:
        ffmpeg = _find_ffmpeg()
        ffprobe = _find_ffprobe(ffmpeg)
        with ThreadPoolExecutor(max_workers=max(1, video_workers)) as pool:
            futures = [
                pool.submit(
                    _extract_exact_clip,
                    ffmpeg,
                    ffprobe,
                    source,
                    destination,
                    start_frame=start,
                    frame_count=count,
                    fps=fps,
                )
                for source, destination, start, count, fps in video_jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"encode {dataset_view} videos"):
                future.result()
    return work_root


def validate_work_dataset(
    root: Path,
    scan: PhysicalScan,
    *,
    repo_id: str,
    expected_policy_weight: float,
    expected_intervention_weight: float,
    validation_samples: int,
    validate_videos: bool,
) -> dict[tuple[int, int, int], bytes]:
    info = legacy._read_json(root / "meta" / "info.json")
    if int(info["total_episodes"]) != len(scan.segments):
        raise ValueError("Output episode count mismatch")
    if int(info["total_frames"]) != scan.retained_frames:
        raise ValueError("Output frame count mismatch")
    for key in REMOVED_MASK_COLUMNS:
        if key in info.get("features", {}):
            raise ValueError(f"Physical output unexpectedly advertises {key}")

    identities: dict[tuple[int, int, int], bytes] = {}
    total = 0
    for output_episode_index, segment in enumerate(scan.segments):
        df = legacy._read_lerobot_episode(root, info, output_episode_index)
        if len(df) != segment.length:
            raise ValueError(f"Output segment length mismatch at episode {output_episode_index}")
        for key in REMOVED_MASK_COLUMNS:
            if key in df.columns:
                raise ValueError(f"Physical output unexpectedly contains {key}")
        control = annotated._scalar_array(df, "piperx.control_source")
        indicator = annotated._scalar_array(df, "piperx.indicator")
        weights = annotated._scalar_array(df, "piperx.sample_weight", dtype=np.float32)
        if not np.all(control == segment.control_source):
            raise ValueError(f"Control source changes in output episode {output_episode_index}")
        expected_indicator = 1 if segment.control_source == CONTROL_INTERVENTION else 0
        expected_weight = (
            expected_intervention_weight
            if segment.control_source == CONTROL_INTERVENTION
            else expected_policy_weight
        )
        if not np.all(indicator == expected_indicator):
            raise ValueError(f"Bad IWR indicator in output episode {output_episode_index}")
        if not np.all(weights == np.float32(expected_weight)) or np.any(weights <= 0):
            raise ValueError(f"Bad IWR weight in output episode {output_episode_index}")
        source_ids, original_episodes, original_frames = annotated._frame_key_arrays(df)
        executed = annotated._vector_array(df, "piperx.executed_action")
        if len(original_frames) > 1 and not np.all(np.diff(original_frames) == 1):
            raise ValueError(f"Non-contiguous original frames in output episode {output_episode_index}")
        if segment.control_source == CONTROL_INTERVENTION:
            for index in range(len(df)):
                key = (int(source_ids[index]), int(original_episodes[index]), int(original_frames[index]))
                if key in identities:
                    raise ValueError(f"Duplicate output intervention identity {key}")
                identities[key] = executed[index].tobytes()
        total += len(df)
    if total != scan.retained_frames:
        raise ValueError(f"Output row total mismatch: {total} != {scan.retained_frames}")
    if validate_videos and validation_samples > 0:
        legacy._validate_output_dataset(root, repo_id, validation_samples, "pyav")
    return identities


def _publish_all(work_and_final: Sequence[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for work, final in work_and_final:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = _backup_path(final)
                final.rename(backup)
                backups.append((backup, final))
            work.rename(final)
            published.append((final, work))
        for backup, _ in backups:
            shutil.rmtree(backup)
    except Exception:
        for final, work in reversed(published):
            if final.exists() and not work.exists():
                final.rename(work)
        for backup, final in reversed(backups):
            if backup.exists() and not final.exists():
                backup.rename(final)
        raise


def default_intervention_repo_id(repo_id: str) -> str:
    return f"{repo_id}{INTERVENTION_SUFFIX}"


def default_intervention_output_root(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}{INTERVENTION_SUFFIX}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically remove unusable DAgger frames and publish full/intervention LeRobot v2.1 views."
    )
    parser.add_argument("--src-roots", nargs="+", required=True, help="Full DAgger LeRobot v2.1 roots")
    parser.add_argument("--intervention-reference-roots", nargs="*", default=())
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--intervention-repo-id", default=None)
    parser.add_argument("--intervention-root", default=None)
    parser.add_argument("--no-intervention-output", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=60)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--intervention-weight", default="auto")
    parser.add_argument("--drop-before-switch-frames", type=int, default=0)
    parser.add_argument("--drop-after-switch-frames", type=int, default=0)
    parser.add_argument("--stall-min-frames", type=int, default=60)
    parser.add_argument("--stall-state-delta-eps", type=float, default=1e-4)
    parser.add_argument("--stall-action-delta-eps", type=float, default=1e-4)
    parser.add_argument("--require-future-clean", action="store_true")
    parser.add_argument("--min-segment-frames", type=int, default=1)
    parser.add_argument("--drop-failed-episodes", action="store_true")
    parser.add_argument("--drop-unknown-episodes", action="store_true")
    parser.add_argument("--max-episodes-per-source", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--video-workers", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = tuple(Path(value).expanduser().resolve() for value in args.src_roots)
    references = tuple(Path(value).expanduser().resolve() for value in args.intervention_reference_roots)
    for root in roots + references:
        if not root.exists():
            raise FileNotFoundError(root)
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.min_segment_frames <= 0:
        raise ValueError("--min-segment-frames must be positive")
    if not np.isfinite(args.policy_weight) or args.policy_weight <= 0:
        raise ValueError("--policy-weight must be positive")

    config = annotated.IwrConfig(
        action_horizon=args.action_horizon,
        policy_weight=args.policy_weight,
        intervention_weight=1.0,
        drop_before_switch_frames=args.drop_before_switch_frames,
        drop_after_switch_frames=args.drop_after_switch_frames,
        stall_min_frames=args.stall_min_frames,
        stall_state_delta_eps=args.stall_state_delta_eps,
        stall_action_delta_eps=args.stall_action_delta_eps,
        require_future_clean=args.require_future_clean,
    )
    reference_audit = annotated.audit_intervention_reference(roots, references) if references else None
    full_scan = scan_full_sources(
        roots,
        config,
        min_segment_frames=args.min_segment_frames,
        max_episodes_per_source=args.max_episodes_per_source,
        drop_failed_episodes=args.drop_failed_episodes,
        drop_unknown_episodes=args.drop_unknown_episodes,
    )
    intervention_weight = resolve_intervention_weight(full_scan, args.intervention_weight, args.policy_weight)
    retained_actions = retained_intervention_actions(full_scan)
    intervention_scan: PhysicalScan | None = None
    projection_audit: dict[str, int] | None = None
    if references and not args.no_intervention_output:
        intervention_scan, projection_audit = scan_intervention_projection(
            references,
            retained_actions,
            min_segment_frames=args.min_segment_frames,
        )

    print(
        f"Physical full output: source_episodes={full_scan.source_episodes} "
        f"segments={len(full_scan.segments)} source_frames={full_scan.source_frames} "
        f"retained={full_scan.retained_frames} removed={full_scan.source_frames - full_scan.retained_frames}"
    )
    print(
        f"Retained train frames: policy={full_scan.policy_frames} "
        f"intervention={full_scan.intervention_frames}; "
        f"IWR weights={args.policy_weight:.6f}/{intervention_weight:.6f}"
    )
    if intervention_scan is not None:
        print(
            f"Physical intervention output: segments={len(intervention_scan.segments)} "
            f"frames={intervention_scan.retained_frames}"
        )
    if args.dry_run:
        return

    output_root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else Path(os.environ.get("HF_LEROBOT_HOME", DEFAULT_LEROBOT_HOME)) / args.repo_id
    )
    intervention_repo_id = args.intervention_repo_id or default_intervention_repo_id(args.repo_id)
    intervention_root = (
        Path(args.intervention_root).expanduser().resolve()
        if args.intervention_root
        else default_intervention_output_root(output_root)
    )
    planned = [output_root]
    if intervention_scan is not None:
        planned.append(intervention_root)
    if len(set(planned)) != len(planned):
        raise ValueError("Full and intervention output roots must differ")
    if not args.overwrite:
        existing = [str(path) for path in planned if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists: {existing}. Pass --overwrite to replace it.")

    work_items: list[tuple[Path, Path]] = []
    try:
        full_work = write_work_dataset(
            full_scan,
            final_root=output_root,
            repo_id=args.repo_id,
            dataset_view="full",
            policy_weight=args.policy_weight,
            intervention_weight=intervention_weight,
            cleaning_config=config,
            min_segment_frames=args.min_segment_frames,
            reference_audit=(reference_audit or {}) | (projection_audit or {}),
            skip_videos=args.skip_videos,
            video_workers=args.video_workers,
        )
        work_items.append((full_work, output_root))
        full_identities = validate_work_dataset(
            full_work,
            full_scan,
            repo_id=args.repo_id,
            expected_policy_weight=args.policy_weight,
            expected_intervention_weight=intervention_weight,
            validation_samples=0 if args.no_validate else args.validation_samples,
            validate_videos=not args.skip_videos,
        )
        if full_identities != retained_actions:
            raise ValueError("Written full output intervention frames differ from the scan")

        if intervention_scan is not None:
            intervention_work = write_work_dataset(
                intervention_scan,
                final_root=intervention_root,
                repo_id=intervention_repo_id,
                dataset_view="intervention_segments",
                policy_weight=args.policy_weight,
                intervention_weight=intervention_weight,
                cleaning_config=config,
                min_segment_frames=args.min_segment_frames,
                reference_audit=(reference_audit or {}) | (projection_audit or {}),
                skip_videos=args.skip_videos,
                video_workers=args.video_workers,
            )
            work_items.append((intervention_work, intervention_root))
            intervention_identities = validate_work_dataset(
                intervention_work,
                intervention_scan,
                repo_id=intervention_repo_id,
                expected_policy_weight=args.policy_weight,
                expected_intervention_weight=intervention_weight,
                validation_samples=0 if args.no_validate else args.validation_samples,
                validate_videos=not args.skip_videos,
            )
            if intervention_identities != full_identities:
                raise ValueError("Physical full/intervention outputs are not exact frame projections")

        _publish_all(work_items)
    except Exception:
        print("[ERROR] Physical cleaning failed; existing published outputs were not replaced.", file=sys.stderr)
        for work, _ in work_items:
            print(f"Incomplete work directory: {work}", file=sys.stderr)
        raise

    print(f"Published full dataset: {output_root}")
    if intervention_scan is not None:
        print(f"Published intervention dataset: {intervention_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
