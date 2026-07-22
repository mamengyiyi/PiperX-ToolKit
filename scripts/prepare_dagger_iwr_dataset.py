#!/usr/bin/env python3
"""Annotate DAgger rollouts for IWR training without duplicating frames.

The input is the full LeRobot v2.1 view produced from intervention rollouts. The
output keeps each complete source episode, including frames that are context-only
for an action chunk. Training eligibility and IWR weights are stored per frame.

An optional intervention-only LeRobot view is audited against the full view and
can also be written as a cleaned segment-level reference dataset. Its frames are
not appended to the full output, because they already exist there.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import prepare_intervention_dataset as legacy


CONTROL_POLICY = 0
CONTROL_INTERVENTION = 1
UNKNOWN_SUCCESS = 255
SCHEMA_VERSION = 1
DEFAULT_LEROBOT_HOME = Path("/root/data/my/piperx/lerobot_datasets")
INTERVENTION_SUFFIX = "_intervention"

REQUIRED_VECTOR_COLUMNS = (
    "observation.state",
    "action",
    "piperx.policy_action",
    "piperx.human_action",
    "piperx.executed_action",
)
REQUIRED_SCALAR_COLUMNS = (
    "piperx.control_source",
    "piperx.intervention_mask",
    "piperx.episode_success",
    "piperx.policy_action_valid",
    "piperx.human_action_valid",
    "piperx.original_episode_index",
    "piperx.original_frame_index",
    "piperx.source_id",
    "piperx.towel_type_id",
)
ANNOTATION_FEATURES = {
    "piperx.train_mask": {"dtype": "bool", "shape": [1], "names": ["train_mask"]},
    "piperx.stall_mask": {"dtype": "bool", "shape": [1], "names": ["stall_mask"]},
    "piperx.switch_mask": {"dtype": "bool", "shape": [1], "names": ["switch_mask"]},
    "piperx.sample_weight": {
        "dtype": "float32",
        "shape": [1],
        "names": ["sample_weight"],
    },
    "piperx.indicator": {"dtype": "int64", "shape": [1], "names": ["indicator"]},
}


@dataclasses.dataclass(frozen=True)
class IwrConfig:
    action_horizon: int = 60
    policy_weight: float = 1.0
    intervention_weight: float = 1.0
    drop_before_switch_frames: int = 0
    drop_after_switch_frames: int = 0
    stall_min_frames: int = 0
    stall_state_delta_eps: float = 1e-4
    stall_action_delta_eps: float = 1e-4
    require_future_clean: bool = False


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    source_root: Path
    source_episode_index: int
    output_episode_index: int
    length: int
    tasks: tuple[str, ...]
    success: int
    policy_frames: int
    intervention_frames: int
    train_policy_frames: int
    train_intervention_frames: int
    context_only_frames: int
    stall_frames: int
    switch_frames: int


@dataclasses.dataclass(frozen=True)
class ScanResult:
    plans: tuple[EpisodePlan, ...]
    source_infos: tuple[dict[str, Any], ...]
    total_frames: int
    policy_frames: int
    intervention_frames: int
    train_policy_frames: int
    train_intervention_frames: int
    context_only_frames: int
    success_episodes: int
    failure_episodes: int
    unknown_episodes: int


def _scalar_array(df: pd.DataFrame, key: str, dtype: Any = np.int64) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Missing required column: {key}")
    values = df[key].to_numpy()
    if len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
        values = np.asarray([np.asarray(value).reshape(-1)[0] for value in values])
    return np.asarray(values, dtype=dtype).reshape(-1)


def _vector_array(df: pd.DataFrame, key: str, dim: int = 14) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Missing required column: {key}")
    values = legacy._as_matrix(df[key], dim=dim)
    if values.shape != (len(df), dim):
        raise ValueError(f"{key} has shape {values.shape}; expected {(len(df), dim)}")
    return values.astype(np.float32, copy=False)


def _constant_scalar(values: np.ndarray, key: str, episode_index: int) -> int:
    if values.size == 0 or np.any(values != values[0]):
        raise ValueError(f"{key} is not constant in source episode {episode_index}")
    return int(values[0])


def validate_episode(df: pd.DataFrame, source_root: Path, episode_index: int) -> dict[str, np.ndarray | int]:
    if df.empty:
        raise ValueError(f"Empty source episode {episode_index} in {source_root}")
    for key in REQUIRED_VECTOR_COLUMNS:
        if key not in df.columns:
            raise KeyError(f"Missing required column {key} in {source_root} episode {episode_index}")
    for key in REQUIRED_SCALAR_COLUMNS:
        if key not in df.columns:
            raise KeyError(f"Missing required column {key} in {source_root} episode {episode_index}")

    state = _vector_array(df, "observation.state")
    action = _vector_array(df, "action")
    policy = _vector_array(df, "piperx.policy_action")
    human = _vector_array(df, "piperx.human_action")
    executed = _vector_array(df, "piperx.executed_action")
    for key, values in (("observation.state", state), ("action", action), ("piperx.executed_action", executed)):
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {key} in {source_root} episode {episode_index}")
    if not np.array_equal(action, executed):
        max_delta = float(np.max(np.abs(action - executed)))
        raise ValueError(
            f"action differs from piperx.executed_action in {source_root} episode {episode_index}; "
            f"max delta={max_delta}"
        )

    control = _scalar_array(df, "piperx.control_source")
    intervention_mask = _scalar_array(df, "piperx.intervention_mask")
    policy_valid = _scalar_array(df, "piperx.policy_action_valid")
    human_valid = _scalar_array(df, "piperx.human_action_valid")
    if not np.isin(control, [CONTROL_POLICY, CONTROL_INTERVENTION]).all():
        raise ValueError(
            f"Invalid piperx.control_source in {source_root} episode {episode_index}: "
            f"{sorted(set(int(value) for value in np.unique(control)))}"
        )
    if not np.array_equal(intervention_mask.astype(bool), control == CONTROL_INTERVENTION):
        raise ValueError(f"control_source/intervention_mask mismatch in {source_root} episode {episode_index}")
    if np.any(policy_valid[control == CONTROL_POLICY] != 1):
        raise ValueError(f"Policy frame lacks valid policy action in {source_root} episode {episode_index}")
    if np.any(human_valid[control == CONTROL_INTERVENTION] != 1):
        raise ValueError(f"Intervention frame lacks valid human action in {source_root} episode {episode_index}")
    if np.any(policy_valid[control == CONTROL_INTERVENTION] == 0) and not np.all(
        policy[control == CONTROL_INTERVENTION][policy_valid[control == CONTROL_INTERVENTION] == 0] == 0
    ):
        raise ValueError(f"Invalid policy action is not zero-filled in {source_root} episode {episode_index}")
    if np.any(human_valid[control == CONTROL_POLICY] == 0) and not np.all(
        human[control == CONTROL_POLICY][human_valid[control == CONTROL_POLICY] == 0] == 0
    ):
        raise ValueError(f"Invalid human action is not zero-filled in {source_root} episode {episode_index}")
    if np.any(control == CONTROL_INTERVENTION) and not np.array_equal(
        human[control == CONTROL_INTERVENTION], executed[control == CONTROL_INTERVENTION]
    ):
        raise ValueError(f"Human/executed action mismatch in {source_root} episode {episode_index}")

    success_values = _scalar_array(df, "piperx.episode_success")
    success = _constant_scalar(success_values, "piperx.episode_success", episode_index)
    if success not in (0, 1, UNKNOWN_SUCCESS):
        raise ValueError(f"Invalid episode_success={success} in {source_root} episode {episode_index}")
    original_episode = _scalar_array(df, "piperx.original_episode_index")
    _constant_scalar(original_episode, "piperx.original_episode_index", episode_index)
    original_frame = _scalar_array(df, "piperx.original_frame_index")
    if len(original_frame) > 1 and not np.all(np.diff(original_frame) == 1):
        raise ValueError(f"Non-contiguous original frame indices in {source_root} episode {episode_index}")
    _constant_scalar(_scalar_array(df, "piperx.source_id"), "piperx.source_id", episode_index)
    _constant_scalar(_scalar_array(df, "piperx.towel_type_id"), "piperx.towel_type_id", episode_index)

    return {
        "state": state,
        "action": action,
        "control": control,
        "success": success,
    }


def _mark_long_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 0:
        return np.zeros_like(mask, dtype=bool)
    return legacy._mark_long_runs(mask, minimum)


def compute_masks(df: pd.DataFrame, validated: dict[str, np.ndarray | int], config: IwrConfig) -> dict[str, np.ndarray]:
    n = len(df)
    control = np.asarray(validated["control"], dtype=np.int64)
    state = np.asarray(validated["state"], dtype=np.float32)
    action = np.asarray(validated["action"], dtype=np.float32)

    stall = np.zeros(n, dtype=bool)
    if config.stall_min_frames > 0 and n > 1:
        state_delta = np.full(n, np.inf, dtype=np.float32)
        action_delta = np.full(n, np.inf, dtype=np.float32)
        state_delta[1:] = np.linalg.norm(np.diff(state, axis=0), axis=1)
        action_delta[1:] = np.linalg.norm(np.diff(action, axis=0), axis=1)
        stationary = (state_delta <= config.stall_state_delta_eps) & (
            action_delta <= config.stall_action_delta_eps
        )
        stall = _mark_long_runs(stationary, config.stall_min_frames)

    switch = np.zeros(n, dtype=bool)
    if config.drop_before_switch_frames > 0 or config.drop_after_switch_frames > 0:
        switch_events = np.zeros(n, dtype=bool)
        switch_events[1:] = control[1:] != control[:-1]
        switch = legacy._dilate_true(
            switch_events,
            config.drop_before_switch_frames,
            config.drop_after_switch_frames,
        )

    train = ~(stall | switch)
    if config.require_future_clean:
        train &= legacy._future_clean_mask(train, config.action_horizon)
    if config.action_horizon > 1:
        tail = min(n, config.action_horizon - 1)
        train[n - tail :] = False
    return {
        "control": control,
        "train": train,
        "stall": stall,
        "switch": switch,
    }


def _episode_success_from_row(row: dict[str, Any], fallback: int) -> int:
    for key in ("episode_success", "success"):
        if key in row:
            return int(row[key])
    return fallback


def _validate_source_infos(roots: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    if not roots:
        raise ValueError("At least one --src-root is required")
    infos = tuple(legacy._read_json(root / "meta" / "info.json") for root in roots)
    reference = infos[0]
    for root, info in zip(roots, infos, strict=True):
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"Expected LeRobot v2.1 input at {root}")
        if info.get("fps") != reference.get("fps"):
            raise ValueError(f"FPS mismatch for {root}")
        if info.get("robot_type") != reference.get("robot_type"):
            raise ValueError(f"robot_type mismatch for {root}")
        for key in REQUIRED_VECTOR_COLUMNS + REQUIRED_SCALAR_COLUMNS:
            if key not in info.get("features", {}):
                raise KeyError(f"Missing feature {key} in {root / 'meta/info.json'}")
    return infos


def scan_sources(
    roots: Sequence[Path],
    config: IwrConfig,
    *,
    max_episodes_per_source: int | None = None,
    drop_failed_episodes: bool = False,
    drop_unknown_episodes: bool = False,
) -> ScanResult:
    infos = _validate_source_infos(roots)
    plans: list[EpisodePlan] = []
    counters = {
        "total_frames": 0,
        "policy_frames": 0,
        "intervention_frames": 0,
        "train_policy_frames": 0,
        "train_intervention_frames": 0,
        "context_only_frames": 0,
        "success_episodes": 0,
        "failure_episodes": 0,
        "unknown_episodes": 0,
    }
    output_episode_index = 0

    for root, info in zip(roots, infos, strict=True):
        tasks = legacy._read_tasks_jsonl(root)
        episode_rows = legacy._read_episodes_jsonl(root)
        if max_episodes_per_source is not None:
            episode_rows = episode_rows[:max_episodes_per_source]
        for row in tqdm(episode_rows, desc=f"scan {root.name}"):
            source_episode_index = int(row["episode_index"])
            df = legacy._read_lerobot_episode(root, info, source_episode_index)
            validated = validate_episode(df, root, source_episode_index)
            masks = compute_masks(df, validated, config)
            success = _episode_success_from_row(row, int(validated["success"]))
            if success == 0 and drop_failed_episodes:
                continue
            if success == UNKNOWN_SUCCESS and drop_unknown_episodes:
                continue

            control = masks["control"]
            train = masks["train"]
            policy_frames = int(np.sum(control == CONTROL_POLICY))
            intervention_frames = int(np.sum(control == CONTROL_INTERVENTION))
            train_policy = int(np.sum(train & (control == CONTROL_POLICY)))
            train_intervention = int(np.sum(train & (control == CONTROL_INTERVENTION)))
            context_only = int(np.sum(~train))
            tasks_for_episode = legacy._episode_task_names(row, tasks)
            plans.append(
                EpisodePlan(
                    source_root=root,
                    source_episode_index=source_episode_index,
                    output_episode_index=output_episode_index,
                    length=len(df),
                    tasks=tasks_for_episode,
                    success=success,
                    policy_frames=policy_frames,
                    intervention_frames=intervention_frames,
                    train_policy_frames=train_policy,
                    train_intervention_frames=train_intervention,
                    context_only_frames=context_only,
                    stall_frames=int(masks["stall"].sum()),
                    switch_frames=int(masks["switch"].sum()),
                )
            )
            output_episode_index += 1
            counters["total_frames"] += len(df)
            counters["policy_frames"] += policy_frames
            counters["intervention_frames"] += intervention_frames
            counters["train_policy_frames"] += train_policy
            counters["train_intervention_frames"] += train_intervention
            counters["context_only_frames"] += context_only
            if success == 1:
                counters["success_episodes"] += 1
            elif success == 0:
                counters["failure_episodes"] += 1
            else:
                counters["unknown_episodes"] += 1

    if not plans:
        raise ValueError("No episodes remain after filtering")
    return ScanResult(plans=tuple(plans), source_infos=infos, **counters)


def resolve_intervention_weight(scan: ScanResult, requested: str, policy_weight: float) -> float:
    if requested != "auto":
        weight = float(requested)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("--intervention-weight must be positive or 'auto'")
        return weight
    if scan.train_intervention_frames <= 0:
        raise ValueError("Cannot compute automatic intervention weight without trainable intervention frames")
    ratio = scan.train_policy_frames / scan.train_intervention_frames
    return float(policy_weight * max(1.0, ratio))


def default_intervention_repo_id(repo_id: str) -> str:
    return f"{repo_id}{INTERVENTION_SUFFIX}"


def default_intervention_output_root(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}{INTERVENTION_SUFFIX}")


def annotate_episode(df: pd.DataFrame, masks: dict[str, np.ndarray], config: IwrConfig) -> pd.DataFrame:
    out = df.copy()
    control = masks["control"]
    train = masks["train"]
    weights = np.where(
        control == CONTROL_INTERVENTION,
        np.float32(config.intervention_weight),
        np.float32(config.policy_weight),
    ).astype(np.float32)
    weights[~train] = np.float32(0.0)
    out["piperx.train_mask"] = train.astype(bool)
    out["piperx.stall_mask"] = masks["stall"].astype(bool)
    out["piperx.switch_mask"] = masks["switch"].astype(bool)
    out["piperx.sample_weight"] = weights
    out["piperx.indicator"] = (control == CONTROL_INTERVENTION).astype(np.int64)
    return out


def _frame_key_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _scalar_array(df, "piperx.source_id"),
        _scalar_array(df, "piperx.original_episode_index"),
        _scalar_array(df, "piperx.original_frame_index"),
    )


def _collect_intervention_frames(roots: Sequence[Path], require_all_intervention: bool) -> dict[tuple[int, int, int], bytes]:
    frames: dict[tuple[int, int, int], bytes] = {}
    infos = _validate_source_infos(roots)
    for root, info in zip(roots, infos, strict=True):
        for row in tqdm(legacy._read_episodes_jsonl(root), desc=f"reference {root.name}"):
            episode_index = int(row["episode_index"])
            df = legacy._read_lerobot_episode(root, info, episode_index)
            validated = validate_episode(df, root, episode_index)
            control = np.asarray(validated["control"], dtype=np.int64)
            selected = control == CONTROL_INTERVENTION
            if require_all_intervention and not selected.all():
                raise ValueError(f"Intervention reference contains policy frames: {root} episode {episode_index}")
            source_ids, original_episodes, original_frames = _frame_key_arrays(df)
            executed = _vector_array(df, "piperx.executed_action")
            for index in np.flatnonzero(selected):
                key = (int(source_ids[index]), int(original_episodes[index]), int(original_frames[index]))
                if key in frames:
                    raise ValueError(f"Duplicate intervention frame key {key} in {root}")
                frames[key] = executed[index].tobytes()
    return frames


def audit_intervention_reference(full_roots: Sequence[Path], reference_roots: Sequence[Path]) -> dict[str, int]:
    full = _collect_intervention_frames(full_roots, require_all_intervention=False)
    reference = _collect_intervention_frames(reference_roots, require_all_intervention=True)
    missing = sorted(set(full) - set(reference))
    extra = sorted(set(reference) - set(full))
    if missing or extra:
        raise ValueError(
            f"Intervention reference key mismatch: missing={len(missing)} extra={len(extra)}; "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    mismatched = [key for key in full if full[key] != reference[key]]
    if mismatched:
        raise ValueError(f"Intervention reference action mismatch at {mismatched[0]}")
    return {"full_intervention_frames": len(full), "reference_frames": len(reference), "mismatches": 0}


def _augment_info(info: dict[str, Any], repo_id: str, episodes: int, frames: int, tasks: int, videos: int) -> dict[str, Any]:
    out = dict(info)
    features = dict(info["features"])
    features.update(ANNOTATION_FEATURES)
    out.update(
        {
            "repo_id": repo_id,
            "codebase_version": "v2.1",
            "features": features,
            "total_episodes": episodes,
            "total_frames": frames,
            "total_tasks": tasks,
            "total_videos": videos,
            "total_chunks": 1,
            "chunks_size": max(1000, episodes),
            "splits": {"train": f"0:{episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    )
    return out


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _temporary_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.tmp-{os.getpid()}-{int(time.time())}")


def _backup_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.old-{os.getpid()}-{int(time.time())}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(legacy._jsonify(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_output(root: Path, repo_id: str, scan: ScanResult, config: IwrConfig, samples: int) -> None:
    info = legacy._read_json(root / "meta" / "info.json")
    if (int(info["total_episodes"]), int(info["total_frames"])) != (len(scan.plans), scan.total_frames):
        raise ValueError("Output episode/frame count mismatch")
    for key in ANNOTATION_FEATURES:
        if key not in info["features"]:
            raise ValueError(f"Output is missing feature {key}")

    train_policy = 0
    train_intervention = 0
    context = 0
    for plan in scan.plans:
        df = legacy._read_lerobot_episode(root, info, plan.output_episode_index)
        if len(df) != plan.length:
            raise ValueError(f"Episode length changed for output episode {plan.output_episode_index}")
        train = _scalar_array(df, "piperx.train_mask", dtype=bool)
        control = _scalar_array(df, "piperx.control_source")
        weights = _scalar_array(df, "piperx.sample_weight", dtype=np.float32)
        indicator = _scalar_array(df, "piperx.indicator")
        if not np.array_equal(indicator, (control == CONTROL_INTERVENTION).astype(np.int64)):
            raise ValueError(f"Bad IWR indicator in output episode {plan.output_episode_index}")
        expected = np.where(
            control == CONTROL_INTERVENTION,
            np.float32(config.intervention_weight),
            np.float32(config.policy_weight),
        )
        expected[~train] = 0
        if not np.array_equal(weights, expected.astype(np.float32)):
            raise ValueError(f"Bad sample weights in output episode {plan.output_episode_index}")
        train_policy += int(np.sum(train & (control == CONTROL_POLICY)))
        train_intervention += int(np.sum(train & (control == CONTROL_INTERVENTION)))
        context += int(np.sum(~train))
    if (train_policy, train_intervention, context) != (
        scan.train_policy_frames,
        scan.train_intervention_frames,
        scan.context_only_frames,
    ):
        raise ValueError("Output annotation totals do not match scan")
    legacy._validate_output_dataset(root, repo_id, samples, "pyav")


def write_dataset(
    scan: ScanResult,
    config: IwrConfig,
    *,
    output_root: Path,
    repo_id: str,
    reference_audit: dict[str, int] | None,
    dataset_view: str,
    overwrite: bool,
    skip_videos: bool,
    video_workers: int,
    validate_samples: int,
) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_root}. Pass --overwrite to replace it.")
    work_root = _temporary_path(output_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    (work_root / "meta").mkdir(parents=True)
    (work_root / "data" / "chunk-000").mkdir(parents=True)

    video_keys = legacy._video_keys_from_info(scan.source_infos[0])
    task_names: list[str] = []
    for plan in scan.plans:
        for task in plan.tasks:
            if task not in task_names:
                task_names.append(task)
    if not task_names:
        task_names = [""]
    with (work_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for task_index, task in enumerate(task_names):
            file.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")

    global_frame = 0
    video_jobs: list[tuple[Path, Path]] = []
    manifest: list[dict[str, Any]] = []
    output_info = _augment_info(
        scan.source_infos[0],
        repo_id,
        len(scan.plans),
        scan.total_frames,
        len(task_names),
        0 if skip_videos else len(scan.plans) * len(video_keys),
    )
    feature_keys = list(output_info["features"])

    with (work_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, (
        work_root / "meta" / "episodes_stats.jsonl"
    ).open("w", encoding="utf-8") as stats_file:
        for plan in tqdm(scan.plans, desc="write IWR episodes"):
            source_info = legacy._read_json(plan.source_root / "meta" / "info.json")
            source_df = legacy._read_lerobot_episode(
                plan.source_root, source_info, plan.source_episode_index
            )
            validated = validate_episode(source_df, plan.source_root, plan.source_episode_index)
            masks = compute_masks(source_df, validated, config)
            out = annotate_episode(source_df, masks, config)
            out["timestamp"] = np.arange(len(out), dtype=np.float32) / np.float32(source_info["fps"])
            out["frame_index"] = np.arange(len(out), dtype=np.int64)
            out["episode_index"] = np.full(len(out), plan.output_episode_index, dtype=np.int64)
            out["index"] = np.arange(global_frame, global_frame + len(out), dtype=np.int64)
            task = next((value for value in plan.tasks if value), "")
            out["task_index"] = np.full(len(out), task_names.index(task), dtype=np.int64)
            pq.write_table(
                pa.Table.from_pandas(out, preserve_index=False),
                work_root / "data" / "chunk-000" / f"episode_{plan.output_episode_index:06d}.parquet",
            )
            episode_file.write(
                json.dumps(
                    {
                        "episode_index": plan.output_episode_index,
                        "tasks": list(plan.tasks),
                        "length": len(out),
                        "episode_success": plan.success,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats_file.write(
                json.dumps(
                    {
                        "episode_index": plan.output_episode_index,
                        "stats": legacy._episode_stats(out, feature_keys),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest.append(dataclasses.asdict(plan) | {"source_root": str(plan.source_root)})
            if not skip_videos:
                for video_key in video_keys:
                    source_video = legacy._find_lerobot_video(
                        plan.source_root,
                        source_info,
                        plan.source_episode_index,
                        video_key,
                    )
                    destination = (
                        work_root
                        / "videos"
                        / "chunk-000"
                        / video_key
                        / f"episode_{plan.output_episode_index:06d}.mp4"
                    )
                    video_jobs.append((source_video, destination))
            global_frame += len(out)

    _write_json(work_root / "meta" / "info.json", output_info)
    (work_root / "meta" / "piperx_iwr_episode_manifest.jsonl").write_text(
        "".join(json.dumps(legacy._jsonify(item), ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "complete_rollout_iwr",
        "dataset_view": dataset_view,
        "source_roots": [str(path) for path in sorted({plan.source_root for plan in scan.plans})],
        "output_root": str(output_root),
        "repo_id": repo_id,
        "action_horizon": config.action_horizon,
        "policy_weight": config.policy_weight,
        "intervention_weight": config.intervention_weight,
        "frames_are_physically_removed": False,
        "context_only_frames_have_zero_weight": True,
        "intervention_reference_is_appended": False,
        "statistics": {
            key: getattr(scan, key)
            for key in (
                "total_frames",
                "policy_frames",
                "intervention_frames",
                "train_policy_frames",
                "train_intervention_frames",
                "context_only_frames",
                "success_episodes",
                "failure_episodes",
                "unknown_episodes",
            )
        }
        | {"episodes": len(scan.plans)},
        "intervention_reference_audit": reference_audit,
    }
    _write_json(work_root / "meta" / "piperx_iwr_cleaning.json", report)

    if not skip_videos:
        with ThreadPoolExecutor(max_workers=max(1, video_workers)) as executor:
            futures = [executor.submit(_link_or_copy, source, destination) for source, destination in video_jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="link/copy videos"):
                future.result()

    backup: Path | None = None
    try:
        _validate_output(work_root, repo_id, scan, config, validate_samples)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            backup = _backup_path(output_root)
            output_root.rename(backup)
        work_root.rename(output_root)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not output_root.exists():
            backup.rename(output_root)
        raise
    print(f"Published: {output_root}")
    print(f"episodes={len(scan.plans)} frames={scan.total_frames} videos={0 if skip_videos else len(video_jobs)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate complete DAgger rollouts for IWR training without duplicating intervention frames."
    )
    parser.add_argument("--src-roots", nargs="+", required=True, help="Full LeRobot v2.1 rollout roots")
    parser.add_argument(
        "--intervention-reference-roots",
        nargs="*",
        default=(),
        help="Optional intervention-only v2.1 roots used for integrity checks and cleaned segment output",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--intervention-repo-id",
        default=None,
        help="Cleaned intervention-only repo id; defaults to ${repo-id}_intervention",
    )
    parser.add_argument(
        "--intervention-root",
        default=None,
        help="Explicit cleaned intervention-only output root; defaults to ${root}_intervention or ${repo-id}_intervention",
    )
    parser.add_argument(
        "--no-intervention-output",
        action="store_true",
        help="Only audit intervention reference roots; do not write a cleaned intervention-only dataset",
    )
    parser.add_argument("--action-horizon", type=int, default=60)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument(
        "--intervention-weight",
        default="auto",
        help="Positive float or auto; auto balances trainable policy/intervention frame counts",
    )
    parser.add_argument("--drop-before-switch-frames", type=int, default=0)
    parser.add_argument("--drop-after-switch-frames", type=int, default=0)
    parser.add_argument("--stall-min-frames", type=int, default=0, help="0 disables stall masking")
    parser.add_argument("--stall-state-delta-eps", type=float, default=1e-4)
    parser.add_argument("--stall-action-delta-eps", type=float, default=1e-4)
    parser.add_argument("--require-future-clean", action="store_true")
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
    args = _parse_args()
    roots = tuple(Path(value).expanduser().resolve() for value in args.src_roots)
    references = tuple(Path(value).expanduser().resolve() for value in args.intervention_reference_roots)
    for path in roots + references:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if not np.isfinite(args.policy_weight) or args.policy_weight <= 0:
        raise ValueError("--policy-weight must be positive")
    if args.drop_before_switch_frames < 0 or args.drop_after_switch_frames < 0:
        raise ValueError("Switch window sizes must be non-negative")

    scan_config = IwrConfig(
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
    scan = scan_sources(
        roots,
        scan_config,
        max_episodes_per_source=args.max_episodes_per_source,
        drop_failed_episodes=args.drop_failed_episodes,
        drop_unknown_episodes=args.drop_unknown_episodes,
    )
    intervention_weight = resolve_intervention_weight(scan, args.intervention_weight, args.policy_weight)
    config = dataclasses.replace(scan_config, intervention_weight=intervention_weight)
    reference_audit = audit_intervention_reference(roots, references) if references else None

    print(f"Episodes: {len(scan.plans)}")
    print(f"Full frames retained: {scan.total_frames}")
    print(
        f"Trainable frames: policy={scan.train_policy_frames} "
        f"intervention={scan.train_intervention_frames} context_only={scan.context_only_frames}"
    )
    print(f"IWR weights: policy={config.policy_weight:.6f} intervention={config.intervention_weight:.6f}")
    if reference_audit is not None:
        print(f"Intervention reference audit: {json.dumps(reference_audit, sort_keys=True)}")
    if args.dry_run:
        return

    output_root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else Path(os.environ.get("HF_LEROBOT_HOME", DEFAULT_LEROBOT_HOME)) / args.repo_id
    )
    intervention_output_root: Path | None = None
    intervention_repo_id = args.intervention_repo_id or default_intervention_repo_id(args.repo_id)
    intervention_scan: ScanResult | None = None
    if references and not args.no_intervention_output:
        intervention_output_root = (
            Path(args.intervention_root).expanduser().resolve()
            if args.intervention_root
            else default_intervention_output_root(output_root)
        )
        if intervention_output_root == output_root:
            raise ValueError("Intervention output root must differ from full output root")
        intervention_scan = scan_sources(
            references,
            scan_config,
            max_episodes_per_source=None,
            drop_failed_episodes=args.drop_failed_episodes,
            drop_unknown_episodes=args.drop_unknown_episodes,
        )
        print(
            f"Cleaned intervention output: episodes={len(intervention_scan.plans)} "
            f"frames={intervention_scan.total_frames} root={intervention_output_root}"
        )

    planned_outputs = [output_root]
    if intervention_output_root is not None:
        planned_outputs.append(intervention_output_root)
    if not args.overwrite:
        existing = [str(path) for path in planned_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f"Output exists: {existing}. Pass --overwrite to replace it."
            )

    write_dataset(
        scan,
        config,
        output_root=output_root,
        repo_id=args.repo_id,
        reference_audit=reference_audit,
        dataset_view="full",
        overwrite=args.overwrite,
        skip_videos=args.skip_videos,
        video_workers=args.video_workers,
        validate_samples=0 if args.no_validate else args.validation_samples,
    )
    if intervention_scan is not None and intervention_output_root is not None:
        write_dataset(
            intervention_scan,
            config,
            output_root=intervention_output_root,
            repo_id=intervention_repo_id,
            reference_audit=reference_audit,
            dataset_view="intervention_segments",
            overwrite=args.overwrite,
            skip_videos=args.skip_videos,
            video_workers=args.video_workers,
            validate_samples=0 if args.no_validate else args.validation_samples,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
