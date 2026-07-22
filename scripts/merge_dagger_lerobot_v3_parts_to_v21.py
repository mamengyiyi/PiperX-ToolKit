#!/usr/bin/env python3
"""Merge paired DAgger full/intervention LeRobot v3 parts into v2.1 datasets.

The two views are deliberately kept separate. Before writing output, this script
verifies that every intervention row is an exact projection of an intervention
frame in the full rollout view.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import merge_lerobot_v3_parts_to_v21 as generic_merge


DEFAULT_LEROBOT_HOME = Path("/root/data/my/piperx/lerobot_datasets")
REQUIRED_COLUMNS = {
    "observation.state",
    "action",
    "piperx.policy_action",
    "piperx.human_action",
    "piperx.executed_action",
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
    "episode_index",
}
VECTOR_COLUMNS = (
    "observation.state",
    "action",
    "piperx.policy_action",
    "piperx.human_action",
    "piperx.executed_action",
)
SCALAR_PROJECTION_COLUMNS = (
    "piperx.control_source",
    "piperx.intervention_mask",
    "piperx.episode_success",
    "piperx.policy_action_valid",
    "piperx.human_action_valid",
    "piperx.intervention_segment_index",
    "piperx.source_id",
    "piperx.towel_type_id",
)
IDENTITY_COLUMNS = (
    "piperx.source_id",
    "piperx.original_episode_index",
    "piperx.original_frame_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge paired DAgger full/intervention LeRobot v3 parts into two validated v2.1 datasets."
    )
    parser.add_argument("--full-src-roots", nargs="+", required=True)
    parser.add_argument("--intervention-src-roots", nargs="+", required=True)
    parser.add_argument("--full-repo-id", required=True)
    parser.add_argument(
        "--intervention-repo-id",
        default=None,
        help="Defaults to FULL_REPO_ID_intervention.",
    )
    parser.add_argument("--full-root", default=None)
    parser.add_argument("--intervention-root", default=None)
    parser.add_argument("--video-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paired DAgger sources and print the manifest without writing outputs.",
    )
    parser.add_argument("--no-reader-validation", action="store_true")
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--validation-video-backend", default="pyav")
    return parser.parse_args()


def _tmp_root(dst: Path, view: str) -> Path:
    return dst.with_name(f".{dst.name}.tmp-dagger-{view}-{os.getpid()}-{int(time.time())}")


def _old_root(dst: Path, view: str) -> Path:
    return dst.with_name(f".{dst.name}.old-dagger-{view}-{os.getpid()}-{int(time.time())}")


def _require_columns(data: pd.DataFrame, root: Path, view: str) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"{view} source {root} is missing DAgger columns: {missing}")


def _scalar_set(group: pd.DataFrame, column: str, context: str) -> set[Any]:
    values = set(group[column].tolist())
    if len(values) != 1:
        raise ValueError(f"{context} has inconsistent {column}: {sorted(values)}")
    return values


def _single_scalar(group: pd.DataFrame, column: str, context: str) -> Any:
    return next(iter(_scalar_set(group, column, context)))


def _vector_matrix(series: pd.Series, column: str, context: str) -> np.ndarray:
    try:
        matrix = np.stack([np.asarray(value) for value in series], axis=0)
    except Exception as exc:
        raise ValueError(f"Could not stack {column} in {context}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 14:
        raise ValueError(f"{column} must be [N,14] in {context}; got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{column} contains non-finite values in {context}")
    return matrix


def _validate_action_semantics(data: pd.DataFrame, context: str) -> None:
    action = _vector_matrix(data["action"], "action", context)
    executed = _vector_matrix(data["piperx.executed_action"], "piperx.executed_action", context)
    if not np.array_equal(action, executed):
        error = float(np.max(np.abs(action - executed)))
        raise ValueError(f"action != piperx.executed_action in {context}; max_abs_error={error}")


def _validate_full_root(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = generic_merge._read_data(root)
    _require_columns(data, root, "full")
    _validate_action_semantics(data, f"full source {root}")

    source_ids = sorted(set(int(value) for value in data["piperx.source_id"]))
    if len(source_ids) != 1:
        raise ValueError(f"Full source {root} must contain exactly one source_id, got {source_ids}")

    episode_keys: set[tuple[int, int]] = set()
    intervention_frames = 0
    success_counts: dict[str, int] = {}
    for episode_index, group in data.groupby("episode_index", sort=True):
        context = f"full source {root}, episode_index={episode_index}"
        source_id = int(_single_scalar(group, "piperx.source_id", context))
        original_episode = int(_single_scalar(group, "piperx.original_episode_index", context))
        _single_scalar(group, "piperx.episode_success", context)
        _single_scalar(group, "piperx.towel_type_id", context)
        key = (source_id, original_episode)
        if key in episode_keys:
            raise ValueError(f"Duplicate full rollout identity inside {root}: {key}")
        episode_keys.add(key)

        original_frames = group["piperx.original_frame_index"].to_numpy(dtype=np.int64)
        if len(np.unique(original_frames)) != len(original_frames):
            raise ValueError(f"Duplicate original_frame_index in {context}")
        if len(original_frames) > 1 and not np.all(np.diff(original_frames) == 1):
            raise ValueError(f"Non-contiguous original_frame_index in {context}")

        mask = group["piperx.intervention_mask"].to_numpy(dtype=np.int8)
        control = group["piperx.control_source"].to_numpy(dtype=np.int8)
        if not np.isin(mask, [0, 1]).all() or not np.isin(control, [0, 1]).all():
            raise ValueError(f"Illegal control source or intervention mask in {context}")
        if not np.array_equal(mask, control):
            raise ValueError(f"control_source and intervention_mask disagree in {context}")
        intervention_frames += int(mask.sum())

        success = str(int(group["piperx.episode_success"].iloc[0]))
        success_counts[success] = success_counts.get(success, 0) + 1

    return data, {
        "root": str(root),
        "source_id": source_ids[0],
        "episodes": len(episode_keys),
        "frames": len(data),
        "intervention_frames": intervention_frames,
        "episode_success_counts": success_counts,
    }


def _validate_intervention_root(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = generic_merge._read_data(root)
    _require_columns(data, root, "intervention")
    _validate_action_semantics(data, f"intervention source {root}")

    source_ids = sorted(set(int(value) for value in data["piperx.source_id"]))
    if len(source_ids) != 1:
        raise ValueError(f"Intervention source {root} must contain exactly one source_id, got {source_ids}")
    if not (data["piperx.intervention_mask"].astype(int) == 1).all():
        raise ValueError(f"Intervention source {root} contains non-intervention rows")
    if not (data["piperx.control_source"].astype(int) == 1).all():
        raise ValueError(f"Intervention source {root} contains policy control rows")
    if not (data["piperx.human_action_valid"].astype(int) == 1).all():
        raise ValueError(f"Intervention source {root} contains invalid human actions")

    segment_keys: set[tuple[int, int, int]] = set()
    for episode_index, group in data.groupby("episode_index", sort=True):
        context = f"intervention source {root}, episode_index={episode_index}"
        source_id = int(_single_scalar(group, "piperx.source_id", context))
        original_episode = int(_single_scalar(group, "piperx.original_episode_index", context))
        segment = int(_single_scalar(group, "piperx.intervention_segment_index", context))
        _single_scalar(group, "piperx.episode_success", context)
        _single_scalar(group, "piperx.towel_type_id", context)
        if segment < 0:
            raise ValueError(f"Negative intervention segment index in {context}")
        key = (source_id, original_episode, segment)
        if key in segment_keys:
            raise ValueError(f"Duplicate intervention segment identity inside {root}: {key}")
        segment_keys.add(key)

        original_frames = group["piperx.original_frame_index"].to_numpy(dtype=np.int64)
        if len(np.unique(original_frames)) != len(original_frames):
            raise ValueError(f"Duplicate original_frame_index in {context}")
        if len(original_frames) > 1 and not np.all(np.diff(original_frames) == 1):
            raise ValueError(f"Non-contiguous intervention segment in {context}")

    return data, {
        "root": str(root),
        "source_id": source_ids[0],
        "segments": len(segment_keys),
        "frames": len(data),
    }


def _sorted_projection(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def _assert_projection_equal(full: pd.DataFrame, intervention: pd.DataFrame, context: str) -> None:
    full_projection = _sorted_projection(full[full["piperx.intervention_mask"].astype(bool)].copy())
    intervention_projection = _sorted_projection(intervention.copy())
    if len(full_projection) != len(intervention_projection):
        raise ValueError(
            f"Intervention frame count mismatch in {context}: "
            f"full={len(full_projection)} intervention={len(intervention_projection)}"
        )

    for column in IDENTITY_COLUMNS + SCALAR_PROJECTION_COLUMNS:
        left = full_projection[column].to_numpy()
        right = intervention_projection[column].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Projection mismatch for {column} in {context}")

    for column in VECTOR_COLUMNS:
        left = _vector_matrix(full_projection[column], column, f"full projection {context}")
        right = _vector_matrix(intervention_projection[column], column, f"intervention projection {context}")
        if not np.array_equal(left, right):
            error = float(np.max(np.abs(left - right)))
            raise ValueError(f"Projection mismatch for {column} in {context}; max_abs_error={error}")


def validate_dagger_sources(
    full_roots: list[Path], intervention_roots: list[Path]
) -> dict[str, Any]:
    if len(full_roots) != len(intervention_roots):
        raise ValueError(
            f"Expected one intervention root per full root; got "
            f"{len(full_roots)} full and {len(intervention_roots)} intervention"
        )

    generic_merge._validate_sources(full_roots)
    generic_merge._validate_sources(intervention_roots)

    all_full_keys: set[tuple[int, int]] = set()
    all_segment_keys: set[tuple[int, int, int]] = set()
    seen_source_ids: set[int] = set()
    pairs: list[dict[str, Any]] = []

    for full_root, intervention_root in zip(full_roots, intervention_roots, strict=True):
        full, full_stats = _validate_full_root(full_root)
        intervention, intervention_stats = _validate_intervention_root(intervention_root)
        if full_stats["source_id"] != intervention_stats["source_id"]:
            raise ValueError(
                f"Paired roots have different source_id: {full_root} vs {intervention_root}"
            )
        source_id = int(full_stats["source_id"])
        if source_id in seen_source_ids:
            raise ValueError(f"source_id={source_id} appears in more than one source pair")
        seen_source_ids.add(source_id)

        full_keys = {
            (int(source), int(episode))
            for source, episode in full[["piperx.source_id", "piperx.original_episode_index"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        duplicate_full = all_full_keys & full_keys
        if duplicate_full:
            raise ValueError(f"Duplicate full rollout identities across roots: {sorted(duplicate_full)[:5]}")
        all_full_keys.update(full_keys)

        segment_keys = {
            (int(source), int(episode), int(segment))
            for source, episode, segment in intervention[
                [
                    "piperx.source_id",
                    "piperx.original_episode_index",
                    "piperx.intervention_segment_index",
                ]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        duplicate_segments = all_segment_keys & segment_keys
        if duplicate_segments:
            raise ValueError(f"Duplicate intervention segment identities across roots: {sorted(duplicate_segments)[:5]}")
        all_segment_keys.update(segment_keys)

        _assert_projection_equal(full, intervention, f"source_id={source_id}")
        pairs.append({"full": full_stats, "intervention": intervention_stats})

    return {
        "source_pairs": pairs,
        "source_ids": sorted(seen_source_ids),
        "full_episodes": sum(pair["full"]["episodes"] for pair in pairs),
        "full_frames": sum(pair["full"]["frames"] for pair in pairs),
        "intervention_segments": sum(pair["intervention"]["segments"] for pair in pairs),
        "intervention_frames": sum(pair["intervention"]["frames"] for pair in pairs),
        "projection_validation": "exact",
    }


def _publish_pair(
    full_work: Path,
    intervention_work: Path,
    full_dst: Path,
    intervention_dst: Path,
) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for dst, view in ((full_dst, "full"), (intervention_dst, "intervention")):
            if dst.exists():
                backup = _old_root(dst, view)
                dst.rename(backup)
                backups[dst] = backup
        full_work.rename(full_dst)
        published.append(full_dst)
        intervention_work.rename(intervention_dst)
        published.append(intervention_dst)
    except Exception:
        for dst in reversed(published):
            if dst.exists():
                dst.rename(dst.with_name(f".{dst.name}.failed-publish-{os.getpid()}"))
        for dst, backup in backups.items():
            if backup.exists() and not dst.exists():
                backup.rename(dst)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)


def main() -> None:
    args = parse_args()
    full_roots = [Path(path).resolve() for path in args.full_src_roots]
    intervention_roots = [Path(path).resolve() for path in args.intervention_src_roots]
    for root in full_roots + intervention_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)

    intervention_repo_id = args.intervention_repo_id or f"{args.full_repo_id}_intervention"
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", DEFAULT_LEROBOT_HOME))
    full_dst = Path(args.full_root).resolve() if args.full_root else (lerobot_home / args.full_repo_id).resolve()
    intervention_dst = (
        Path(args.intervention_root).resolve()
        if args.intervention_root
        else (lerobot_home / intervention_repo_id).resolve()
    )
    if full_dst == intervention_dst:
        raise ValueError("Full and intervention outputs must be different directories")

    print("Validating DAgger source pairs...")
    manifest = validate_dagger_sources(full_roots, intervention_roots)
    manifest.update(
        {
            "full_repo_id": args.full_repo_id,
            "intervention_repo_id": intervention_repo_id,
            "full_output": str(full_dst),
            "intervention_output": str(intervention_dst),
            "views_are_separate": True,
            "full_and_intervention_must_not_be_mixed_1_to_1": True,
        }
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.validate_only:
        print("DAgger source validation OK (--validate-only).")
        return

    for dst in (full_dst, intervention_dst):
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {dst}. Pass --overwrite to replace it.")

    full_work = _tmp_root(full_dst, "full")
    intervention_work = _tmp_root(intervention_dst, "intervention")
    for work in (full_work, intervention_work):
        if work.exists():
            shutil.rmtree(work)

    try:
        generic_merge._write_merged_v21(
            src_roots=full_roots,
            dst_root=full_work,
            repo_id=args.full_repo_id,
            max_episodes=None,
            video_workers=args.video_workers,
            skip_videos=args.skip_videos,
        )
        generic_merge._write_merged_v21(
            src_roots=intervention_roots,
            dst_root=intervention_work,
            repo_id=intervention_repo_id,
            max_episodes=None,
            video_workers=args.video_workers,
            skip_videos=args.skip_videos,
        )

        for work in (full_work, intervention_work):
            (work / "meta" / "piperx_dagger_merge.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        if not args.no_reader_validation and not args.skip_videos:
            generic_merge._validate_output_dataset(
                full_work,
                args.full_repo_id,
                args.validation_samples,
                args.validation_video_backend,
            )
            generic_merge._validate_output_dataset(
                intervention_work,
                intervention_repo_id,
                args.validation_samples,
                args.validation_video_backend,
            )

        _publish_pair(full_work, intervention_work, full_dst, intervention_dst)
    except Exception:
        print(f"[ERROR] Incomplete full output: {full_work}", file=sys.stderr)
        print(f"[ERROR] Incomplete intervention output: {intervention_work}", file=sys.stderr)
        raise

    print(f"Published full: {full_dst}")
    print(f"Published intervention: {intervention_dst}")


if __name__ == "__main__":
    main()
