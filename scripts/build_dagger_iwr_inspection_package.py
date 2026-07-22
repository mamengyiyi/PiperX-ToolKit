#!/usr/bin/env python3
"""Build a small, identity-aligned inspection package from DAgger/IWR datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


IDENTITY_COLUMNS = (
    "piperx.source_id",
    "piperx.original_episode_index",
    "episode_index",
)

TIMELINE_COLUMNS = (
    "frame_index",
    "piperx.original_frame_index",
    "piperx.control_source",
    "piperx.intervention_mask",
    "piperx.train_mask",
    "piperx.stall_mask",
    "piperx.switch_mask",
    "piperx.sample_weight",
    "piperx.indicator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-full", type=Path, required=True)
    parser.add_argument("--cleaned-full", type=Path, required=True)
    parser.add_argument("--original-intervention", type=Path, required=True)
    parser.add_argument("--cleaned-intervention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", type=int, default=0)
    parser.add_argument("--original-episode-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_values(table: Any, name: str) -> list[Any]:
    if name not in table.column_names:
        return []
    return table[name].combine_chunks().to_pylist()


def find_matching_parquets(root: Path, source_id: int, original_episode_index: int) -> list[Path]:
    matches: list[Path] = []
    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root}")

    for path in files:
        schema_names = pq.ParquetFile(path).schema_arrow.names
        missing = [name for name in IDENTITY_COLUMNS if name not in schema_names]
        if missing:
            raise ValueError(f"{path} is missing identity columns: {missing}")
        identity = pq.read_table(path, columns=list(IDENTITY_COLUMNS))
        if identity.num_rows == 0:
            continue
        source_values = set(scalar_values(identity, "piperx.source_id"))
        episode_values = set(scalar_values(identity, "piperx.original_episode_index"))
        if source_values == {source_id} and episode_values == {original_episode_index}:
            matches.append(path)
    return matches


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
        "sum": sum(finite),
    }


def summarize_table(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    result: dict[str, Any] = {
        "parquet": str(path),
        "frames": table.num_rows,
        "columns": table.column_names,
    }
    for name in (
        "episode_index",
        "piperx.source_id",
        "piperx.original_episode_index",
        "piperx.episode_success",
    ):
        values = scalar_values(table, name)
        if values:
            result[name] = sorted(set(values))

    for name in (
        "piperx.intervention_mask",
        "piperx.train_mask",
        "piperx.stall_mask",
        "piperx.switch_mask",
        "piperx.indicator",
    ):
        values = scalar_values(table, name)
        if values:
            result[f"{name}.true_frames"] = sum(bool(value) for value in values)

    control_values = scalar_values(table, "piperx.control_source")
    if control_values:
        result["piperx.control_source.counts"] = {
            str(value): control_values.count(value) for value in sorted(set(control_values))
        }

    weights = scalar_values(table, "piperx.sample_weight")
    if weights:
        result["piperx.sample_weight"] = numeric_summary(weights)
    return result


def copy_selected_dataset(
    label: str,
    root: Path,
    destination: Path,
    source_id: int,
    original_episode_index: int,
) -> dict[str, Any]:
    selected = find_matching_parquets(root, source_id, original_episode_index)
    if not selected:
        raise RuntimeError(
            f"No episode matched source_id={source_id}, "
            f"original_episode_index={original_episode_index} in {root}"
        )

    target = destination / label
    summaries: list[dict[str, Any]] = []
    copied_videos: list[str] = []
    for parquet in selected:
        relative = parquet.relative_to(root)
        parquet_target = target / relative
        parquet_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet, parquet_target)

        table = pq.read_table(parquet, columns=["episode_index"])
        episode_indices = sorted(set(scalar_values(table, "episode_index")))
        if len(episode_indices) != 1:
            raise ValueError(f"Expected one episode_index in {parquet}, got {episode_indices}")
        episode_index = int(episode_indices[0])
        for video in sorted(root.glob(f"videos/chunk-*/observation.images.*/episode_{episode_index:06d}.mp4")):
            video_relative = video.relative_to(root)
            video_target = target / video_relative
            video_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, video_target)
            copied_videos.append(str(video_relative))

        summaries.append(summarize_table(parquet))

    meta_source = root / "meta"
    if meta_source.is_dir():
        shutil.copytree(meta_source, target / "meta", dirs_exist_ok=True)

    return {
        "label": label,
        "source_root": str(root),
        "selected_parquets": [str(path.relative_to(root)) for path in selected],
        "selected_videos": copied_videos,
        "episodes_or_segments": len(selected),
        "frames": sum(item["frames"] for item in summaries),
        "details": summaries,
    }


def write_cleaned_full_timeline(cleaned_full_dir: Path, output: Path) -> None:
    parquets = sorted(cleaned_full_dir.glob("data/chunk-*/episode_*.parquet"))
    if len(parquets) != 1:
        raise ValueError(f"Expected one cleaned full parquet, got {len(parquets)}")
    table = pq.read_table(parquets[0])
    present_columns = [name for name in TIMELINE_COLUMNS if name in table.column_names]
    columns = {name: scalar_values(table, name) for name in present_columns}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=present_columns)
        writer.writeheader()
        for index in range(table.num_rows):
            writer.writerow({name: columns[name][index] for name in present_columns})


def write_readme(output: Path, summary: dict[str, Any]) -> None:
    selected = summary["selection"]
    text = f"""# Formal DAgger/IWR inspection package

This package contains the same physical rollout across conversion and cleaning outputs.

- source_id: {selected['source_id']}
- original_episode_index: {selected['original_episode_index']}
- full rollout: original and cleaned copies
- intervention view: every continuous intervention segment from that rollout

The cleaner keeps the full videos and rows. Cleaning decisions are represented in the cleaned
parquet with `piperx.train_mask`, `piperx.stall_mask`, `piperx.switch_mask`,
`piperx.sample_weight`, and `piperx.indicator`. Therefore the original and cleaned full videos
are expected to look identical. Use `cleaned_full_timeline.csv` and `summary.json` to inspect
which frames are trainable, stalled, near a control switch, or reweighted.

Each directory preserves the original dataset-relative parquet and video paths. The copied
`meta/` describes the complete source dataset and is included for reference; this package is an
inspection subset, not a standalone renumbered LeRobot dataset.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = {
        "original_full": args.original_full,
        "cleaned_full": args.cleaned_full,
        "original_intervention": args.original_intervention,
        "cleaned_intervention": args.cleaned_intervention,
    }
    for label, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{label} dataset does not exist: {root}")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    summary: dict[str, Any] = {
        "selection": {
            "source_id": args.source_id,
            "original_episode_index": args.original_episode_index,
        },
        "datasets": {},
    }
    for label, root in roots.items():
        summary["datasets"][label] = copy_selected_dataset(
            label,
            root,
            args.output,
            args.source_id,
            args.original_episode_index,
        )

    write_cleaned_full_timeline(args.output / "cleaned_full", args.output / "cleaned_full_timeline.csv")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
