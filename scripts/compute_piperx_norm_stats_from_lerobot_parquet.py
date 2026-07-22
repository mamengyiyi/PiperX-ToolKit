#!/usr/bin/env python3
"""Compute PiperX state/action norm stats directly from LeRobot parquet files.

This is a fast path for PiperX LeRobot datasets whose norm stats only need
`observation.state` and chunked `action` values. It avoids the standard
LeRobot DataLoader because that path decodes videos even though RGB frames are
not used for normalization.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as openpi_config


def _resolve_repo_root(repo_id_or_path: str) -> Path:
    path = Path(repo_id_or_path).expanduser()
    if path.exists():
        return path
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if not hf_lerobot_home:
        raise FileNotFoundError(
            f"{repo_id_or_path} does not exist and HF_LEROBOT_HOME is not set."
        )
    path = Path(hf_lerobot_home) / repo_id_or_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _array_column(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    values = table[column].to_pylist()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected {column} to be 2D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: non-finite values in {column}")
    return array


def _chunk_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"expected actions to be 2D, got shape {actions.shape}")
    n = actions.shape[0]
    if n == 0:
        return actions.reshape(0, actions.shape[-1])
    offsets = np.arange(horizon, dtype=np.int64)
    base = np.arange(n, dtype=np.int64)[:, None]
    indices = np.minimum(base + offsets[None, :], n - 1)
    return actions[indices].reshape(-1, actions.shape[-1])


def _load_episode_files(root: Path, episodes: int | None) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    if episodes is not None:
        files = files[:episodes]
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    args = parser.parse_args()

    config = openpi_config.get_config(args.config_name)
    repo_root = _resolve_repo_root(args.repo_id)
    horizon = args.action_horizon or config.model.action_horizon
    if horizon <= 0:
        raise ValueError(f"invalid action horizon: {horizon}")

    info_path = repo_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if key not in features:
            raise ValueError(f"{repo_root}: missing feature {key}")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    total_frames = 0
    total_action_vectors = 0

    files = _load_episode_files(repo_root, args.episodes)
    for path in tqdm.tqdm(files, desc="Computing parquet stats"):
        states = _array_column(path, "observation.state")
        actions = _array_column(path, "action")
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"{path}: state/action length mismatch: {states.shape[0]} vs {actions.shape[0]}"
            )
        state_stats.update(states)
        action_chunks = _chunk_actions(actions, horizon)
        action_stats.update(action_chunks)
        total_frames += int(states.shape[0])
        total_action_vectors += int(action_chunks.shape[0])

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    if args.output_dir:
        output_path = Path(args.output_dir).expanduser()
    else:
        data_config = dataclasses.replace(config.data, repo_id=args.repo_id).create(config.assets_dirs, config.model)
        output_path = Path(config.assets_dirs) / data_config.repo_id

    normalize.save(output_path, norm_stats)
    print(f"Writing stats to: {output_path}")
    print(f"episodes={len(files)} frames={total_frames} action_vectors={total_action_vectors} horizon={horizon}")


if __name__ == "__main__":
    main()
