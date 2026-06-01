#!/usr/bin/env python3
"""Zarr → LeRobot v3 数据集转换脚本 (ARX LIFT2) — 优化版.

相比 convert_to_lerobot_v3.py 的优化:
  - 整 episode 一次性 transpose（避免逐帧 numpy 调用）
  - 非视频模式下，PIL 转换使用线程池并行处理
  - 视频模式下启用 streaming_encoding（实时编码，跳过中间 PNG 写入）
  - add_frame 前预构建好所有 frame dict（减少循环内开销）
  - 更多 image_writer_threads (8)

用法::

    source .venv/bin/activate
    python data_collection/convert_to_lerobot_v3_fast.py \\
        --zarr data_collection/datasets/add_bottom_bread_vr_20260526_143202.zarr \\
        --output lerobot_datasets/add_bottom_bread \\
        --repo-id bluecontra/arx_add_bottom_bread \\
        --task "add bottom bread" --fps 30

    # --dry-run 只查看 Zarr 内容
    python data_collection/convert_to_lerobot_v3_fast.py \\
        --zarr data_collection/datasets/add_bottom_bread_vr_20260526_143202.zarr --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


_EXCLUDE_KEYS = {"timestamp", "episode"}
_IMAGE_PREFIXES = ("rgb_", "depth_")


def _is_numeric_array(key: str) -> bool:
    if key in _EXCLUDE_KEYS:
        return False
    for prefix in _IMAGE_PREFIXES:
        if key.startswith(prefix):
            return False
    return True


def _read_meta_config(meta_group) -> dict:
    raw = meta_group.attrs.get("config")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _get_action_mode(meta_group) -> str:
    return str(_read_meta_config(meta_group).get("action_mode", "unknown"))


def _get_episode_ranges(data_group, meta_group, max_episodes=None):
    if "episode_ends" in meta_group:
        episode_ends = meta_group["episode_ends"][:]
    else:
        print("[WARN] meta/episode_ends 缺失，从 data/episode 自动重建...")
        all_ep = data_group["episode"][:]
        unique_eps = np.unique(all_ep)
        ends = []
        running = 0
        for ep in unique_eps:
            running += int(np.sum(all_ep == ep))
            ends.append(running)
        episode_ends = np.array(ends, dtype=np.uint32)
        print(f"[INFO] 重建完成: {len(episode_ends)} 个 episode, ends={ends}")

    if max_episodes is not None and max_episodes < len(episode_ends):
        print(f"[INFO] 只使用前 {max_episodes} / {len(episode_ends)} 个 episode")
        episode_ends = episode_ends[:max_episodes]

    ranges = []
    prev = 0
    for end in episode_ends:
        ranges.append((int(prev), int(end)))
        prev = end
    return ranges


def _discover_arrays(data_group):
    numeric_arrays = []
    camera_set = set()
    for key in sorted(data_group.keys()):
        arr = data_group[key]
        if _is_numeric_array(key):
            numeric_arrays.append((key, arr.shape, str(arr.dtype)))
        elif key.startswith("rgb_"):
            cam_name = key[4:]
            camera_set.add(cam_name)
    return numeric_arrays, sorted(camera_set)


def _calc_dim(keys, data_group):
    total = 0
    for key in keys:
        shape = data_group[key].shape
        total += shape[1] if len(shape) > 1 else 1
    return total


def dry_run(zarr_path: str, max_episodes: int | None = None):
    import zarr
    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]
    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    action_mode = _get_action_mode(meta)
    numeric_arrays, camera_names = _discover_arrays(data)

    print(f"\n=== Zarr 数据集: {zarr_path} ===")
    print(f"Episodes: {n_eps}, 总帧数: {n_frames}")
    print(f"[INFO] Zarr action_mode = {action_mode}\n")

    print("  可用数组:")
    for idx, (key, shape, dtype) in enumerate(numeric_arrays):
        print(f"    [{idx}] {key:25s} {str(shape):20s} {dtype}")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print("\n  相机:")
    for i, cam in enumerate(camera_names):
        shape = data[f"rgb_{cam}"].shape
        label = labels[i] if i < len(labels) else str(i)
        print(f"    [{label}] {cam:25s} {str(shape)}")

    print(f"\n  [data/ 全部数组]")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"    {key:25s} shape={str(arr.shape):20s} dtype={arr.dtype}")
    print()


def _check_deps():
    missing = []
    for pkg in ("zarr", "lerobot", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("Pillow" if pkg == "PIL" else pkg)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("  uv pip install -e .")
        sys.exit(1)


def convert(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    state_keys: list[str],
    action_keys: list[str],
    camera_names: list[str],
    fps: int = 30,
    robot_type: str = "arx_lift2",
    task_name: str | None = None,
    max_episodes: int | None = None,
    use_videos: bool = False,
    streaming_encoding: bool = False,
    image_writer_threads: int = 8,
):
    import zarr
    from PIL import Image

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if task_name is None:
        task_name = Path(zarr_path).stem

    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    ep_ranges = _get_episode_ranges(data, meta, max_episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    action_mode = _get_action_mode(meta)

    state_dim = _calc_dim(state_keys, data)
    action_dim = _calc_dim(action_keys, data)

    first_cam_key = f"rgb_{camera_names[0]}"
    _, C, H, W = data[first_cam_key].shape
    image_shape = (H, W, C)

    print(f"[INFO] 图像尺寸: {W}x{H}, Episodes: {n_eps}, 总帧数: {n_frames}")
    print(f"[INFO] Zarr action_mode = {action_mode}")
    print(f"[INFO] state({state_dim}D) + action({action_dim}D) + {len(camera_names)} cameras")
    print(f"[INFO] use_videos={use_videos}, streaming_encoding={streaming_encoding}, "
          f"image_writer_threads={image_writer_threads}")

    output_path = Path(output_dir).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
    }
    image_dtype = "video" if use_videos else "image"
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    print(f"[INFO] 创建 LeRobot 数据集: repo_id={repo_id}, fps={fps}")
    print(f"[INFO] 输出路径: {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output_path,
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        streaming_encoding=streaming_encoding if use_videos else False,
    )

    def _read_and_concat(keys, start, end):
        arrays = []
        for key in keys:
            arr = data[key][start:end].astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
        return np.concatenate(arrays, axis=1)

    t0 = time.time()
    t_read = 0.0
    t_prep = 0.0
    t_write = 0.0

    for ep_idx, (start, end) in enumerate(ep_ranges):
        ep_len = end - start

        # --- Read: 批量读取所有数据 ---
        t1 = time.time()
        state_batch = _read_and_concat(state_keys, start, end)
        action_batch = _read_and_concat(action_keys, start, end)
        cam_batches = {}
        for cam in camera_names:
            cam_batches[cam] = data[f"rgb_{cam}"][start:end]  # (L, C, H, W)
        t_read += time.time() - t1

        # --- Prep: 一次性 transpose + PIL 并行转换 ---
        t1 = time.time()
        cam_data = {}
        for cam in camera_names:
            cam_data[cam] = cam_batches[cam].transpose(0, 2, 3, 1)  # (L, H, W, C)

        if use_videos:
            img_frames = cam_data  # ndarray，直接使用
        else:
            # 线程池并行 PIL 转换
            n_workers = min(image_writer_threads, len(camera_names) * 2)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for cam in camera_names:
                    futures[cam] = executor.submit(
                        lambda arr: [Image.fromarray(f) for f in arr],
                        cam_data[cam],
                    )
                img_frames = {cam: futures[cam].result() for cam in camera_names}
        t_prep += time.time() - t1

        # --- Write: 预构建 frame，减少循环内开销 ---
        t1 = time.time()
        for i in range(ep_len):
            frame = {
                "observation.state": state_batch[i],
                "action": action_batch[i],
                "task": task_name,
            }
            for cam in camera_names:
                frame[f"observation.images.{cam}"] = img_frames[cam][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        t_write += time.time() - t1

        print(f"  Episode {ep_idx}: {ep_len} steps")

    elapsed = time.time() - t0

    # Cleanup: video 模式下删除中间 images 目录
    if use_videos:
        image_dir = output_path / "images"
        if image_dir.exists():
            shutil.rmtree(image_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"[DONE] 转换完成!")
    print(f"  输出: {output_path}")
    print(f"  Episodes: {n_eps}, Frames: {n_frames}")
    print(f"  action_mode: {action_mode}")
    print(f"  state({state_dim}D) + action({action_dim}D) + {len(camera_names)} cameras")
    print(f"  总耗时: {elapsed:.1f}s ({n_frames / max(elapsed, 0.1):.0f} fps)")
    print(f"  - 读取:  {t_read:.1f}s")
    print(f"  - 预处理: {t_prep:.1f}s")
    print(f"  - 写入:  {t_write:.1f}s")
    if n_frames > 0:
        print(f"  - 每帧:  {elapsed / n_frames * 1000:.1f} ms")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Zarr → LeRobot v3 数据集转换 (ARX LIFT2) — 优化版",
    )
    parser.add_argument("--zarr", "-i", required=True, help="输入 Zarr 数据集路径")
    parser.add_argument("--output", "-o", default=None, help="LeRobot 输出目录")
    parser.add_argument("--repo-id", default=None, help="数据集 repo ID")
    parser.add_argument("--fps", type=int, default=30, help="帧率 (默认 30)")
    parser.add_argument("--robot-type", default="arx_lift2", help="机器人类型")
    parser.add_argument("--task", default=None, help="任务描述")
    parser.add_argument("--episodes", type=int, default=None, help="只转换前 N 个 episode")
    parser.add_argument("--dry-run", action="store_true", help="只列出 Zarr 中的数组")
    parser.add_argument("--use-videos", action="store_true", help="使用视频格式存储图像（默认图像格式）")
    parser.add_argument("--no-streaming", action="store_true", help="禁用实时视频编码（仅 video 模式有效）")
    parser.add_argument("--image-writer-threads", type=int, default=8, help="图像写入线程数")
    parser.add_argument("--state", default=None, help="非交互: state 字段（逗号分隔）")
    parser.add_argument("--action", default=None, help="非交互: action 字段（逗号分隔）")
    parser.add_argument("--cameras", default=None, help="非交互: 相机名（逗号分隔）")
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    if not zarr_path.exists():
        print(f"[ERROR] 输入路径不存在: {args.zarr}")
        sys.exit(1)

    _check_deps()
    import zarr

    if args.dry_run:
        dry_run(str(zarr_path), max_episodes=args.episodes)
        return

    if args.output is None:
        print("[ERROR] 请指定 --output 输出路径")
        sys.exit(1)
    if args.repo_id is None:
        print("[ERROR] 请指定 --repo-id")
        sys.exit(1)

    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    ep_ranges = _get_episode_ranges(data, meta, args.episodes)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    action_mode = _get_action_mode(meta)
    print(f"[INFO] Zarr action_mode = {action_mode}")

    numeric_arrays, camera_names = _discover_arrays(data)

    if args.state is not None and args.action is not None:
        state_keys = [k.strip() for k in args.state.split(",") if k.strip()]
        action_keys = [k.strip() for k in args.action.split(",") if k.strip()]
        cam_keys = [k.strip() for k in args.cameras.split(",")] if args.cameras else camera_names
    else:
        print(f"\n=== {n_eps} episodes, {n_frames} frames ===\n")
        print("可用数组:")
        for idx, (key, shape, dtype) in enumerate(numeric_arrays):
            print(f"  [{idx}] {key:25s} {str(shape):20s} {dtype}")
        print("\n相机:")
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, cam in enumerate(camera_names):
            label = labels[i] if i < len(labels) else str(i)
            print(f"  [{label}] {cam}")
        print("\n请使用 --state / --action / --cameras 指定字段")
        print(f"示例: --state left_joint_pos,right_joint_pos --action action_left,action_right "
              f"--cameras camera_h,camera_l,camera_r")
        sys.exit(1)

    use_videos = args.use_videos
    streaming_encoding = use_videos and not args.no_streaming

    convert(
        zarr_path=str(zarr_path),
        output_dir=args.output,
        repo_id=args.repo_id,
        state_keys=state_keys,
        action_keys=action_keys,
        camera_names=cam_keys,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task,
        max_episodes=args.episodes,
        use_videos=use_videos,
        streaming_encoding=streaming_encoding,
        image_writer_threads=args.image_writer_threads,
    )


if __name__ == "__main__":
    main()

