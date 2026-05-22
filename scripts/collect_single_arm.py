#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.cameras import CameraConfig, CameraManager
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.utils.logging import setup_logging


class KeyboardListener:
    def __init__(self):
        import termios

        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)

    def start(self) -> None:
        import tty

        tty.setraw(sys.stdin.fileno())

    def stop(self) -> None:
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        import select

        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None


@dataclass
class EpisodeStats:
    steps: int
    duration_s: float
    fps: float
    camera_failures: int


def compressor() -> Any:
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        return None


def require_array(group: Any, name: str, shape_tail: tuple[int, ...], dtype: Any, chunks: tuple[int, ...] | None = None) -> Any:
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
    new_shape = (old + values.shape[0], *array.shape[1:])
    array.resize(new_shape)
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


def open_or_create_dataset(path: str, args: argparse.Namespace) -> tuple[Any, Any, int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required. Install with `uv pip install -e '.[hardware,data]'`.") from exc

    exists = os.path.exists(path)
    try:
        root = zarr.open(path, mode="a", zarr_format=2)
    except TypeError:
        root = zarr.open(path, mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")
    image_shape = (3, args.height, args.width)

    require_array(data, "rgb_front", image_shape, np.uint8, chunks=(1, *image_shape))
    for key in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort", "action"):
        require_array(data, key, (7,), np.float32, chunks=(1024, 7))
    require_array(data, "timestamp", (), np.float64, chunks=(4096,))
    require_array(data, "episode", (), np.uint32, chunks=(4096,))
    if "episode_ends" not in meta:
        meta.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(1024,))

    meta.attrs["config"] = json.dumps(
        {
            "robot_type": "piperx_single_arm",
            "side": args.side,
            "can": args.can,
            "camera": "front",
            "camera_backend": args.camera_backend,
            "camera_device": str(args.camera_device),
            "image_size": [args.width, args.height],
            "hz": args.hz,
            "task": args.task,
            "action_mode": "absolute_joint",
            "action_from": "future_joint_pos",
            "action_shift_frames": args.action_shift_frames,
        },
        ensure_ascii=True,
    )

    start_episode = 0
    if exists and "episode" in data and len(data["episode"]) > 0:
        start_episode = int(np.max(data["episode"][:])) + 1
    return data, meta, start_episode


def ensure_rgb_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")
    if image.shape[0] == height and image.shape[1] == width:
        return image
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def camera_device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def make_camera(args: argparse.Namespace) -> CameraManager:
    return CameraManager(
        {
            "front": CameraConfig(
                name="front",
                backend=args.camera_backend,
                device=camera_device_arg(args.camera_device),
                width=args.width,
                height=args.height,
                fps=int(args.hz),
            )
        }
    )


def connect_camera(args: argparse.Namespace) -> CameraManager | None:
    if args.no_camera:
        print("Camera disabled by --no-camera; rgb_front will be filled with black frames.")
        return None

    last_error: Exception | None = None
    for attempt in range(1, args.camera_open_retries + 1):
        cameras = make_camera(args)
        try:
            cameras.connect()
            time.sleep(max(0.0, args.camera_warmup_s))
            return cameras
        except Exception as exc:
            last_error = exc
            cameras.close()
            if attempt < args.camera_open_retries:
                print(
                    f"Camera open failed on attempt {attempt}/{args.camera_open_retries}: {exc}. "
                    f"Retrying in {args.camera_open_retry_s:.1f}s..."
                )
                time.sleep(max(0.0, args.camera_open_retry_s))

    if args.camera_fail_soft:
        print(f"WARNING: could not open camera; rgb_front will be filled with black frames. Last error: {last_error}")
        return None
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not open camera")


def empty_buffer() -> dict[str, list[np.ndarray]]:
    return {
        "rgb_front": [],
        "joint_pos": [],
        "eef_pos": [],
        "joint_qvel": [],
        "joint_effort": [],
        "action": [],
        "timestamp": [],
        "episode": [],
    }


def capture_step(
    arm: PiperArm,
    cameras: CameraManager | None,
    buffer: dict[str, list[np.ndarray]],
    episode_id: int,
    args: argparse.Namespace,
    last_good_image: np.ndarray | None,
) -> tuple[np.ndarray | None, int]:
    state = arm.read_state()
    buffer["joint_pos"].append(state.joint_pos.reshape(1, 7).astype(np.float32))
    buffer["eef_pos"].append(state.eef_pos.reshape(1, 7).astype(np.float32))
    buffer["joint_qvel"].append(state.joint_qvel.reshape(1, 7).astype(np.float32))
    buffer["joint_effort"].append(state.joint_effort.reshape(1, 7).astype(np.float32))
    buffer["action"].append(state.joint_pos.reshape(1, 7).astype(np.float32))
    buffer["timestamp"].append(np.array([state.timestamp], dtype=np.float64))
    buffer["episode"].append(np.array([episode_id], dtype=np.uint32))

    camera_failures = 0
    image = last_good_image
    if cameras is not None:
        for _ in range(max(1, args.camera_read_retries)):
            try:
                image = ensure_rgb_size(cameras.read_all()["front"], args.width, args.height)
                break
            except RuntimeError as exc:
                camera_failures += 1
                last_error = exc
                time.sleep(0.02)
        else:
            if not args.camera_fail_soft:
                raise last_error
    if image is None:
        image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    buffer["rgb_front"].append(image.transpose(2, 0, 1)[None])
    return image, camera_failures


def finalize_buffer(buffer: dict[str, list[np.ndarray]], t0: float, args: argparse.Namespace, camera_failures: int) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    episode = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
    shift = args.action_shift_frames
    if shift > 0:
        source = episode["joint_pos"]
        action = source.copy()
        if len(source) > shift:
            action[:-shift] = source[shift:]
            action[-shift:] = source[-1]
        episode["action"] = action.astype(np.float32)
    duration = time.time() - t0
    stats = EpisodeStats(
        steps=int(episode["timestamp"].shape[0]),
        duration_s=duration,
        fps=float(episode["timestamp"].shape[0] / max(duration, 1e-6)),
        camera_failures=camera_failures,
    )
    return stats, episode


def append_episode(data: Any, meta: Any, episode: dict[str, np.ndarray]) -> None:
    for key, values in episode.items():
        append_array(data[key], values)
    recompute_episode_ends(data, meta)


def record_for_duration(
    arm: PiperArm,
    cameras: CameraManager | None,
    episode_id: int,
    args: argparse.Namespace,
    duration_s: float,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    buffer = empty_buffer()
    dt = 1.0 / args.hz
    t0 = time.time()
    deadline = time.monotonic() + duration_s
    last_good_image = None
    camera_failures = 0
    while time.monotonic() < deadline:
        loop_t = time.monotonic()
        last_good_image, failures = capture_step(arm, cameras, buffer, episode_id, args, last_good_image)
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return finalize_buffer(buffer, t0, args, camera_failures)


def record_interactive(
    arm: PiperArm,
    cameras: CameraManager | None,
    episode_id: int,
    args: argparse.Namespace,
    kb: KeyboardListener,
) -> tuple[EpisodeStats, dict[str, np.ndarray] | None]:
    print(f"\r\nEpisode {episode_id}: Space start, Enter stop, Ctrl+C quit.\r\n")
    while True:
        key = kb.get_key()
        if key == " ":
            break
        if key == "\x03":
            raise KeyboardInterrupt
        time.sleep(0.05)

    print(f"\r\nRecording episode {episode_id}. Press Enter to stop.\r\n")
    buffer = empty_buffer()
    dt = 1.0 / args.hz
    t0 = time.time()
    last_good_image = None
    camera_failures = 0
    while True:
        loop_t = time.monotonic()
        key = kb.get_key()
        if key in ("\r", "\n"):
            break
        if key == "\x03":
            raise KeyboardInterrupt
        last_good_image, failures = capture_step(arm, cameras, buffer, episode_id, args, last_good_image)
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not buffer["timestamp"]:
        return EpisodeStats(0, 0.0, 0.0, camera_failures), None
    return finalize_buffer(buffer, t0, args, camera_failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one PiperX arm and one RGB camera into a Zarr dataset.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--dataset", "-d", default="datasets/single_arm_front.zarr")
    parser.add_argument("--episodes", "-n", type=int, default=1, help="Number of new episodes to collect.")
    parser.add_argument("--duration", type=float, default=None, help="Auto-record each episode for this many seconds.")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--task", default="")
    parser.add_argument("--action-shift-frames", type=int, default=1)
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    parser.add_argument("--camera-device", default="4", help="OpenCV index or /dev/v4l/by-id/* path for front camera.")
    parser.add_argument("--camera-fail-soft", action="store_true")
    parser.add_argument("--camera-open-retries", type=int, default=3)
    parser.add_argument("--camera-open-retry-s", type=float, default=0.5)
    parser.add_argument("--camera-read-retries", type=int, default=10)
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true", help="Debug only: save black rgb_front frames.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    setup_logging()

    os.makedirs(os.path.dirname(args.dataset) or ".", exist_ok=True)
    data, meta, start_episode = open_or_create_dataset(args.dataset, args)

    arm = PiperArm(PiperArmConfig(name=args.side, can_name=args.can, backend=args.backend))
    arm.connect()
    cameras = connect_camera(args)
    if args.set_motion_output_role:
        arm.set_motion_output_role()
        time.sleep(0.2)

    try:
        if args.duration is not None:
            for offset in range(args.episodes):
                episode_id = start_episode + offset
                stats, episode = record_for_duration(arm, cameras, episode_id, args, args.duration)
                append_episode(data, meta, episode)
                print(
                    f"Saved episode {episode_id}: {stats.steps} steps, "
                    f"{stats.duration_s:.1f}s, {stats.fps:.1f} FPS, camera_failures={stats.camera_failures}"
                )
            return

        kb = KeyboardListener()
        saved = 0
        try:
            kb.start()
            for offset in range(args.episodes):
                episode_id = start_episode + offset
                stats, episode = record_interactive(arm, cameras, episode_id, args, kb)
                if episode is None:
                    continue
                print(
                    f"\r\nEpisode {episode_id}: {stats.steps} steps, "
                    f"{stats.duration_s:.1f}s, {stats.fps:.1f} FPS, camera_failures={stats.camera_failures}. "
                    "[S] save / [D] discard\r\n"
                )
                while True:
                    key = kb.get_key()
                    if key == "s":
                        append_episode(data, meta, episode)
                        saved += 1
                        print(f"\r\nSaved episode {episode_id} -> {args.dataset}\r\n")
                        break
                    if key == "d":
                        print(f"\r\nDiscarded episode {episode_id}.\r\n")
                        break
                    if key == "\x03":
                        raise KeyboardInterrupt
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\r\nInterrupted, closing collector.\r\n")
        finally:
            kb.stop()
            print(f"Collection done. Saved {saved} new episode(s) to {args.dataset}.")
    finally:
        if cameras is not None:
            cameras.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
