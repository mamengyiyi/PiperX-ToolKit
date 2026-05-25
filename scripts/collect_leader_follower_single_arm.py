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

from piperx_toolkit.collect.simple_zarr import append_episode, next_episode_id, require_array
from piperx_toolkit.env.cameras import CameraConfig, CameraManager
from piperx_toolkit.env.piper_arm import PiperArmConfig
from piperx_toolkit.teleop.leader_follower import JointMapping, LeaderFollowerPair, StepResult
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


def parse_vec7(text: str, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"{name} must contain 7 comma-separated values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def camera_device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def ensure_rgb_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")
    if image.shape[0] == height and image.shape[1] == width:
        return image
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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
        print(f"WARNING: could not open camera; rgb_front will be black. Last error: {last_error}")
        return None
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not open camera")


def open_or_create_dataset(path: str, args: argparse.Namespace) -> tuple[Any, Any, int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required. Install with `uv pip install -e '.[hardware,data]'`.") from exc

    try:
        root = zarr.open(path, mode="a", zarr_format=2)
    except TypeError:
        root = zarr.open(path, mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")
    image_shape = (3, args.height, args.width)

    require_array(data, "rgb_front", image_shape, np.uint8, chunks=(1, *image_shape))
    for key in (
        "joint_pos",
        "eef_pos",
        "joint_qvel",
        "joint_effort",
        "action",
        "leader_joint_pos",
        "leader_eef_pos",
        "target_joint_pos",
        "command_joint_pos",
    ):
        require_array(data, key, (7,), np.float32, chunks=(1024, 7))
    require_array(data, "timestamp", (), np.float64, chunks=(4096,))
    require_array(data, "episode", (), np.uint32, chunks=(4096,))
    if "episode_ends" not in meta:
        meta.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(1024,))

    meta.attrs["config"] = json.dumps(
        {
            "robot_type": "piperx_single_arm_leader_follower",
            "side": args.side,
            "leader_can": args.leader_can,
            "follower_can": args.follower_can,
            "leader_backend": args.leader_backend,
            "follower_backend": args.follower_backend,
            "camera": "front",
            "camera_backend": args.camera_backend,
            "camera_device": str(args.camera_device),
            "image_size": [args.width, args.height],
            "hz": args.hz,
            "task": args.task,
            "action_mode": "absolute_joint",
            "action_from": "leader_follower_command_joint_pos",
            "joint_signs": args.joint_signs,
            "joint_offsets": args.joint_offsets,
            "max_joint_delta_rad": args.max_joint_delta_rad,
            "lowpass_alpha": args.lowpass_alpha,
        },
        ensure_ascii=True,
    )
    return data, meta, next_episode_id(data)


def empty_buffer() -> dict[str, list[np.ndarray]]:
    return {
        "rgb_front": [],
        "joint_pos": [],
        "eef_pos": [],
        "joint_qvel": [],
        "joint_effort": [],
        "action": [],
        "leader_joint_pos": [],
        "leader_eef_pos": [],
        "target_joint_pos": [],
        "command_joint_pos": [],
        "timestamp": [],
        "episode": [],
    }


def _array_or_zeros(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(7, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(7)


def capture_step(
    pair: LeaderFollowerPair,
    execute: bool,
    cameras: CameraManager | None,
    buffer: dict[str, list[np.ndarray]],
    episode_id: int,
    args: argparse.Namespace,
    last_good_image: np.ndarray | None,
) -> tuple[np.ndarray | None, int, StepResult]:
    result = pair.step(execute=execute)
    buffer["joint_pos"].append(result.follower_joint_pos.reshape(1, 7).astype(np.float32))
    buffer["eef_pos"].append(_array_or_zeros(result.follower_eef_pos).reshape(1, 7))
    buffer["joint_qvel"].append(_array_or_zeros(result.follower_joint_qvel).reshape(1, 7))
    buffer["joint_effort"].append(_array_or_zeros(result.follower_joint_effort).reshape(1, 7))
    buffer["action"].append(result.command_joint_pos.reshape(1, 7).astype(np.float32))
    buffer["leader_joint_pos"].append(result.leader_joint_pos.reshape(1, 7).astype(np.float32))
    buffer["leader_eef_pos"].append(_array_or_zeros(result.leader_eef_pos).reshape(1, 7))
    buffer["target_joint_pos"].append(result.target_joint_pos.reshape(1, 7).astype(np.float32))
    buffer["command_joint_pos"].append(result.command_joint_pos.reshape(1, 7).astype(np.float32))
    buffer["timestamp"].append(np.array([result.timestamp if result.timestamp is not None else time.time()], dtype=np.float64))
    buffer["episode"].append(np.array([episode_id], dtype=np.uint32))

    camera_failures = 0
    image = last_good_image
    if cameras is not None:
        last_error: Exception | None = None
        for _ in range(max(1, args.camera_read_retries)):
            try:
                image = ensure_rgb_size(cameras.read_all()["front"], args.width, args.height)
                break
            except RuntimeError as exc:
                camera_failures += 1
                last_error = exc
                time.sleep(0.02)
        else:
            if not args.camera_fail_soft and last_error is not None:
                raise last_error
    if image is None:
        image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    buffer["rgb_front"].append(image.transpose(2, 0, 1)[None])
    return image, camera_failures, result


def finalize_buffer(
    buffer: dict[str, list[np.ndarray]],
    t0: float,
    camera_failures: int,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    episode = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
    duration = time.time() - t0
    steps = int(episode["timestamp"].shape[0])
    return EpisodeStats(steps, duration, steps / max(duration, 1e-6), camera_failures), episode


def make_pair(args: argparse.Namespace) -> LeaderFollowerPair:
    mapping = JointMapping(
        signs=parse_vec7(args.joint_signs, "--joint-signs"),
        offsets=parse_vec7(args.joint_offsets, "--joint-offsets"),
    )
    leader_config = PiperArmConfig(
        name=f"{args.side}_leader",
        can_name=args.leader_can,
        backend=args.leader_backend,
        enable_on_connect=False,
    )
    follower_config = PiperArmConfig(
        name=f"{args.side}_follower",
        can_name=args.follower_can,
        backend=args.follower_backend,
        enable_on_connect=False,
        speed_ratio=args.speed_ratio,
        high_follow=not args.no_high_follow,
    )
    return LeaderFollowerPair.from_configs(
        leader_config=leader_config,
        follower_config=follower_config,
        mapping=mapping,
        max_joint_delta_rad=args.max_joint_delta_rad,
        lowpass_alpha=args.lowpass_alpha,
    )


def record_for_duration(
    pair: LeaderFollowerPair,
    execute: bool,
    cameras: CameraManager | None,
    episode_id: int,
    args: argparse.Namespace,
    duration_s: float,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    buffer = empty_buffer()
    dt = 1.0 / max(args.hz, 1e-6)
    t0 = time.time()
    deadline = time.monotonic() + duration_s
    last_good_image = None
    camera_failures = 0
    while time.monotonic() < deadline:
        loop_t = time.monotonic()
        last_good_image, failures, _ = capture_step(pair, execute, cameras, buffer, episode_id, args, last_good_image)
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return finalize_buffer(buffer, t0, camera_failures)


def record_interactive(
    pair: LeaderFollowerPair,
    execute: bool,
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
    dt = 1.0 / max(args.hz, 1e-6)
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
        last_good_image, failures, _ = capture_step(pair, execute, cameras, buffer, episode_id, args, last_good_image)
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not buffer["timestamp"]:
        return EpisodeStats(0, 0.0, 0.0, camera_failures), None
    return finalize_buffer(buffer, t0, camera_failures)


def prepare_pair(pair: LeaderFollowerPair, execute: bool, args: argparse.Namespace) -> None:
    if not execute:
        print("DRY-RUN: follower commands will not be sent. Add --execute to record executable demonstrations.")
    pair.start(
        set_motion_output_role=args.set_motion_output_role,
        enable_follower=execute and args.enable_follower,
        startup_sleep_s=args.startup_sleep_s,
    )
    initial = pair.step(execute=False)
    pair.previous_command = None
    print(
        "Initial snapshot: "
        f"max_err={initial.max_abs_error_rad:.4f} rad "
        f"leader={np.array2string(initial.leader_joint_pos, precision=4, suppress_small=True)} "
        f"follower={np.array2string(initial.follower_joint_pos, precision=4, suppress_small=True)}"
    )
    if execute and initial.max_abs_error_rad > args.require_near_rad and not args.approach_start:
        raise RuntimeError(
            f"Follower is {initial.max_abs_error_rad:.4f} rad away from mapped leader target. "
            "Move both arms near each other, or pass --approach-start for a slow automatic approach."
        )
    if execute and args.approach_start:
        result = pair.approach_to_leader(
            hz=args.approach_hz,
            tolerance_rad=args.start_tolerance_rad,
            timeout_s=args.approach_timeout_s,
            execute=True,
        )
        pair.previous_command = None
        print(f"Approach done: max_err={result.max_abs_error_rad:.4f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect single-arm PiperX leader-follower demonstrations into Zarr.")
    parser.add_argument("--leader-can", default="can0")
    parser.add_argument("--follower-can", default="can1")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--leader-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--follower-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--dataset", "-d", default="datasets/leader_follower_single_arm.zarr")
    parser.add_argument("--episodes", "-n", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None, help="Auto-record each episode for this many seconds.")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--task", default="")
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-sleep-s", type=float, default=0.2)
    parser.add_argument("--enable-follower", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08)
    parser.add_argument("--lowpass-alpha", type=float, default=0.8)
    parser.add_argument("--joint-signs", default="1,1,1,1,1,1,1")
    parser.add_argument("--joint-offsets", default="0,0,0,0,0,0,0")
    parser.add_argument("--require-near-rad", type=float, default=0.35)
    parser.add_argument("--approach-start", action="store_true")
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-timeout-s", type=float, default=20.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the follower arm.")
    parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    parser.add_argument("--camera-device", default="4", help="OpenCV index or /dev/v4l/by-id/* path for front camera.")
    parser.add_argument("--camera-fail-soft", action="store_true")
    parser.add_argument("--camera-open-retries", type=int, default=3)
    parser.add_argument("--camera-open-retry-s", type=float, default=0.5)
    parser.add_argument("--camera-read-retries", type=int, default=10)
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    setup_logging()

    os.makedirs(os.path.dirname(args.dataset) or ".", exist_ok=True)
    data, meta, start_episode = open_or_create_dataset(args.dataset, args)
    pair = make_pair(args)
    cameras = connect_camera(args)

    try:
        prepare_pair(pair, args.execute, args)
        if args.duration is not None:
            for offset in range(args.episodes):
                episode_id = start_episode + offset
                stats, episode = record_for_duration(pair, args.execute, cameras, episode_id, args, args.duration)
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
                stats, episode = record_interactive(pair, args.execute, cameras, episode_id, args, kb)
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
        pair.stop()


if __name__ == "__main__":
    main()
