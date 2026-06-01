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
from piperx_toolkit.teleop.leader_follower import BimanualLeaderFollowerTeleop, JointMapping, LeaderFollowerPair, StepResult
from piperx_toolkit.utils.logging import setup_logging

CAMERA_NAMES = ("front", "left_wrist", "right_wrist")


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


def camera_devices(args: argparse.Namespace) -> dict[str, str]:
    return {
        "front": args.front_device,
        "left_wrist": args.left_wrist_device,
        "right_wrist": args.right_wrist_device,
    }


def ensure_rgb_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")
    if image.shape[0] == height and image.shape[1] == width:
        return image
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_cameras(args: argparse.Namespace) -> CameraManager:
    configs = {}
    for name, device in camera_devices(args).items():
        if str(device).strip() == "":
            continue
        configs[name] = CameraConfig(
            name=name,
            backend=args.camera_backend,
            device=camera_device_arg(device),
            width=args.width,
            height=args.height,
            fps=int(args.hz),
        )
    return CameraManager(configs)


def connect_cameras(args: argparse.Namespace) -> CameraManager | None:
    if args.no_camera:
        print("Cameras disabled by --no-camera; all RGB streams will be filled with black frames.")
        return None

    configured = {name: device for name, device in camera_devices(args).items() if str(device).strip()}
    if not configured:
        print("No camera devices configured; all RGB streams will be filled with black frames.")
        return None

    last_error: Exception | None = None
    for attempt in range(1, args.camera_open_retries + 1):
        cameras = make_cameras(args)
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
        print(f"WARNING: could not open cameras; RGB streams will be black. Last error: {last_error}")
        return None
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not open cameras")


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

    for cam in CAMERA_NAMES:
        require_array(data, f"rgb_{cam}", image_shape, np.uint8, chunks=(1, *image_shape))

    for side in ("left", "right"):
        for suffix in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort"):
            require_array(data, f"{side}_{suffix}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"action_{side}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"leader_{side}_joint_pos", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"leader_{side}_eef_pos", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"target_{side}_joint_pos", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"command_{side}_joint_pos", (7,), np.float32, chunks=(1024, 7))

    require_array(data, "timestamp", (), np.float64, chunks=(4096,))
    require_array(data, "episode", (), np.uint32, chunks=(4096,))
    if "episode_ends" not in meta:
        meta.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(1024,))

    meta.attrs["config"] = json.dumps(
        {
            "robot_type": "piperx_bimanual_leader_follower",
            "left_leader_can": args.left_leader_can,
            "left_follower_can": args.left_follower_can,
            "right_leader_can": args.right_leader_can,
            "right_follower_can": args.right_follower_can,
            "leader_backend": args.leader_backend,
            "follower_backend": args.follower_backend,
            "cameras": camera_devices(args),
            "camera_backend": args.camera_backend,
            "image_size": [args.width, args.height],
            "hz": args.hz,
            "task": args.task,
            "action_mode": "absolute_joint",
            "action_from": "leader_follower_command_joint_pos",
            "left_joint_signs": args.left_joint_signs,
            "left_joint_offsets": args.left_joint_offsets,
            "right_joint_signs": args.right_joint_signs,
            "right_joint_offsets": args.right_joint_offsets,
            "max_joint_delta_rad": args.max_joint_delta_rad,
            "lowpass_alpha": args.lowpass_alpha,
        },
        ensure_ascii=True,
    )
    return data, meta, next_episode_id(data)


def empty_buffer() -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {f"rgb_{cam}": [] for cam in CAMERA_NAMES}
    for side in ("left", "right"):
        for suffix in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort"):
            out[f"{side}_{suffix}"] = []
        out[f"action_{side}"] = []
        out[f"leader_{side}_joint_pos"] = []
        out[f"leader_{side}_eef_pos"] = []
        out[f"target_{side}_joint_pos"] = []
        out[f"command_{side}_joint_pos"] = []
    out["timestamp"] = []
    out["episode"] = []
    return out


def _array_or_zeros(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(7, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(7)


def append_result(buffer: dict[str, list[np.ndarray]], side: str, result: StepResult) -> None:
    buffer[f"{side}_joint_pos"].append(result.follower_joint_pos.reshape(1, 7).astype(np.float32))
    buffer[f"{side}_eef_pos"].append(_array_or_zeros(result.follower_eef_pos).reshape(1, 7))
    buffer[f"{side}_joint_qvel"].append(_array_or_zeros(result.follower_joint_qvel).reshape(1, 7))
    buffer[f"{side}_joint_effort"].append(_array_or_zeros(result.follower_joint_effort).reshape(1, 7))
    buffer[f"action_{side}"].append(result.command_joint_pos.reshape(1, 7).astype(np.float32))
    buffer[f"leader_{side}_joint_pos"].append(result.leader_joint_pos.reshape(1, 7).astype(np.float32))
    buffer[f"leader_{side}_eef_pos"].append(_array_or_zeros(result.leader_eef_pos).reshape(1, 7))
    buffer[f"target_{side}_joint_pos"].append(result.target_joint_pos.reshape(1, 7).astype(np.float32))
    buffer[f"command_{side}_joint_pos"].append(result.command_joint_pos.reshape(1, 7).astype(np.float32))


def capture_images(
    cameras: CameraManager | None,
    buffer: dict[str, list[np.ndarray]],
    args: argparse.Namespace,
    last_good_images: dict[str, np.ndarray | None],
) -> tuple[dict[str, np.ndarray | None], int]:
    camera_failures = 0
    images = dict(last_good_images)
    if cameras is not None:
        last_error: Exception | None = None
        for _ in range(max(1, args.camera_read_retries)):
            try:
                raw = cameras.read_all()
                for name, image in raw.items():
                    if name in CAMERA_NAMES:
                        images[name] = ensure_rgb_size(image, args.width, args.height)
                break
            except RuntimeError as exc:
                camera_failures += 1
                last_error = exc
                time.sleep(0.02)
        else:
            if not args.camera_fail_soft and last_error is not None:
                raise last_error

    for cam in CAMERA_NAMES:
        image = images.get(cam)
        if image is None:
            image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        buffer[f"rgb_{cam}"].append(image.transpose(2, 0, 1)[None])
        images[cam] = image
    return images, camera_failures


def capture_step(
    teleop: BimanualLeaderFollowerTeleop,
    execute: bool,
    cameras: CameraManager | None,
    buffer: dict[str, list[np.ndarray]],
    episode_id: int,
    args: argparse.Namespace,
    last_good_images: dict[str, np.ndarray | None],
) -> tuple[dict[str, np.ndarray | None], int, dict[str, StepResult]]:
    results = teleop.step(execute=execute)
    append_result(buffer, "left", results["left"])
    append_result(buffer, "right", results["right"])
    ts = max(
        results["left"].timestamp if results["left"].timestamp is not None else time.time(),
        results["right"].timestamp if results["right"].timestamp is not None else time.time(),
    )
    buffer["timestamp"].append(np.array([ts], dtype=np.float64))
    buffer["episode"].append(np.array([episode_id], dtype=np.uint32))
    images, failures = capture_images(cameras, buffer, args, last_good_images)
    return images, failures, results


def finalize_buffer(
    buffer: dict[str, list[np.ndarray]],
    t0: float,
    camera_failures: int,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    episode = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
    duration = time.time() - t0
    steps = int(episode["timestamp"].shape[0])
    return EpisodeStats(steps, duration, steps / max(duration, 1e-6), camera_failures), episode


def make_pair(
    side: str,
    leader_can: str,
    follower_can: str,
    signs: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> LeaderFollowerPair:
    return LeaderFollowerPair.from_configs(
        leader_config=PiperArmConfig(
            name=f"{side}_leader",
            can_name=leader_can,
            backend=args.leader_backend,
            enable_on_connect=False,
        ),
        follower_config=PiperArmConfig(
            name=f"{side}_follower",
            can_name=follower_can,
            backend=args.follower_backend,
            enable_on_connect=False,
            speed_ratio=args.speed_ratio,
            high_follow=not args.no_high_follow,
        ),
        mapping=JointMapping(signs=signs, offsets=offsets),
        max_joint_delta_rad=args.max_joint_delta_rad,
        lowpass_alpha=args.lowpass_alpha,
    )


def make_teleop(args: argparse.Namespace) -> BimanualLeaderFollowerTeleop:
    return BimanualLeaderFollowerTeleop(
        left_pair=make_pair(
            "left",
            args.left_leader_can,
            args.left_follower_can,
            parse_vec7(args.left_joint_signs, "--left-joint-signs"),
            parse_vec7(args.left_joint_offsets, "--left-joint-offsets"),
            args,
        ),
        right_pair=make_pair(
            "right",
            args.right_leader_can,
            args.right_follower_can,
            parse_vec7(args.right_joint_signs, "--right-joint-signs"),
            parse_vec7(args.right_joint_offsets, "--right-joint-offsets"),
            args,
        ),
    )


def record_for_duration(
    teleop: BimanualLeaderFollowerTeleop,
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
    last_good_images = {cam: None for cam in CAMERA_NAMES}
    camera_failures = 0
    while time.monotonic() < deadline:
        loop_t = time.monotonic()
        last_good_images, failures, _ = capture_step(
            teleop,
            execute,
            cameras,
            buffer,
            episode_id,
            args,
            last_good_images,
        )
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return finalize_buffer(buffer, t0, camera_failures)


def record_interactive(
    teleop: BimanualLeaderFollowerTeleop,
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
    last_good_images = {cam: None for cam in CAMERA_NAMES}
    camera_failures = 0
    while True:
        loop_t = time.monotonic()
        key = kb.get_key()
        if key in ("\r", "\n"):
            break
        if key == "\x03":
            raise KeyboardInterrupt
        last_good_images, failures, _ = capture_step(
            teleop,
            execute,
            cameras,
            buffer,
            episode_id,
            args,
            last_good_images,
        )
        camera_failures += failures
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not buffer["timestamp"]:
        return EpisodeStats(0, 0.0, 0.0, camera_failures), None
    return finalize_buffer(buffer, t0, camera_failures)


def prepare_teleop(teleop: BimanualLeaderFollowerTeleop, execute: bool, args: argparse.Namespace) -> None:
    if not execute:
        print("DRY-RUN: follower commands will not be sent. Add --execute to record executable demonstrations.")
    teleop.start(
        set_motion_output_role=args.set_motion_output_role,
        enable_followers=execute and args.enable_followers,
        startup_sleep_s=args.startup_sleep_s,
    )
    initial = teleop.step(execute=False)
    teleop.left_pair.previous_command = None
    teleop.right_pair.previous_command = None
    left_err = initial["left"].max_abs_error_rad
    right_err = initial["right"].max_abs_error_rad
    print(f"Initial snapshot: left_err={left_err:.4f} rad, right_err={right_err:.4f} rad")
    initial_max_err = max(left_err, right_err)
    if execute and initial_max_err > args.require_near_rad and not args.approach_start:
        raise RuntimeError(
            f"At least one follower is {initial_max_err:.4f} rad away from mapped leader target. "
            "Move all arms near their leaders, or pass --approach-start for a slow automatic approach."
        )
    if execute and args.approach_start:
        results = teleop.approach_to_leaders(
            hz=args.approach_hz,
            tolerance_rad=args.start_tolerance_rad,
            timeout_s=args.approach_timeout_s,
            execute=True,
        )
        teleop.left_pair.previous_command = None
        teleop.right_pair.previous_command = None
        print(
            "Approach done: "
            f"left_err={results['left'].max_abs_error_rad:.4f} rad, "
            f"right_err={results['right'].max_abs_error_rad:.4f} rad"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bimanual PiperX leader-follower demonstrations into Zarr.")
    parser.add_argument("--left-leader-can", default="can0")
    parser.add_argument("--left-follower-can", default="can1")
    parser.add_argument("--right-leader-can", default="can2")
    parser.add_argument("--right-follower-can", default="can3")
    parser.add_argument("--leader-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--follower-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--dataset", "-d", default="datasets/leader_follower_bimanual.zarr")
    parser.add_argument("--episodes", "-n", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None, help="Auto-record each episode for this many seconds.")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--task", default="")
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-sleep-s", type=float, default=0.2)
    parser.add_argument("--enable-followers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08)
    parser.add_argument("--lowpass-alpha", type=float, default=0.8)
    parser.add_argument("--left-joint-signs", default="1,1,1,1,1,1,1")
    parser.add_argument("--left-joint-offsets", default="0,0,0,0,0,0,0")
    parser.add_argument("--right-joint-signs", default="1,1,1,1,1,1,1")
    parser.add_argument("--right-joint-offsets", default="0,0,0,0,0,0,0")
    parser.add_argument("--require-near-rad", type=float, default=0.35)
    parser.add_argument("--approach-start", action="store_true")
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-timeout-s", type=float, default=20.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to follower arms.")
    #parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    #parser.add_argument("--front-device", default="10", help="OpenCV index or /dev/v4l/by-id/* path for main/front camera.")
    #parser.add_argument("--left-wrist-device", default="14", help="OpenCV index or /dev/v4l/by-id/* path for left follower wrist camera.")
    #parser.add_argument("--right-wrist-device", default="2", help="OpenCV index or /dev/v4l/by-id/* path for right follower wrist camera.")
    parser.add_argument("--camera-backend", default="realsense", choices=["mock", "opencv", "realsense"])
    parser.add_argument("--front-device", default="347622070427", help="RealSense serial, OpenCV index, or /dev/v4l/by-id/* path for front camera.")
    parser.add_argument("--left-wrist-device", default="260322278131", help="RealSense serial, OpenCV index, or /dev/v4l/by-id/* path for left wrist camera.")
    parser.add_argument("--right-wrist-device", default="335122272832", help="RealSense serial, OpenCV index, or /dev/v4l/by-id/* path for right wrist camera.")
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
    teleop = make_teleop(args)
    cameras = connect_cameras(args)

    try:
        prepare_teleop(teleop, args.execute, args)
        if args.duration is not None:
            for offset in range(args.episodes):
                episode_id = start_episode + offset
                stats, episode = record_for_duration(teleop, args.execute, cameras, episode_id, args, args.duration)
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
                stats, episode = record_interactive(teleop, args.execute, cameras, episode_id, args, kb)
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
        teleop.stop()


if __name__ == "__main__":
    main()
