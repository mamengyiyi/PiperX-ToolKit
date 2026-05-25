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

from piperx_toolkit.collect.schema import episode_ranges
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.utils.logging import setup_logging


@dataclass
class BimanualEpisodeData:
    left_actions: np.ndarray
    right_actions: np.ndarray
    start: int
    end: int
    fps: float
    meta: dict[str, Any]


def load_meta(meta: Any) -> dict[str, Any]:
    raw = meta.attrs.get("config", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _available_numeric_keys(data: Any) -> str:
    return ", ".join(sorted(k for k in data.keys() if not str(k).startswith("rgb_")))


def _load_action_slice(data: Any, key: str, start: int, end: int) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing action key '{key}'. Available numeric keys: {_available_numeric_keys(data)}")
    actions = np.asarray(data[key][start:end], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected {key} shape (N, 7), got {actions.shape}")
    return actions


def load_episode(
    path: str,
    episode_index: int,
    left_key: str,
    right_key: str,
    fps: float | None,
    max_frames: int | None,
) -> BimanualEpisodeData:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required. Install with `uv pip install -e '.[hardware,data]'`.") from exc

    root = zarr.open(path, mode="r")
    data = root["data"]
    meta = root["meta"]
    ranges = episode_ranges(data, meta)
    if episode_index < 0 or episode_index >= len(ranges):
        raise IndexError(f"Episode {episode_index} out of range. Dataset has {len(ranges)} episode(s).")

    start, end = ranges[episode_index]
    if max_frames is not None:
        end = min(end, start + max(0, max_frames))

    left_actions = _load_action_slice(data, left_key, start, end)
    right_actions = _load_action_slice(data, right_key, start, end)
    if left_actions.shape[0] != right_actions.shape[0]:
        raise ValueError(
            f"Left/right action lengths differ: {left_key}={left_actions.shape[0]}, "
            f"{right_key}={right_actions.shape[0]}"
        )

    cfg = load_meta(meta)
    replay_fps = float(fps if fps is not None else cfg.get("hz", 30.0))
    return BimanualEpisodeData(
        left_actions=left_actions,
        right_actions=right_actions,
        start=start,
        end=end,
        fps=replay_fps,
        meta=cfg,
    )


def _joint_step_stats(actions: np.ndarray) -> tuple[float, float]:
    joint_steps = np.abs(np.diff(actions[:, :6], axis=0)) if len(actions) > 1 else np.zeros((0, 6), dtype=np.float32)
    grip_steps = np.abs(np.diff(actions[:, 6], axis=0)) if len(actions) > 1 else np.zeros((0,), dtype=np.float32)
    max_joint_step = float(joint_steps.max()) if joint_steps.size else 0.0
    max_grip_step = float(grip_steps.max()) if grip_steps.size else 0.0
    return max_joint_step, max_grip_step


def _print_side_summary(name: str, key: str, actions: np.ndarray) -> None:
    max_joint_step, max_grip_step = _joint_step_stats(actions)
    print(f"{name} replay key: {key}")
    print(f"{name} first target: {np.array2string(actions[0], precision=4)}")
    print(f"{name} last target : {np.array2string(actions[-1], precision=4)}")
    print(f"{name} joint min  : {np.array2string(actions[:, :6].min(axis=0), precision=4)}")
    print(f"{name} joint max  : {np.array2string(actions[:, :6].max(axis=0), precision=4)}")
    print(f"{name} max joint step between dataset frames: {max_joint_step:.4f} rad")
    print(f"{name} max gripper step between dataset frames: {max_grip_step:.4f}")


def print_summary(ep: BimanualEpisodeData, left_key: str, right_key: str) -> None:
    frames = len(ep.left_actions)
    if frames == 0:
        print("Episode is empty.")
        return
    print(f"Dataset frames: {frames}")
    print(f"Source frame range: [{ep.start}, {ep.end})")
    print(f"Replay FPS: {ep.fps:.2f}")
    print(f"Estimated duration: {frames / max(ep.fps, 1e-6):.2f}s")
    _print_side_summary("left", left_key, ep.left_actions)
    _print_side_summary("right", right_key, ep.right_actions)
    if ep.meta:
        print(f"Dataset meta: {ep.meta}")


def clipped_target(current: np.ndarray, target: np.ndarray, max_joint_delta_rad: float | None) -> np.ndarray:
    out = np.asarray(target, dtype=np.float32).copy()
    out[6] = float(np.clip(out[6], 0.0, 1.0))
    if max_joint_delta_rad is None or max_joint_delta_rad <= 0:
        return out
    current = np.asarray(current, dtype=np.float32).reshape(7)
    delta = np.clip(out[:6] - current[:6], -max_joint_delta_rad, max_joint_delta_rad)
    out[:6] = current[:6] + delta
    return out


def _max_joint_error(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(target[:6]) - np.asarray(current[:6]))))


def approach_start(
    left_arm: PiperArm,
    right_arm: PiperArm,
    left_target: np.ndarray,
    right_target: np.ndarray,
    hz: float,
    max_joint_delta_rad: float,
    tolerance_rad: float,
    timeout_s: float,
) -> None:
    t0 = time.monotonic()
    dt = 1.0 / max(hz, 1e-6)
    while True:
        loop_t = time.monotonic()
        left_current = left_arm.read_state().joint_pos
        right_current = right_arm.read_state().joint_pos
        left_err = _max_joint_error(left_current, left_target)
        right_err = _max_joint_error(right_current, right_target)
        if max(left_err, right_err) <= tolerance_rad:
            return
        if loop_t - t0 > timeout_s:
            raise TimeoutError(
                "Approach start timed out. "
                f"left_err={left_err:.4f} rad, right_err={right_err:.4f} rad."
            )

        left_step = clipped_target(left_current, left_target, max_joint_delta_rad)
        right_step = clipped_target(right_current, right_target, max_joint_delta_rad)
        left_arm.send_joint_target(left_step)
        right_arm.send_joint_target(right_step)
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)


def make_arm(name: str, can_name: str, args: argparse.Namespace) -> PiperArm:
    return PiperArm(
        PiperArmConfig(
            name=name,
            can_name=can_name,
            backend=args.backend,
            enable_on_connect=False,
            speed_ratio=args.speed_ratio,
            high_follow=not args.no_high_follow,
        )
    )


def replay(args: argparse.Namespace, ep: BimanualEpisodeData) -> None:
    left_arm = make_arm("left", args.left_can, args)
    right_arm = make_arm("right", args.right_can, args)
    left_arm.connect()
    try:
        right_arm.connect()
        try:
            if args.set_motion_output_role:
                left_arm.set_motion_output_role()
                right_arm.set_motion_output_role()
                time.sleep(0.2)
            if args.enable:
                if not left_arm.enable():
                    raise RuntimeError("EnablePiper() timed out for left arm.")
                if not right_arm.enable():
                    raise RuntimeError("EnablePiper() timed out for right arm.")

            first_left = ep.left_actions[0]
            first_right = ep.right_actions[0]
            left_current = left_arm.read_state().joint_pos
            right_current = right_arm.read_state().joint_pos
            left_start_err = _max_joint_error(left_current, first_left)
            right_start_err = _max_joint_error(right_current, first_right)
            print(f"Current left joint_pos : {np.array2string(left_current, precision=4)}")
            print(f"Current right joint_pos: {np.array2string(right_current, precision=4)}")
            print(f"Distance to first target: left={left_start_err:.4f} rad, right={right_start_err:.4f} rad")
            if max(left_start_err, right_start_err) > args.require_near_rad and not args.approach_start:
                raise RuntimeError(
                    "At least one arm is too far from the first target. "
                    "Move both arms near the recorded start pose by hand, or pass --approach-start."
                )

            if args.approach_start:
                print("Approaching first bimanual target...")
                approach_start(
                    left_arm,
                    right_arm,
                    first_left,
                    first_right,
                    hz=args.approach_hz,
                    max_joint_delta_rad=args.approach_max_joint_delta_rad,
                    tolerance_rad=args.start_tolerance_rad,
                    timeout_s=args.approach_timeout_s,
                )

            print("Replaying bimanual episode...")
            dt = 1.0 / max(ep.fps, 1e-6)
            sent = 0
            t0 = time.monotonic()
            for idx, (raw_left, raw_right) in enumerate(zip(ep.left_actions, ep.right_actions, strict=True)):
                loop_t = time.monotonic()
                left_current = left_arm.read_state().joint_pos
                right_current = right_arm.read_state().joint_pos
                left_target = clipped_target(left_current, raw_left, args.max_joint_delta_rad)
                right_target = clipped_target(right_current, raw_right, args.max_joint_delta_rad)
                left_arm.send_joint_target(left_target)
                right_arm.send_joint_target(right_target)
                sent += 1
                if args.print_every > 0 and (idx == 0 or (idx + 1) % args.print_every == 0):
                    left_err = _max_joint_error(left_current, raw_left)
                    right_err = _max_joint_error(right_current, raw_right)
                    print(
                        f"frame={idx + 1:05d}/{len(ep.left_actions)} "
                        f"left_err_before_send={left_err:.4f} rad "
                        f"right_err_before_send={right_err:.4f} rad"
                    )
                sleep_s = dt - (time.monotonic() - loop_t)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            elapsed = time.monotonic() - t0
            left_final = left_arm.read_state().joint_pos
            right_final = right_arm.read_state().joint_pos
            left_final_err = _max_joint_error(left_final, ep.left_actions[-1])
            right_final_err = _max_joint_error(right_final, ep.right_actions[-1])
            print(f"Replay done: sent={sent}, elapsed={elapsed:.2f}s, actual_fps={sent / max(elapsed, 1e-6):.2f}")
            print(f"Final left joint_pos : {np.array2string(left_final, precision=4)}")
            print(f"Final right joint_pos: {np.array2string(right_final, precision=4)}")
            print(f"Final max joint error: left={left_final_err:.4f} rad, right={right_final_err:.4f} rad")
        finally:
            right_arm.disconnect()
    finally:
        left_arm.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one bimanual PiperX Zarr episode as synchronized joint targets.")
    parser.add_argument("--zarr", "-i", required=True, help="Path to a PiperX bimanual Zarr dataset.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--left-key", default="action_left")
    parser.add_argument("--right-key", default="action_right")
    parser.add_argument("--fps", type=float, default=None, help="Override replay FPS. Defaults to dataset meta hz.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--left-can", default="can1", help="Follower CAN interface for the left arm.")
    parser.add_argument("--right-can", default="can3", help="Follower CAN interface for the right arm.")
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable", action="store_true", help="Call EnablePiper() before replay.")
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08, help="Per-frame joint safety clamp; <=0 disables.")
    parser.add_argument("--require-near-rad", type=float, default=0.35)
    parser.add_argument("--approach-start", action="store_true", help="Move gradually from current poses to first targets.")
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-max-joint-delta-rad", type=float, default=0.04)
    parser.add_argument("--approach-timeout-s", type=float, default=20.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to both arms. Without this, only prints stats.")
    args = parser.parse_args()
    setup_logging()

    if not os.path.exists(args.zarr):
        raise FileNotFoundError(args.zarr)

    ep = load_episode(args.zarr, args.episode, args.left_key, args.right_key, args.fps, args.max_frames)
    print_summary(ep, args.left_key, args.right_key)
    if len(ep.left_actions) == 0:
        return
    if not args.execute:
        print("\nDry-run only. Add --execute to send this synchronized trajectory to the robot.")
        return
    replay(args, ep)


if __name__ == "__main__":
    main()
