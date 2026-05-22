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
class EpisodeData:
    actions: np.ndarray
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


def load_episode(path: str, episode_index: int, key: str, fps: float | None, max_frames: int | None) -> EpisodeData:
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
    if key not in data:
        available = ", ".join(sorted(k for k in data.keys() if not k.startswith("rgb_")))
        raise KeyError(f"Missing action key '{key}'. Available numeric keys: {available}")

    start, end = ranges[episode_index]
    if max_frames is not None:
        end = min(end, start + max(0, max_frames))
    actions = np.asarray(data[key][start:end], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected {key} shape (N, 7), got {actions.shape}")
    cfg = load_meta(meta)
    replay_fps = float(fps if fps is not None else cfg.get("hz", 30.0))
    return EpisodeData(actions=actions, start=start, end=end, fps=replay_fps, meta=cfg)


def print_summary(ep: EpisodeData, key: str) -> None:
    actions = ep.actions
    if len(actions) == 0:
        print("Episode is empty.")
        return
    joint_steps = np.abs(np.diff(actions[:, :6], axis=0)) if len(actions) > 1 else np.zeros((0, 6), dtype=np.float32)
    grip_steps = np.abs(np.diff(actions[:, 6], axis=0)) if len(actions) > 1 else np.zeros((0,), dtype=np.float32)
    print(f"Dataset frames: {len(actions)}")
    print(f"Source frame range: [{ep.start}, {ep.end})")
    print(f"Replay key: {key}")
    print(f"Replay FPS: {ep.fps:.2f}")
    print(f"Estimated duration: {len(actions) / max(ep.fps, 1e-6):.2f}s")
    print(f"First target: {np.array2string(actions[0], precision=4)}")
    print(f"Last target : {np.array2string(actions[-1], precision=4)}")
    print(f"Joint min  : {np.array2string(actions[:, :6].min(axis=0), precision=4)}")
    print(f"Joint max  : {np.array2string(actions[:, :6].max(axis=0), precision=4)}")
    print(f"Max joint step between dataset frames: {float(joint_steps.max()) if joint_steps.size else 0.0:.4f} rad")
    print(f"Max gripper step between dataset frames: {float(grip_steps.max()) if grip_steps.size else 0.0:.4f}")
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


def approach_start(
    arm: PiperArm,
    target: np.ndarray,
    hz: float,
    max_joint_delta_rad: float,
    tolerance_rad: float,
    timeout_s: float,
) -> None:
    t0 = time.monotonic()
    dt = 1.0 / hz
    while True:
        loop_t = time.monotonic()
        current = arm.read_state().joint_pos
        joint_err = target[:6] - current[:6]
        if float(np.max(np.abs(joint_err))) <= tolerance_rad:
            return
        if loop_t - t0 > timeout_s:
            raise TimeoutError(
                f"Approach start timed out. Max joint error is {float(np.max(np.abs(joint_err))):.4f} rad."
            )
        target_step = clipped_target(current, target, max_joint_delta_rad)
        arm.send_joint_target(target_step)
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)


def replay(args: argparse.Namespace, ep: EpisodeData) -> None:
    arm = PiperArm(
        PiperArmConfig(
            name=args.side,
            can_name=args.can,
            backend=args.backend,
            enable_on_connect=False,
            speed_ratio=args.speed_ratio,
            high_follow=not args.no_high_follow,
        )
    )
    arm.connect()
    try:
        if args.set_motion_output_role:
            arm.set_motion_output_role()
            time.sleep(0.2)
        if args.enable:
            if not arm.enable():
                raise RuntimeError("EnablePiper() timed out. Check arm power, emergency stop, and CAN link.")

        first = ep.actions[0]
        current = arm.read_state().joint_pos
        start_err = float(np.max(np.abs(first[:6] - current[:6])))
        print(f"Current joint_pos: {np.array2string(current, precision=4)}")
        print(f"Distance to first target: {start_err:.4f} rad")
        if start_err > args.require_near_rad and not args.approach_start:
            raise RuntimeError(
                f"Current pose is {start_err:.4f} rad away from the first target. "
                "Move the arm near the recorded start pose by hand, or pass --approach-start."
            )

        if args.approach_start:
            print("Approaching first target...")
            approach_start(
                arm,
                first,
                hz=args.approach_hz,
                max_joint_delta_rad=args.approach_max_joint_delta_rad,
                tolerance_rad=args.start_tolerance_rad,
                timeout_s=args.approach_timeout_s,
            )

        print("Replaying episode...")
        dt = 1.0 / ep.fps
        sent = 0
        t0 = time.monotonic()
        for idx, raw_target in enumerate(ep.actions):
            loop_t = time.monotonic()
            current = arm.read_state().joint_pos
            target = clipped_target(current, raw_target, args.max_joint_delta_rad)
            arm.send_joint_target(target)
            sent += 1
            if args.print_every > 0 and (idx == 0 or (idx + 1) % args.print_every == 0):
                err = float(np.max(np.abs(raw_target[:6] - current[:6])))
                print(f"frame={idx + 1:05d}/{len(ep.actions)} target_err_before_send={err:.4f} rad")
            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)

        elapsed = time.monotonic() - t0
        final = arm.read_state().joint_pos
        final_err = float(np.max(np.abs(ep.actions[-1, :6] - final[:6])))
        print(f"Replay done: sent={sent}, elapsed={elapsed:.2f}s, actual_fps={sent / max(elapsed, 1e-6):.2f}")
        print(f"Final joint_pos: {np.array2string(final, precision=4)}")
        print(f"Final max joint error vs last target: {final_err:.4f} rad")
    finally:
        arm.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one single-arm PiperX Zarr episode as absolute joint targets.")
    parser.add_argument("--zarr", "-i", required=True, help="Path to a PiperX single-arm Zarr dataset.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--key", default="action", help="Dataset key to replay, usually action or joint_pos.")
    parser.add_argument("--fps", type=float, default=None, help="Override replay FPS. Defaults to dataset meta hz.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable", action="store_true", help="Call EnablePiper() before replay.")
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08, help="Per-frame joint safety clamp; <=0 disables.")
    parser.add_argument("--require-near-rad", type=float, default=0.35)
    parser.add_argument("--approach-start", action="store_true", help="Move gradually from current pose to the first target before replay.")
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-max-joint-delta-rad", type=float, default=0.04)
    parser.add_argument("--approach-timeout-s", type=float, default=20.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the arm. Without this, only prints stats.")
    args = parser.parse_args()
    setup_logging()

    if not os.path.exists(args.zarr):
        raise FileNotFoundError(args.zarr)

    ep = load_episode(args.zarr, args.episode, args.key, args.fps, args.max_frames)
    print_summary(ep, args.key)
    if len(ep.actions) == 0:
        return
    if not args.execute:
        print("\nDry-run only. Add --execute to send this trajectory to the robot.")
        return
    replay(args, ep)


if __name__ == "__main__":
    main()
