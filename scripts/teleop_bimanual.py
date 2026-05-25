#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.piper_arm import PiperArmConfig
from piperx_toolkit.teleop.leader_follower import (
    BimanualLeaderFollowerTeleop,
    JointMapping,
    LeaderFollowerPair,
)
from piperx_toolkit.utils.logging import setup_logging


def parse_vec7(text: str, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"{name} must contain 7 comma-separated values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def format_vec(arr: np.ndarray) -> str:
    return np.array2string(np.asarray(arr, dtype=np.float32), precision=4, suppress_small=True)


def make_pair(
    side: str,
    leader_can: str,
    follower_can: str,
    signs: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> LeaderFollowerPair:
    mapping = JointMapping(signs=signs, offsets=offsets)
    leader_config = PiperArmConfig(
        name=f"{side}_leader",
        can_name=leader_can,
        backend=args.leader_backend,
        enable_on_connect=False,
    )
    follower_config = PiperArmConfig(
        name=f"{side}_follower",
        can_name=follower_can,
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


def make_teleop(args: argparse.Namespace) -> BimanualLeaderFollowerTeleop:
    left_pair = make_pair(
        "left",
        args.left_leader_can,
        args.left_follower_can,
        parse_vec7(args.left_joint_signs, "--left-joint-signs"),
        parse_vec7(args.left_joint_offsets, "--left-joint-offsets"),
        args,
    )
    right_pair = make_pair(
        "right",
        args.right_leader_can,
        args.right_follower_can,
        parse_vec7(args.right_joint_signs, "--right-joint-signs"),
        parse_vec7(args.right_joint_offsets, "--right-joint-offsets"),
        args,
    )
    return BimanualLeaderFollowerTeleop(left_pair=left_pair, right_pair=right_pair)


def print_results(step: int, results: dict[str, object]) -> None:
    chunks: list[str] = [f"step={step:06d}"]
    for side, result in results.items():
        chunks.append(
            f"{side}:err={result.max_abs_error_rad:.4f} "
            f"leader={format_vec(result.leader_joint_pos)} "
            f"follower={format_vec(result.follower_joint_pos)} "
            f"command={format_vec(result.command_joint_pos)}"
        )
    print(" | ".join(chunks))


def run(args: argparse.Namespace) -> None:
    teleop = make_teleop(args)
    execute = bool(args.execute)
    if not execute:
        print("DRY-RUN: no follower commands will be sent. Add --execute to control the follower arms.")

    teleop.start(
        set_motion_output_role=args.set_motion_output_role,
        enable_followers=execute and args.enable_followers,
        startup_sleep_s=args.startup_sleep_s,
    )
    try:
        initial = teleop.step(execute=False)
        teleop.left_pair.previous_command = None
        teleop.right_pair.previous_command = None
        print("Initial leader/follower snapshot:")
        print_results(0, initial)

        initial_max_err = max(result.max_abs_error_rad for result in initial.values())
        if execute and initial_max_err > args.require_near_rad and not args.approach_start:
            raise RuntimeError(
                f"At least one follower is {initial_max_err:.4f} rad away from the mapped leader target. "
                "Move all arms to similar poses by hand, or pass --approach-start for a slow automatic approach."
            )

        if execute and args.approach_start:
            print("Approaching mapped leader poses before live teleoperation...")
            results = teleop.approach_to_leaders(
                hz=args.approach_hz,
                tolerance_rad=args.start_tolerance_rad,
                timeout_s=args.approach_timeout_s,
                execute=True,
            )
            print_results(0, results)

        print("Bimanual teleoperation loop started. Press Ctrl+C to stop.")
        dt = 1.0 / max(args.hz, 1e-6)
        start_t = time.monotonic()
        step = 0
        while True:
            loop_t = time.monotonic()
            if args.duration > 0 and loop_t - start_t >= args.duration:
                break
            if args.max_steps is not None and step >= args.max_steps:
                break

            results = teleop.step(execute=execute)
            step += 1
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print_results(step, results)

            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)

        elapsed = time.monotonic() - start_t
        print(f"Bimanual teleoperation stopped: steps={step}, elapsed={elapsed:.2f}s, rate={step / max(elapsed, 1e-6):.2f} Hz")
    finally:
        teleop.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bimanual PiperX leader-follower joint-space teleoperation.")
    parser.add_argument("--left-leader-can", default="can0")
    parser.add_argument("--left-follower-can", default="can1")
    parser.add_argument("--right-leader-can", default="can2")
    parser.add_argument("--right-follower-can", default="can3")
    parser.add_argument("--leader-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--follower-backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Run seconds; <=0 means until Ctrl+C or --max-steps.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-sleep-s", type=float, default=0.2)
    parser.add_argument("--enable-followers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08, help="Per-cycle joint command clamp; <=0 disables.")
    parser.add_argument("--lowpass-alpha", type=float, default=0.8, help="1.0 disables smoothing; lower is smoother.")
    parser.add_argument("--left-joint-signs", default="1,1,1,1,1,1,1")
    parser.add_argument("--left-joint-offsets", default="0,0,0,0,0,0,0")
    parser.add_argument("--right-joint-signs", default="1,1,1,1,1,1,1")
    parser.add_argument("--right-joint-offsets", default="0,0,0,0,0,0,0")
    parser.add_argument("--require-near-rad", type=float, default=0.35)
    parser.add_argument("--approach-start", action="store_true", help="Slowly move followers to mapped leader poses first.")
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-timeout-s", type=float, default=20.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the follower arms.")
    args = parser.parse_args()
    setup_logging()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
