#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_can_interfaces() -> list[str]:
    net_dir = Path("/sys/class/net")
    if not net_dir.exists():
        return []
    return sorted(name for name in os.listdir(net_dir) if name.startswith("can"))


def fmt_vec(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=4, suppress_small=True)


def print_can_state(can_name: str, label: str) -> None:
    result = subprocess.run(
        ["ip", "-details", "link", "show", can_name],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"{can_name}: {label}: ip link failed: {result.stderr.strip()}")
        return
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    summary = " | ".join(lines[:3])
    print(f"{can_name}: {label}: {summary}")


def probe_one(can_name: str, samples: int, interval_s: float, can_auto_init: bool) -> None:
    print_can_state(can_name, "before")
    arm = PiperArm(
        PiperArmConfig(
            name=can_name,
            can_name=can_name,
            backend="sdk",
            enable_on_connect=False,
            can_auto_init=can_auto_init,
        )
    )
    try:
        arm.connect()
        first = arm.read_state()
        last = first
        max_delta = 0.0
        for _ in range(max(1, samples - 1)):
            time.sleep(interval_s)
            state = arm.read_state()
            max_delta = max(max_delta, float(np.max(np.abs(state.joint_pos - last.joint_pos))))
            last = state
        print(
            f"{can_name}: OK ctrl_mode={last.ctrl_mode} "
            f"joint={fmt_vec(last.joint_pos)} eef={fmt_vec(last.eef_pos)} "
            f"max_sample_delta={max_delta:.5f}"
        )
    except Exception as exc:
        print(f"{can_name}: ERROR {exc!r}")
    finally:
        arm.disconnect()
        print_can_state(can_name, "after")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Piper arms on CAN interfaces. Move one physical arm while this runs; "
            "the interface with changing joint/max_sample_delta is that arm."
        )
    )
    parser.add_argument("--interfaces", default="", help="Comma-separated CAN interfaces. Default: all /sys/class/net/can*.")
    parser.add_argument("--samples", type=int, default=5, help="Samples per interface.")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between samples.")
    parser.add_argument(
        "--no-can-auto-init",
        action="store_true",
        help="Do not let piper_sdk initialize CAN. Only use this for SDK-level diagnostics.",
    )
    args = parser.parse_args()

    interfaces = split_csv(args.interfaces) if args.interfaces else default_can_interfaces()
    if not interfaces:
        raise SystemExit("No CAN interfaces found. Try --interfaces can0,can1,can2,can3.")

    for can_name in interfaces:
        probe_one(can_name, samples=args.samples, interval_s=args.interval, can_auto_init=not args.no_can_auto_init)


if __name__ == "__main__":
    main()
