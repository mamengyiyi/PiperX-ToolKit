#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.cameras import CameraConfig, CameraManager
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one real PiperX arm plus optional front camera.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--camera-backend", default="mock", choices=["mock", "opencv"])
    parser.add_argument("--camera-device", default="2", help="OpenCV index or /dev/video* path for front camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    setup_logging()

    camera_device: int | str
    camera_device = int(args.camera_device) if args.camera_device.isdigit() else args.camera_device
    arm = PiperArm(PiperArmConfig(name="left", can_name=args.can, backend=args.backend))
    cameras = CameraManager(
        {
            "front": CameraConfig(
                name="front",
                backend=args.camera_backend,
                device=camera_device,
                width=args.width,
                height=args.height,
            )
        }
    )
    arm.connect()
    cameras.connect()
    count = 0
    t0 = time.time()
    try:
        last_state = None
        last_image = None
        while time.time() - t0 < args.duration:
            last_state = arm.read_state()
            last_image = cameras.read_all()["front"]
            count += 1
        elapsed = time.time() - t0
        print(f"Read {count} samples in {elapsed:.2f}s ({count / max(elapsed, 1e-6):.1f} Hz)")
        if last_state is not None:
            print("left_joint_pos:", last_state.joint_pos)
            print("left_eef_pos:", last_state.eef_pos)
            print("left_ctrl_mode:", last_state.ctrl_mode)
        if last_image is not None:
            print("front_color:", last_image.shape, last_image.dtype)
    finally:
        cameras.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
