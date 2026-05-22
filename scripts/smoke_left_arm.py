#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.cameras import CameraConfig, CameraManager
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one real PiperX arm plus optional front camera.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--set-motion-output-role", action="store_true")
    parser.add_argument("--set-teaching-input-role", action="store_true")
    parser.add_argument("--camera-backend", default="mock", choices=["mock", "opencv"])
    parser.add_argument("--camera-device", default="2", help="OpenCV index or /dev/video* path for front camera.")
    parser.add_argument("--no-camera", action="store_true", help="Only read the arm; skip camera initialization.")
    parser.add_argument("--camera-fail-soft", action="store_true", help="Keep reading the arm if the camera read fails.")
    parser.add_argument("--camera-read-retries", type=int, default=3)
    parser.add_argument("--camera-warmup-s", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    setup_logging()

    camera_device: int | str
    camera_device = int(args.camera_device) if args.camera_device.isdigit() else args.camera_device
    arm = PiperArm(PiperArmConfig(name="left", can_name=args.can, backend=args.backend))
    cameras = None
    if not args.no_camera:
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
    if args.set_motion_output_role:
        arm.set_motion_output_role()
        time.sleep(0.2)
    if args.set_teaching_input_role:
        arm.set_teaching_input_role()
        time.sleep(0.2)
    if cameras is not None:
        cameras.connect()
        time.sleep(max(0.0, args.camera_warmup_s))
    count = 0
    camera_failures = 0
    t0 = time.time()
    try:
        last_state = None
        last_image = None
        first_joint = None
        first_eef = None
        max_joint_delta = None
        max_eef_delta = None
        while time.time() - t0 < args.duration:
            loop_t = time.monotonic()
            last_state = arm.read_state()
            if first_joint is None:
                first_joint = last_state.joint_pos.copy()
                first_eef = last_state.eef_pos.copy()
                max_joint_delta = np.zeros_like(first_joint)
                max_eef_delta = np.zeros_like(first_eef)
            else:
                max_joint_delta = np.maximum(max_joint_delta, np.abs(last_state.joint_pos - first_joint))
                max_eef_delta = np.maximum(max_eef_delta, np.abs(last_state.eef_pos - first_eef))
            if cameras is not None:
                for _ in range(max(1, args.camera_read_retries)):
                    try:
                        last_image = cameras.read_all()["front"]
                        break
                    except RuntimeError as exc:
                        camera_failures += 1
                        last_error = exc
                        time.sleep(0.02)
                else:
                    if not args.camera_fail_soft:
                        raise last_error
            count += 1
            if args.hz > 0:
                sleep_s = (1.0 / args.hz) - (time.monotonic() - loop_t)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        elapsed = time.time() - t0
        print(f"Read {count} samples in {elapsed:.2f}s ({count / max(elapsed, 1e-6):.1f} Hz)")
        if last_state is not None:
            if first_joint is not None:
                print("first_joint_pos:", first_joint)
            print("left_joint_pos:", last_state.joint_pos)
            if max_joint_delta is not None:
                print("max_abs_joint_delta:", max_joint_delta)
            if first_eef is not None:
                print("first_eef_pos:", first_eef)
            print("left_eef_pos:", last_state.eef_pos)
            if max_eef_delta is not None:
                print("max_abs_eef_delta:", max_eef_delta)
            print("left_ctrl_mode:", last_state.ctrl_mode)
        if last_image is not None:
            print("front_color:", last_image.shape, last_image.dtype)
        if camera_failures:
            print(f"camera_failures: {camera_failures}")
    finally:
        if cameras is not None:
            cameras.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
