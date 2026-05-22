#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.utils.logging import setup_logging


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PiperX ToolKit runtime dependencies.")
    parser.add_argument("--check-cameras", action="store_true", help="Try opening OpenCV camera indices 0,1,2.")
    args = parser.parse_args()
    setup_logging()

    print("PiperX ToolKit doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {os.getcwd()}")

    for module in ("numpy", "piper_sdk", "can", "cv2", "zarr", "PIL", "lerobot", "yaml", "torch"):
        print(f"{module:12s}: {'OK' if has_module(module) else 'missing'}")

    net_dir = "/sys/class/net"
    if os.path.isdir(net_dir):
        interfaces = sorted(os.listdir(net_dir))
        can_ifaces = [name for name in interfaces if name.startswith("can")]
        print(f"Network CAN interfaces: {can_ifaces or 'none found'}")
    else:
        print("Network CAN interfaces: /sys/class/net unavailable on this OS")

    if args.check_cameras:
        if not has_module("cv2"):
            print("Camera check skipped: cv2 missing")
            return
        import cv2

        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            if ok:
                ret, frame = cap.read()
                shape = None if not ret or frame is None else frame.shape
                print(f"Camera {idx}: opened, frame_shape={shape}")
            else:
                print(f"Camera {idx}: not opened")
            cap.release()


if __name__ == "__main__":
    main()
