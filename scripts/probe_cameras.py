#!/usr/bin/env python3
from __future__ import annotations

import argparse


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenCV-readable camera devices.")
    parser.add_argument("--indices", default="0,1,2,3,4,5", help="Comma-separated OpenCV camera indices.")
    parser.add_argument("--devices", default="", help="Comma-separated /dev/video* or /dev/v4l/by-id/* paths.")
    args = parser.parse_args()

    import cv2

    devices: list[int | str] = []
    devices.extend(int(item) for item in split_csv(args.indices))
    devices.extend(split_csv(args.devices))

    for device in devices:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            print(f"{device}: not opened")
            continue
        ok, frame = cap.read()
        shape = None if not ok or frame is None else frame.shape
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{device}: opened, frame_shape={shape}, reported=({width:.0f}x{height:.0f}@{fps:.1f})")
        cap.release()


if __name__ == "__main__":
    main()
