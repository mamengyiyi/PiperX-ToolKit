#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_CAMERAS = {
    "front": {
        "serial": "347622070427",
        "json": "front.json",
    },
    "left": {
        "serial": "260322279642",
        "json": "left.json",
    },
    "right": {
        "serial": "260322278131",
        "json": "right.json",
    },
}


WIDTH = 640
HEIGHT = 480
FPS = 30


def list_devices():
    ctx = rs.context()
    devices = ctx.query_devices()

    print(f"RealSense devices found: {len(devices)}")

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)

        print(f"\n{name}")
        print(f"  serial: {serial}")
        print(f"  firmware: {firmware}")

        for sensor in dev.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"  sensor: {sensor_name}")

            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue

                video_profile = profile.as_video_stream_profile()
                print(
                    f"    color: "
                    f"{video_profile.width()}x{video_profile.height()} "
                    f"{profile.format()} "
                    f"{profile.fps()}fps"
                )


def capture_rgb(camera_name, serial, out_dir, warmup=30):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print(f"{camera_name}: starting serial={serial}, {WIDTH}x{HEIGHT}@{FPS}")
    pipeline.start(config)

    try:
        color_frame = None

        for _ in range(max(1, warmup)):
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()

        if color_frame is None:
            raise RuntimeError(f"{camera_name}: no color frame received")

        image = np.asanyarray(color_frame.get_data())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = out_dir / f"{camera_name}_rgb_{timestamp}.png"

        ok = cv2.imwrite(str(save_path), image)
        if not ok:
            raise RuntimeError(f"{camera_name}: failed to save image: {save_path}")

        print(f"{camera_name}: saved {save_path}")

    finally:
        pipeline.stop()
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        choices=["front", "left", "right", "all"],
        default="all",
        help="camera to capture",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="number of warmup frames before saving",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list connected RealSense devices and color profiles",
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    script_dir = Path(__file__).resolve().parent

    if args.camera == "all":
        cameras = DEFAULT_CAMERAS.items()
    else:
        cameras = [(args.camera, DEFAULT_CAMERAS[args.camera])]

    for camera_name, camera_cfg in cameras:
        capture_rgb(
            camera_name=camera_name,
            serial=camera_cfg["serial"],
            out_dir=script_dir,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
