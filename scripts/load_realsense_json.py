#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAMERAS = {
    "front": {
        "serial": "347622070427",
        "json": "front.json",
    },
    "left": {
        "serial": "260322278131",
        "json": "left.json",
    },
    "right": {
        "serial": "335122272832",
        "json": "right.json",
    },
}

DEFAULT_OPTION_PROFILES = {
    "front": {
        "enable_auto_exposure": 0,
        "exposure": 50,
        "enable_auto_white_balance": 0,
        "white_balance": 4600,
    },
    "left": {
        "enable_auto_exposure": 0,
        "exposure": 8000,
        "enable_auto_white_balance": 0,
    },
    "right": {
        "enable_auto_exposure": 0,
        "exposure": 8000,
        "enable_auto_white_balance": 0,
    },
}


@dataclass(frozen=True)
class CameraTarget:
    role: str
    serial: str
    json_path: Path


@dataclass(frozen=True)
class OptionTarget:
    role: str
    serial: str
    options: dict[str, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def load_rs():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyrealsense2 is required. Install it with: python -m pip install pyrealsense2"
        ) from exc
    return rs


def device_serial(rs, device) -> str:
    return device.get_info(rs.camera_info.serial_number)


def device_name(rs, device) -> str:
    if device.supports(rs.camera_info.name):
        return device.get_info(rs.camera_info.name)
    return "unknown"


def device_firmware(rs, device) -> str:
    if device.supports(rs.camera_info.firmware_version):
        return device.get_info(rs.camera_info.firmware_version)
    return "unknown"


def connected_devices(rs) -> dict[str, object]:
    ctx = rs.context()
    devices = {}
    for device in ctx.query_devices():
        devices[device_serial(rs, device)] = device
    return devices


def serializable_device(rs, device):
    if hasattr(device, "as_serializable_device"):
        return device.as_serializable_device()
    return rs.serializable_device(device)


def print_devices(rs, devices: dict[str, object]) -> None:
    print(f"RealSense devices: {len(devices)}")
    for serial, device in sorted(devices.items()):
        print(
            f"  serial={serial} "
            f"name={device_name(rs, device)} "
            f"firmware={device_firmware(rs, device)}"
        )


def load_json_to_device(rs, devices: dict[str, object], target: CameraTarget) -> None:
    device = devices.get(target.serial)
    if device is None:
        available = ", ".join(sorted(devices)) or "none"
        raise RuntimeError(
            f"{target.role}: RealSense serial {target.serial} not found. Available serials: {available}"
        )
    if not target.json_path.exists():
        raise FileNotFoundError(f"{target.role}: JSON file not found: {target.json_path}")

    text = target.json_path.read_text()
    serializable_device(rs, device).load_json(text)
    print(
        f"{target.role}: loaded {target.json_path} "
        f"into serial={target.serial} name={device_name(rs, device)}"
    )


def dump_device_json(rs, devices: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for serial, device in sorted(devices.items()):
        out = out_dir / f"realsense_{serial}.json"
        text = serializable_device(rs, device).serialize_json()
        out.write_text(text)
        print(f"dumped serial={serial} name={device_name(rs, device)} -> {out}")


def option_enum(rs, name: str):
    if not hasattr(rs.option, name):
        raise ValueError(f"Unknown RealSense option: {name}")
    return getattr(rs.option, name)


def sensor_name(rs, sensor) -> str:
    if sensor.supports(rs.camera_info.name):
        return sensor.get_info(rs.camera_info.name)
    return "unknown"


def set_options_on_device(rs, devices: dict[str, object], target: OptionTarget) -> None:
    device = devices.get(target.serial)
    if device is None:
        available = ", ".join(sorted(devices)) or "none"
        raise RuntimeError(
            f"{target.role}: RealSense serial {target.serial} not found. Available serials: {available}"
        )

    print(f"{target.role}: setting options on serial={target.serial} name={device_name(rs, device)}")
    applied = 0
    for sensor in device.query_sensors():
        name = sensor_name(rs, sensor)
        sensor_applied = 0
        for option_name, value in target.options.items():
            opt = option_enum(rs, option_name)
            if not sensor.supports(opt):
                continue
            sensor.set_option(opt, float(value))
            sensor_applied += 1
            applied += 1
            print(f"  {name}: set {option_name}={value}")
        if sensor_applied == 0:
            print(f"  {name}: no matching options")
    if applied == 0:
        raise RuntimeError(f"{target.role}: none of the requested options were supported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check connected RealSense devices and load Viewer-exported JSON settings by serial."
    )
    parser.add_argument("--check", action="store_true", help="Only list connected RealSense devices.")
    parser.add_argument("--dump-current", default=None, help="Directory to dump current JSON settings for all devices.")
    parser.add_argument(
        "--mode",
        default="options",
        choices=["options", "json"],
        help="Apply safe set_option profiles, or load full Viewer-exported JSON files.",
    )

    parser.add_argument("--front-serial", default=DEFAULT_CAMERAS["front"]["serial"])
    parser.add_argument("--front-json", default=DEFAULT_CAMERAS["front"]["json"])
    parser.add_argument("--left-serial", default=DEFAULT_CAMERAS["left"]["serial"])
    parser.add_argument("--left-json", default=DEFAULT_CAMERAS["left"]["json"])
    parser.add_argument("--right-serial", default=DEFAULT_CAMERAS["right"]["serial"])
    parser.add_argument("--right-json", default=DEFAULT_CAMERAS["right"]["json"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rs = load_rs()
    devices = connected_devices(rs)
    print_devices(rs, devices)

    if args.dump_current:
        dump_device_json(rs, devices, resolve_path(args.dump_current))

    if args.check:
        return

    if args.mode == "options":
        targets = [
            OptionTarget("front", args.front_serial, DEFAULT_OPTION_PROFILES["front"]),
            OptionTarget("left", args.left_serial, DEFAULT_OPTION_PROFILES["left"]),
            OptionTarget("right", args.right_serial, DEFAULT_OPTION_PROFILES["right"]),
        ]
        for target in targets:
            set_options_on_device(rs, devices, target)
        return

    targets = [
        CameraTarget("front", args.front_serial, resolve_path(args.front_json)),
        CameraTarget("left", args.left_serial, resolve_path(args.left_json)),
        CameraTarget("right", args.right_serial, resolve_path(args.right_json)),
    ]
    for target in targets:
        load_json_to_device(rs, devices, target)


if __name__ == "__main__":
    main()
