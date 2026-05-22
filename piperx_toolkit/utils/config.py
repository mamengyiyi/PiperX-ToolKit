from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from piperx_toolkit.env.cameras import CameraConfig, default_camera_configs
from piperx_toolkit.env.dual_piper_env import DualPiperXEnvConfig


def load_mapping(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required to read YAML config files") from exc
        loaded = yaml.safe_load(p.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        return loaded
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(p.read_text())
    raise ValueError(f"Unsupported config extension: {suffix}")


def env_config_from_mapping(mapping: dict[str, Any], backend: str | None = None, camera_backend: str | None = None) -> DualPiperXEnvConfig:
    arm = mapping.get("arms", {})
    camera_map = mapping.get("cameras", {})
    cam_backend = camera_backend or mapping.get("camera_backend", "mock")
    cameras = default_camera_configs(cam_backend)
    for name, cfg in camera_map.items():
        base = cameras.get(name, CameraConfig(name=name, backend=cam_backend))
        if isinstance(cfg, dict):
            cameras[name] = CameraConfig(
                name=name,
                backend=cfg.get("backend", base.backend),
                device=cfg.get("device", base.device),
                width=int(cfg.get("width", base.width)),
                height=int(cfg.get("height", base.height)),
                fps=int(cfg.get("fps", base.fps)),
            )
    return DualPiperXEnvConfig(
        left_can=arm.get("left_can", mapping.get("left_can", "can0")),
        right_can=arm.get("right_can", mapping.get("right_can", "can1")),
        backend=backend or mapping.get("backend", "sdk"),
        camera_backend=cam_backend,
        cameras=cameras,
        enable_on_connect=bool(mapping.get("enable_on_connect", False)),
        gripper_open_raw=int(mapping.get("gripper_open_raw", 70_000)),
        gripper_closed_raw=int(mapping.get("gripper_closed_raw", 0)),
        speed_ratio=int(mapping.get("speed_ratio", 100)),
        high_follow=bool(mapping.get("high_follow", True)),
    )


def load_env_config(path: str | None, backend: str | None = None, camera_backend: str | None = None) -> DualPiperXEnvConfig:
    return env_config_from_mapping(load_mapping(path), backend=backend, camera_backend=camera_backend)

