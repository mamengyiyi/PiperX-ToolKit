from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

CameraBackend = Literal["mock", "opencv"]


@dataclass
class CameraConfig:
    name: str
    backend: CameraBackend = "mock"
    device: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30


class BaseCamera:
    def connect(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockCamera(BaseCamera):
    def __init__(self, config: CameraConfig):
        self.config = config
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def read(self) -> np.ndarray:
        h, w = self.config.height, self.config.width
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 0] = x[None, :]
        frame[..., 1] = y
        frame[..., 2] = int(time.time() * 10) % 255
        return frame

    def close(self) -> None:
        self.connected = False


class OpenCVCamera(BaseCamera):
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None

    def connect(self) -> None:
        import cv2

        self.cap = cv2.VideoCapture(self.config.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.config.name}: {self.config.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError(f"Camera {self.config.name} is not connected")
        import cv2

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not read camera {self.config.name}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.config.width or frame_rgb.shape[0] != self.config.height:
            frame_rgb = cv2.resize(frame_rgb, (self.config.width, self.config.height))
        return frame_rgb

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CameraManager:
    def __init__(self, configs: dict[str, CameraConfig]):
        self.configs = configs
        self.cameras: dict[str, BaseCamera] = {}

    def connect(self) -> None:
        for name, cfg in self.configs.items():
            camera: BaseCamera
            if cfg.backend == "mock":
                camera = MockCamera(cfg)
            elif cfg.backend == "opencv":
                camera = OpenCVCamera(cfg)
            else:
                raise ValueError(f"Unsupported camera backend for {name}: {cfg.backend}")
            camera.connect()
            self.cameras[name] = camera

    def read_all(self) -> dict[str, np.ndarray]:
        return {name: camera.read() for name, camera in self.cameras.items()}

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()


def default_camera_configs(backend: CameraBackend = "mock") -> dict[str, CameraConfig]:
    return {
        "front": CameraConfig(name="front", backend=backend, device=0),
        "left_wrist": CameraConfig(name="left_wrist", backend=backend, device=1),
        "right_wrist": CameraConfig(name="right_wrist", backend=backend, device=2),
    }

