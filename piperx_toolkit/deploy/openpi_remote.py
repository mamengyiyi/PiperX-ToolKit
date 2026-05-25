from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np

OpenPIObservationFormat = Literal["aloha", "piperx"]


def _as_hwc_uint8(image: np.ndarray, resize: int = 224) -> np.ndarray:
    from openpi_client import image_tools

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with shape HWC or CHW, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {arr.shape}")
    arr = image_tools.convert_to_uint8(arr)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(arr, resize, resize))


def _black_hwc(resize: int = 224) -> np.ndarray:
    return np.zeros((resize, resize, 3), dtype=np.uint8)


def _aloha_images(images: dict[str, np.ndarray], resize: int = 224) -> dict[str, np.ndarray]:
    front = _as_hwc_uint8(images["front"], resize=resize) if "front" in images else _black_hwc(resize)
    left_wrist = (
        _as_hwc_uint8(images["left_wrist"], resize=resize) if "left_wrist" in images else _black_hwc(resize)
    )
    right_wrist = (
        _as_hwc_uint8(images["right_wrist"], resize=resize) if "right_wrist" in images else _black_hwc(resize)
    )
    black = _black_hwc(resize)
    return {
        "cam_high": front.transpose(2, 0, 1),
        "cam_low": black.transpose(2, 0, 1),
        "cam_left_wrist": left_wrist.transpose(2, 0, 1),
        "cam_right_wrist": right_wrist.transpose(2, 0, 1),
    }


def _piperx_observation_images(images: dict[str, np.ndarray], resize: int = 224) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("front", "left_wrist", "right_wrist"):
        out[f"observation.images.{name}"] = (
            _as_hwc_uint8(images[name], resize=resize) if name in images else _black_hwc(resize)
        )
    return out


class OpenPIRemotePolicy:
    """Thin wrapper around openpi-client's websocket policy client."""

    def __init__(
        self,
        host: str,
        port: int = 8000,
        api_key: str | None = None,
        prompt: str = "",
        observation_format: OpenPIObservationFormat = "aloha",
        resize: int = 224,
        action_dim: int | None = None,
    ):
        from openpi_client import websocket_client_policy

        self.host = host
        self.port = port
        self.prompt = prompt
        self.observation_format = observation_format
        self.resize = resize
        self.action_dim = action_dim
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self.action_queue: deque[np.ndarray] = deque()

    def get_server_metadata(self) -> dict[str, Any]:
        if hasattr(self.client, "get_server_metadata"):
            return self.client.get_server_metadata()
        return {}

    def reset(self) -> None:
        self.action_queue.clear()
        if hasattr(self.client, "reset"):
            self.client.reset()

    def make_observation(self, images: dict[str, np.ndarray], state: np.ndarray) -> dict[str, Any]:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.observation_format == "aloha":
            return {
                "state": state,
                "images": _aloha_images(images, resize=self.resize),
                "prompt": self.prompt,
            }
        if self.observation_format == "piperx":
            obs: dict[str, Any] = {
                "observation.state": state,
                "prompt": self.prompt,
            }
            obs.update(_piperx_observation_images(images, resize=self.resize))
            return obs
        raise ValueError(f"Unsupported OpenPI observation format: {self.observation_format}")

    def _split_actions(self, value: Any) -> list[np.ndarray]:
        if isinstance(value, dict):
            for key in ("actions", "action", "prediction"):
                if key in value:
                    value = value[key]
                    break
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported OpenPI action shape: {arr.shape}")
        if self.action_dim is not None:
            if arr.shape[-1] < self.action_dim:
                raise ValueError(f"OpenPI action dim {arr.shape[-1]} is smaller than requested {self.action_dim}")
            arr = arr[:, : self.action_dim]
        return [row.astype(np.float32).copy() for row in arr]

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        if self.action_queue:
            return self.action_queue.popleft()
        response = self.client.infer(self.make_observation(images, state))
        actions = self._split_actions(response)
        self.action_queue.extend(actions[1:])
        return actions[0]
