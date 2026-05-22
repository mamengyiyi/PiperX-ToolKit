from __future__ import annotations

import atexit
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from piperx_toolkit.env.cameras import CameraConfig, CameraManager, default_camera_configs
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.types import ActionMode, Backend, DualArmAction

logger = logging.getLogger(__name__)
VALID_ACTION_MODES = {"absolute_joint", "absolute_eef", "smooth_eef", "delta_eef"}


@dataclass
class SmoothEEFConfig:
    hz: float = 50.0
    linear_speed_m_s: float = 0.15
    angular_speed_rad_s: float = 0.7
    gripper_speed_s: float = 1.0
    min_duration_s: float = 0.08
    max_duration_s: float = 5.0


@dataclass
class DualPiperXEnvConfig:
    left_can: str = "can0"
    right_can: str = "can1"
    backend: Backend = "sdk"
    camera_backend: str = "mock"
    cameras: dict[str, CameraConfig] = field(default_factory=default_camera_configs)
    enable_on_connect: bool = False
    gripper_open_raw: int = 70_000
    gripper_closed_raw: int = 0
    speed_ratio: int = 100
    high_follow: bool = True
    smooth_eef: SmoothEEFConfig = field(default_factory=SmoothEEFConfig)


class DualPiperXEnv:
    """Unified environment for two PiperX arms and three RGB cameras."""

    def __init__(self, config: DualPiperXEnvConfig | None = None, auto_connect: bool = True):
        self.config = config or DualPiperXEnvConfig()
        self.left_arm = PiperArm(self._arm_config("left", self.config.left_can))
        self.right_arm = PiperArm(self._arm_config("right", self.config.right_can))
        self.cameras = CameraManager(self.config.cameras)
        self._connected = False
        self._closed = False
        if auto_connect:
            self.connect()
        atexit.register(self.close)

    def _arm_config(self, name: str, can_name: str) -> PiperArmConfig:
        return PiperArmConfig(
            name=name,
            can_name=can_name,
            backend=self.config.backend,
            enable_on_connect=self.config.enable_on_connect,
            gripper_open_raw=self.config.gripper_open_raw,
            gripper_closed_raw=self.config.gripper_closed_raw,
            speed_ratio=self.config.speed_ratio,
            high_follow=self.config.high_follow,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        self.left_arm.connect()
        self.right_arm.connect()
        self.cameras.connect()
        self._connected = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.cameras.close()
        finally:
            self.left_arm.disconnect()
            self.right_arm.disconnect()
            self._connected = False

    def set_teaching_input_role(self) -> None:
        self.left_arm.set_teaching_input_role()
        self.right_arm.set_teaching_input_role()

    def set_motion_output_role(self) -> None:
        self.left_arm.set_motion_output_role()
        self.right_arm.set_motion_output_role()

    def configure_gripper_teaching_pendants(self, teaching_range: int = 100, max_range_mm: int = 70) -> None:
        self.left_arm.configure_gripper_teaching_pendant(teaching_range, max_range_mm)
        self.right_arm.configure_gripper_teaching_pendant(teaching_range, max_range_mm)

    def reset_after_teaching(self) -> None:
        self.left_arm.reset_after_teaching()
        self.right_arm.reset_after_teaching()

    def guard_can_accept_motion(self) -> None:
        self.left_arm.guard_can_accept_motion()
        self.right_arm.guard_can_accept_motion()

    def get_observation(self, include_arm: bool = True, include_camera: bool = True) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        timestamps = []
        if include_arm:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                state = arm.read_state()
                obs[f"{side}_joint_pos"] = state.joint_pos.copy()
                obs[f"{side}_eef_pos"] = state.eef_pos.copy()
                obs[f"{side}_joint_qvel"] = state.joint_qvel.copy()
                obs[f"{side}_joint_effort"] = state.joint_effort.copy()
                obs[f"{side}_ctrl_mode"] = np.array([-1 if state.ctrl_mode is None else state.ctrl_mode], dtype=np.int32)
                timestamps.append(state.timestamp)
        if include_camera:
            for name, image in self.cameras.read_all().items():
                obs[f"{name}_color"] = image
        obs["timestamp"] = np.array([time.time() if not timestamps else max(timestamps)], dtype=np.float64)
        return obs

    def reset(self) -> dict[str, np.ndarray]:
        return self.get_observation()

    def step(
        self,
        action: DualArmAction,
        action_mode: ActionMode | str = "absolute_joint",
        return_observation: bool = True,
    ) -> dict[str, np.ndarray] | None:
        action_mode = self._normalize_action_mode(action_mode)
        normalized = self._validate_action(action)

        if action_mode == "absolute_joint":
            self._apply_absolute_joint(normalized)
        elif action_mode == "absolute_eef":
            self._apply_absolute_eef(normalized)
        elif action_mode == "smooth_eef":
            self._apply_smooth_eef(normalized)
        elif action_mode == "delta_eef":
            self._apply_delta_eef(normalized)

        if not return_observation:
            return None
        return self.get_observation()

    def step_arm(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        action_mode: ActionMode | str = "absolute_joint",
        return_observation: bool = True,
    ) -> dict[str, np.ndarray] | None:
        return self.step({"left": left, "right": right}, action_mode=action_mode, return_observation=return_observation)

    @staticmethod
    def _normalize_action_mode(action_mode: ActionMode | str) -> ActionMode:
        mode = str(action_mode).strip().lower()
        if mode not in VALID_ACTION_MODES:
            raise ValueError(f"Invalid action_mode={action_mode!r}. Choose from {sorted(VALID_ACTION_MODES)}")
        return mode  # type: ignore[return-value]

    @staticmethod
    def _validate_action(action: DualArmAction | dict) -> dict[str, np.ndarray | None]:
        if not isinstance(action, dict):
            raise TypeError("action must be a dict with keys: left, right")
        missing = {"left", "right"} - set(action.keys())
        if missing:
            raise ValueError(f"action dict missing keys: {sorted(missing)}")
        out: dict[str, np.ndarray | None] = {}
        for side in ("left", "right"):
            value = action[side]
            if value is None:
                out[side] = None
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.shape[0] != 7:
                raise ValueError(f"{side} action must have shape (7,), got {arr.shape}")
            out[side] = arr
        return out

    def _apply_absolute_joint(self, action: dict[str, np.ndarray | None]) -> None:
        if action["left"] is not None:
            self.left_arm.send_joint_target(action["left"])
        if action["right"] is not None:
            self.right_arm.send_joint_target(action["right"])

    def _apply_absolute_eef(self, action: dict[str, np.ndarray | None]) -> None:
        if action["left"] is not None:
            self.left_arm.send_eef_target(action["left"])
        if action["right"] is not None:
            self.right_arm.send_eef_target(action["right"])

    def _apply_delta_eef(self, action: dict[str, np.ndarray | None]) -> None:
        target: dict[str, np.ndarray | None] = {"left": None, "right": None}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            delta = action[side]
            if delta is None:
                continue
            current = arm.read_state().eef_pos
            nxt = current.copy()
            nxt[:6] = current[:6] + delta[:6]
            nxt[6] = float(np.clip(current[6] + delta[6], 0.0, 1.0))
            target[side] = nxt.astype(np.float32)
        self._apply_absolute_eef(target)

    def _apply_smooth_eef(self, action: dict[str, np.ndarray | None]) -> None:
        starts: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            target = action[side]
            if target is None:
                continue
            starts[side] = arm.read_state().eef_pos
            targets[side] = target

        if not targets:
            return

        cfg = self.config.smooth_eef
        duration = cfg.min_duration_s
        for side, target in targets.items():
            start = starts[side]
            linear_dist = float(np.linalg.norm(target[:3] - start[:3]))
            angular_dist = float(np.linalg.norm(target[3:6] - start[3:6]))
            gripper_dist = abs(float(target[6] - start[6]))
            duration = max(
                duration,
                linear_dist / max(cfg.linear_speed_m_s, 1e-6),
                angular_dist / max(cfg.angular_speed_rad_s, 1e-6),
                gripper_dist / max(cfg.gripper_speed_s, 1e-6),
            )
        duration = float(np.clip(duration, cfg.min_duration_s, cfg.max_duration_s))
        steps = max(2, int(round(duration * cfg.hz)))
        dt = 1.0 / max(cfg.hz, 1.0)

        for i in range(1, steps + 1):
            alpha = i / steps
            frame_action: dict[str, np.ndarray | None] = {"left": None, "right": None}
            for side, target in targets.items():
                start = starts[side]
                frame_action[side] = (start + alpha * (target - start)).astype(np.float32)
            self._apply_absolute_eef(frame_action)
            time.sleep(dt)

    def diagnostics_snapshot(self) -> dict[str, object]:
        return {
            "left": self.left_arm.diagnostics_snapshot(),
            "right": self.right_arm.diagnostics_snapshot(),
            "camera_names": sorted(self.config.cameras.keys()),
        }

