from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from piperx_toolkit.env import units
from piperx_toolkit.env.mock_sdk import MockPiperInterface
from piperx_toolkit.types import Backend

logger = logging.getLogger(__name__)

PIPER_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
TEACHING_ROLE_CODE = 0xFA
FOLLOWER_ROLE_CODE = 0xFC
CTRL_MODE_TEACH = 0x02
CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06


@dataclass
class PiperArmConfig:
    name: str
    can_name: str
    backend: Backend = "sdk"
    judge_flag: bool = False
    can_auto_init: bool = True
    log_level: str = "WARNING"
    startup_sleep_s: float = 0.1
    enable_on_connect: bool = False
    enable_timeout_s: float = 3.0
    speed_ratio: int = 100
    high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    gripper_open_raw: int = 70_000
    gripper_closed_raw: int = 0
    gripper_effort: int = 1000
    gripper_status_code: int = 0x01


@dataclass
class PiperArmState:
    joint_pos: np.ndarray
    eef_pos: np.ndarray
    joint_qvel: np.ndarray
    joint_effort: np.ndarray
    timestamp: float
    ctrl_mode: int | None = None


def _safe_call(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(obj, method_name)
    try:
        return method(*args, **kwargs)
    except TypeError:
        if kwargs:
            return method(*args)
        raise


def _to_plain(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_plain(v, depth + 1)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


def _child(obj: Any, *names: str) -> Any | None:
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return None


def _value(obj: Any, candidates: tuple[str, ...], default: float = 0.0) -> float:
    if obj is None:
        return float(default)
    for name in candidates:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except (TypeError, ValueError):
                continue
    return float(default)


def _enum_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_log_level(level_name: str) -> Any | None:
    try:
        from piper_sdk import LogLevel  # type: ignore
    except ModuleNotFoundError:
        return None
    return getattr(LogLevel, level_name.upper(), None)


class PiperArm:
    """Thin adapter around one Piper SDK interface."""

    def __init__(self, config: PiperArmConfig):
        self.config = config
        self.arm: Any | None = None
        self.connected = False
        self._last_mode_refresh_t = 0.0
        self._last_state: PiperArmState | None = None

    def connect(self, start_thread: bool = True) -> None:
        if self.connected:
            return
        if self.config.backend == "mock":
            interface_cls = MockPiperInterface
            kwargs: dict[str, Any] = {"can_name": self.config.can_name}
        else:
            try:
                from piper_sdk import C_PiperInterface_V2  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Could not import piper_sdk. Install piper_sdk on the robot computer "
                    "or use backend='mock' for offline tests."
                ) from exc
            interface_cls = C_PiperInterface_V2
            kwargs = {
                "can_name": self.config.can_name,
                "judge_flag": self.config.judge_flag,
                "can_auto_init": self.config.can_auto_init,
            }
            log_level = _parse_log_level(self.config.log_level)
            if log_level is not None:
                kwargs["logger_level"] = log_level

        self.arm = interface_cls(**kwargs)
        _safe_call(self.arm, "ConnectPort", start_thread=start_thread)
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)
        self.connected = True
        if self.config.enable_on_connect:
            self.enable()

    def disconnect(self) -> None:
        if self.arm is None:
            self.connected = False
            return
        try:
            self.arm.DisconnectPort()
        finally:
            self.connected = False

    def ensure_connected(self) -> Any:
        if self.arm is None or not self.connected:
            raise RuntimeError(f"{self.config.name} arm is not connected")
        return self.arm

    def enable(self) -> bool:
        arm = self.ensure_connected()
        deadline = time.monotonic() + max(0.0, self.config.enable_timeout_s)
        while time.monotonic() <= deadline:
            if bool(arm.EnablePiper()):
                return True
            time.sleep(0.05)
        return False

    def disable(self) -> None:
        arm = self.ensure_connected()
        if hasattr(arm, "DisableArm"):
            arm.DisableArm(7)

    def set_teaching_input_role(self) -> None:
        arm = self.ensure_connected()
        arm.MasterSlaveConfig(TEACHING_ROLE_CODE, 0x00, 0x00, 0x00)

    def set_motion_output_role(self) -> None:
        arm = self.ensure_connected()
        arm.MasterSlaveConfig(FOLLOWER_ROLE_CODE, 0x00, 0x00, 0x00)

    def reset_after_teaching(self) -> None:
        arm = self.ensure_connected()
        arm.MotionCtrl_1(0x02, 0x00, 0x00)

    def configure_gripper_teaching_pendant(self, teaching_range: int = 100, max_range_mm: int = 70) -> None:
        arm = self.ensure_connected()
        if hasattr(arm, "GripperTeachingPendantParamConfig"):
            arm.GripperTeachingPendantParamConfig(teaching_range, max_range_mm, 1)
        if hasattr(arm, "ArmParamEnquiryAndConfig"):
            arm.ArmParamEnquiryAndConfig(4)

    def read_ctrl_mode(self) -> int | None:
        arm = self.ensure_connected()
        msg = arm.GetArmStatus()
        status = _child(msg, "arm_status", "status")
        return _enum_to_int(_child(status, "ctrl_mode") if status is not None else None)

    def guard_can_accept_motion(self) -> None:
        mode = self.read_ctrl_mode()
        if mode in {CTRL_MODE_TEACH, CTRL_MODE_LINKAGE_TEACH_INPUT}:
            raise RuntimeError(
                f"{self.config.name} arm appears to be in teaching/master input mode "
                f"(ctrl_mode=0x{mode:02X}). Switch it to motion/follower role and power-cycle "
                "before policy deployment."
            )

    def _send_motion_mode(self, motion_type: str) -> None:
        arm = self.ensure_connected()
        move_mode = 0x01 if motion_type == "joint" else 0x00
        mit_mode = 0xAD if self.config.high_follow else 0x00
        arm.MotionCtrl_2(0x01, move_mode, self.config.speed_ratio, mit_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_motion_mode_if_needed(self, motion_type: str) -> None:
        interval = self.config.mode_refresh_interval_s
        if interval <= 0:
            return
        if time.monotonic() - self._last_mode_refresh_t >= interval:
            self._send_motion_mode(motion_type)

    def send_joint_target(self, target: np.ndarray) -> None:
        arm = self.ensure_connected()
        arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 7:
            raise ValueError(f"joint target must have shape (7,), got {arr.shape}")
        self._refresh_motion_mode_if_needed("joint")
        joint_raw = units.rad_to_joint_raw(arr[:6])
        grip_raw = units.gripper_norm_to_raw(
            float(arr[6]),
            open_raw=self.config.gripper_open_raw,
            closed_raw=self.config.gripper_closed_raw,
        )
        arm.JointCtrl(*joint_raw)
        arm.GripperCtrl(abs(grip_raw), self.config.gripper_effort, self.config.gripper_status_code, 0x00)

    def send_eef_target(self, target: np.ndarray) -> None:
        arm = self.ensure_connected()
        arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 7:
            raise ValueError(f"EEF target must have shape (7,), got {arr.shape}")
        self._refresh_motion_mode_if_needed("eef")
        pose_raw, grip_raw = units.eef_to_raw(
            arr,
            open_raw=self.config.gripper_open_raw,
            closed_raw=self.config.gripper_closed_raw,
        )
        arm.EndPoseCtrl(*pose_raw)
        arm.GripperCtrl(abs(grip_raw), self.config.gripper_effort, self.config.gripper_status_code, 0x00)

    def read_state(self) -> PiperArmState:
        arm = self.ensure_connected()
        now = time.time()
        joint_msg = arm.GetArmJointMsgs()
        gripper_msg = arm.GetArmGripperMsgs()
        eef_msg = arm.GetArmEndPoseMsgs()

        joint_state = _child(joint_msg, "joint_state", "joint", "arm_joint")
        joint_raw = [
            _value(joint_state, (name, name.upper(), name.replace("_", ""), f"{name}_pos"))
            for name in PIPER_JOINT_NAMES
        ]

        gripper_state = _child(gripper_msg, "gripper_state", "gripper", "gripper_msg")
        gripper_raw = _value(gripper_state, ("grippers_angle", "gripper_angle", "angle", "pos"), default=0.0)
        gripper_norm = units.gripper_raw_to_norm(
            gripper_raw,
            open_raw=self.config.gripper_open_raw,
            closed_raw=self.config.gripper_closed_raw,
        )

        joint_pos = np.concatenate(
            [units.joint_raw_to_rad(joint_raw), np.array([gripper_norm], dtype=np.float32)]
        ).astype(np.float32)

        eef_state = _child(eef_msg, "end_pose", "tcp_pose", "arm_end_pose", "pose")
        raw_pose = [
            _value(eef_state, ("X_axis", "x_axis", "x", "X", "pose_x")),
            _value(eef_state, ("Y_axis", "y_axis", "y", "Y", "pose_y")),
            _value(eef_state, ("Z_axis", "z_axis", "z", "Z", "pose_z")),
            _value(eef_state, ("RX_axis", "rx_axis", "rx", "RX", "roll")),
            _value(eef_state, ("RY_axis", "ry_axis", "ry", "RY", "pitch")),
            _value(eef_state, ("RZ_axis", "rz_axis", "rz", "RZ", "yaw")),
        ]
        eef_pos = units.raw_to_eef(
            raw_pose,
            gripper_raw=gripper_raw,
            open_raw=self.config.gripper_open_raw,
            closed_raw=self.config.gripper_closed_raw,
        )

        if self._last_state is None:
            qvel = np.zeros(7, dtype=np.float32)
        else:
            dt = max(1e-6, now - self._last_state.timestamp)
            qvel = ((joint_pos - self._last_state.joint_pos) / dt).astype(np.float32)

        effort = np.zeros(7, dtype=np.float32)
        ctrl_mode = None
        try:
            ctrl_mode = self.read_ctrl_mode()
        except Exception:
            logger.debug("Could not read ctrl_mode for %s", self.config.name, exc_info=True)

        state = PiperArmState(
            joint_pos=joint_pos,
            eef_pos=eef_pos,
            joint_qvel=qvel,
            joint_effort=effort,
            timestamp=now,
            ctrl_mode=ctrl_mode,
        )
        self._last_state = state
        return state

    def diagnostics_snapshot(self) -> dict[str, Any]:
        arm = self.ensure_connected()
        out: dict[str, Any] = {"config": dataclasses.asdict(self.config)}
        for name in (
            "GetArmStatus",
            "GetArmJointMsgs",
            "GetArmGripperMsgs",
            "GetArmEndPoseMsgs",
            "GetArmJointCtrl",
            "GetArmGripperCtrl",
            "GetGripperTeachingPendantParamFeedback",
        ):
            if not hasattr(arm, name):
                continue
            try:
                out[name] = _to_plain(getattr(arm, name)())
            except Exception as exc:
                out[name] = {"error": repr(exc)}
        return out

