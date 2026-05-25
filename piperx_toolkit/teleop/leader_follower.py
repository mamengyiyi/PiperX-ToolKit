from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig


def _ones7() -> np.ndarray:
    return np.ones(7, dtype=np.float32)


def _zeros7() -> np.ndarray:
    return np.zeros(7, dtype=np.float32)


@dataclass
class JointMapping:
    signs: np.ndarray = field(default_factory=_ones7)
    offsets: np.ndarray = field(default_factory=_zeros7)

    def __post_init__(self) -> None:
        self.signs = np.asarray(self.signs, dtype=np.float32).reshape(7)
        self.offsets = np.asarray(self.offsets, dtype=np.float32).reshape(7)

    def apply(self, leader_joint_pos: np.ndarray) -> np.ndarray:
        target = np.asarray(leader_joint_pos, dtype=np.float32).reshape(7) * self.signs + self.offsets
        target[6] = float(np.clip(target[6], 0.0, 1.0))
        return target.astype(np.float32)


@dataclass
class StepResult:
    leader_joint_pos: np.ndarray
    follower_joint_pos: np.ndarray
    target_joint_pos: np.ndarray
    command_joint_pos: np.ndarray
    leader_eef_pos: np.ndarray | None = None
    follower_eef_pos: np.ndarray | None = None
    follower_joint_qvel: np.ndarray | None = None
    follower_joint_effort: np.ndarray | None = None
    timestamp: float | None = None

    @property
    def max_abs_error_rad(self) -> float:
        return float(np.max(np.abs(self.target_joint_pos[:6] - self.follower_joint_pos[:6])))


def limit_joint_step(target: np.ndarray, current: np.ndarray, max_joint_delta_rad: float) -> np.ndarray:
    command = np.asarray(target, dtype=np.float32).copy()
    command[6] = float(np.clip(command[6], 0.0, 1.0))
    if max_joint_delta_rad > 0:
        delta = np.clip(command[:6] - current[:6], -max_joint_delta_rad, max_joint_delta_rad)
        command[:6] = current[:6] + delta
    return command.astype(np.float32)


def lowpass(command: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous is None or alpha >= 1.0:
        return command.astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * command + (1.0 - alpha) * previous).astype(np.float32)


class LeaderFollowerPair:
    """Joint-space leader-follower teleoperation for one PiperX arm pair."""

    action_mode = "absolute_joint"

    def __init__(
        self,
        leader: PiperArm,
        follower: PiperArm,
        mapping: JointMapping | None = None,
        max_joint_delta_rad: float = 0.08,
        lowpass_alpha: float = 0.8,
    ):
        self.leader = leader
        self.follower = follower
        self.mapping = mapping or JointMapping()
        self.max_joint_delta_rad = max_joint_delta_rad
        self.lowpass_alpha = lowpass_alpha
        self.previous_command: np.ndarray | None = None
        self.running = False

    @classmethod
    def from_configs(
        cls,
        leader_config: PiperArmConfig,
        follower_config: PiperArmConfig,
        mapping: JointMapping | None = None,
        max_joint_delta_rad: float = 0.08,
        lowpass_alpha: float = 0.8,
    ) -> "LeaderFollowerPair":
        return cls(
            leader=PiperArm(leader_config),
            follower=PiperArm(follower_config),
            mapping=mapping,
            max_joint_delta_rad=max_joint_delta_rad,
            lowpass_alpha=lowpass_alpha,
        )

    def start(
        self,
        set_motion_output_role: bool = True,
        enable_follower: bool = True,
        startup_sleep_s: float = 0.2,
    ) -> None:
        try:
            self.leader.connect()
            self.follower.connect()
            if set_motion_output_role:
                self.leader.set_motion_output_role()
                self.follower.set_motion_output_role()
                time.sleep(max(0.0, startup_sleep_s))
            if enable_follower and not self.follower.enable():
                raise RuntimeError(f"EnablePiper() timed out for follower {self.follower.config.name}")
            self.running = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.running = False
        try:
            self.follower.disconnect()
        finally:
            self.leader.disconnect()

    def read_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        leader_state = self.leader.read_state()
        follower_state = self.follower.read_state()
        target = self.mapping.apply(leader_state.joint_pos)
        return leader_state.joint_pos, follower_state.joint_pos, target

    def step(self, execute: bool = True) -> StepResult:
        leader_state = self.leader.read_state()
        follower_state = self.follower.read_state()
        target = self.mapping.apply(leader_state.joint_pos)
        command = limit_joint_step(target, follower_state.joint_pos, self.max_joint_delta_rad)
        command = lowpass(command, self.previous_command, self.lowpass_alpha)
        command[6] = target[6]
        if execute:
            self.follower.send_joint_target(command)
        self.previous_command = command.copy()
        return StepResult(
            leader_joint_pos=leader_state.joint_pos.copy(),
            follower_joint_pos=follower_state.joint_pos.copy(),
            target_joint_pos=target.copy(),
            command_joint_pos=command.copy(),
            leader_eef_pos=leader_state.eef_pos.copy(),
            follower_eef_pos=follower_state.eef_pos.copy(),
            follower_joint_qvel=follower_state.joint_qvel.copy(),
            follower_joint_effort=follower_state.joint_effort.copy(),
            timestamp=follower_state.timestamp,
        )

    def approach_to_leader(
        self,
        hz: float = 20.0,
        tolerance_rad: float = 0.04,
        timeout_s: float = 20.0,
        execute: bool = True,
    ) -> StepResult:
        deadline = time.monotonic() + timeout_s
        dt = 1.0 / max(hz, 1e-6)
        last_result: StepResult | None = None
        while True:
            loop_t = time.monotonic()
            last_result = self.step(execute=execute)
            if last_result.max_abs_error_rad <= tolerance_rad:
                return last_result
            if loop_t >= deadline:
                raise TimeoutError(
                    f"Follower {self.follower.config.name} could not approach leader target within "
                    f"{timeout_s:.1f}s; final max joint error={last_result.max_abs_error_rad:.4f} rad."
                )
            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)


class BimanualLeaderFollowerTeleop:
    """Joint-space leader-follower teleoperation for left and right PiperX pairs."""

    action_mode = "absolute_joint"

    def __init__(self, left_pair: LeaderFollowerPair, right_pair: LeaderFollowerPair):
        self.left_pair = left_pair
        self.right_pair = right_pair
        self.running = False

    def start(
        self,
        set_motion_output_role: bool = True,
        enable_followers: bool = True,
        startup_sleep_s: float = 0.2,
    ) -> None:
        try:
            self.left_pair.start(
                set_motion_output_role=set_motion_output_role,
                enable_follower=enable_followers,
                startup_sleep_s=startup_sleep_s,
            )
            self.right_pair.start(
                set_motion_output_role=set_motion_output_role,
                enable_follower=enable_followers,
                startup_sleep_s=startup_sleep_s,
            )
            self.running = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.running = False
        try:
            self.right_pair.stop()
        finally:
            self.left_pair.stop()

    def step(self, execute: bool = True) -> dict[str, StepResult]:
        return {
            "left": self.left_pair.step(execute=execute),
            "right": self.right_pair.step(execute=execute),
        }

    def approach_to_leaders(
        self,
        hz: float = 20.0,
        tolerance_rad: float = 0.04,
        timeout_s: float = 20.0,
        execute: bool = True,
    ) -> dict[str, StepResult]:
        deadline = time.monotonic() + timeout_s
        dt = 1.0 / max(hz, 1e-6)
        last: dict[str, StepResult] | None = None
        while True:
            loop_t = time.monotonic()
            last = self.step(execute=execute)
            max_err = max(last["left"].max_abs_error_rad, last["right"].max_abs_error_rad)
            if max_err <= tolerance_rad:
                return last
            if loop_t >= deadline:
                raise TimeoutError(
                    f"Bimanual followers could not approach leader targets within {timeout_s:.1f}s; "
                    f"final max joint error={max_err:.4f} rad."
                )
            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)
