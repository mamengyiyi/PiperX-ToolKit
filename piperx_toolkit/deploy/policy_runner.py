from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def default_state(obs: dict[str, np.ndarray], keys: tuple[str, ...] = ("left_joint_pos", "right_joint_pos")) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys], axis=0)


def default_images(obs: dict[str, np.ndarray], cameras: tuple[str, ...] = ("front", "left_wrist", "right_wrist")) -> dict[str, np.ndarray]:
    return {cam: obs[f"{cam}_color"] for cam in cameras if f"{cam}_color" in obs}


def split_bimanual_action(action: np.ndarray) -> dict[str, np.ndarray | None]:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 14:
        raise ValueError(f"Policy action must have at least 14 values, got {arr.shape[0]}")
    return {"left": arr[:7].copy(), "right": arr[7:14].copy()}


class NumpyReplayPolicy:
    def __init__(self, path: str):
        self.actions = np.load(path).astype(np.float32)
        self.idx = 0

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        if self.idx >= len(self.actions):
            return self.actions[-1]
        action = self.actions[self.idx]
        self.idx += 1
        return action


def load_policy(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return NumpyReplayPolicy(path)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to load non-.npy policies") from exc
    policy = torch.load(path, map_location="cpu")
    if hasattr(policy, "eval"):
        policy.eval()
    return policy


def predict_with_policy(policy: Any, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
    if hasattr(policy, "predict"):
        return np.asarray(policy.predict(images, state), dtype=np.float32)
    if callable(policy):
        try:
            return np.asarray(policy(images, state), dtype=np.float32)
        except TypeError:
            return np.asarray(policy(state), dtype=np.float32)
    raise TypeError("Policy must be callable or expose predict(images, state)")


def _clip_gripper(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _lowpass(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    if previous is None or alpha >= 1.0:
        return current
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * current + (1.0 - alpha) * previous).astype(np.float32)


def _clip_vector_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return delta
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm:
        return delta
    return (delta * (max_norm / norm)).astype(np.float32)


@dataclass
class ActionLimiter:
    max_joint_delta_rad: float = 0.15
    lowpass_alpha: float = 1.0

    def filter(self, action: dict[str, np.ndarray | None], current_obs: dict[str, np.ndarray], previous: dict[str, np.ndarray | None] | None) -> dict[str, np.ndarray | None]:
        return BimanualActionSmoother(
            action_mode="absolute_joint",
            max_joint_delta_rad=self.max_joint_delta_rad,
            lowpass_alpha=self.lowpass_alpha,
        ).filter(action, current_obs, previous)


@dataclass
class BimanualActionSmoother:
    """Mode-aware action smoothing for bimanual deployment."""

    action_mode: str = "absolute_joint"
    max_joint_delta_rad: float = 0.08
    max_eef_linear_delta_m: float = 0.03
    max_eef_angular_delta_rad: float = 0.15
    max_delta_eef_linear_m: float = 0.02
    max_delta_eef_angular_rad: float = 0.10
    max_gripper_delta: float = 0.15
    lowpass_alpha: float = 0.8

    def filter(
        self,
        action: dict[str, np.ndarray | None],
        current_obs: dict[str, np.ndarray],
        previous: dict[str, np.ndarray | None] | None,
    ) -> dict[str, np.ndarray | None]:
        mode = str(self.action_mode).strip().lower()
        out: dict[str, np.ndarray | None] = {"left": None, "right": None}
        for side in ("left", "right"):
            target = action.get(side)
            if target is None:
                continue
            prev = previous.get(side) if previous is not None else None
            if mode == "absolute_joint":
                smoothed = self._smooth_absolute_joint(target, current_obs, side, prev)
            elif mode == "absolute_eef":
                smoothed = self._smooth_absolute_eef(target, current_obs, side, prev)
            elif mode == "delta_eef":
                smoothed = self._smooth_delta_eef(target, prev)
            elif mode == "smooth_eef":
                smoothed = self._smooth_absolute_eef(target, current_obs, side, prev)
            else:
                raise ValueError(f"Unsupported action_mode for smoothing: {mode}")
            out[side] = smoothed
        return out

    def _smooth_absolute_joint(
        self,
        target: np.ndarray,
        current_obs: dict[str, np.ndarray],
        side: str,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32).copy()
        target[6] = _clip_gripper(target[6])
        current = np.asarray(current_obs[f"{side}_joint_pos"], dtype=np.float32)
        if self.max_joint_delta_rad > 0:
            delta = np.clip(target[:6] - current[:6], -self.max_joint_delta_rad, self.max_joint_delta_rad)
            target[:6] = current[:6] + delta
        if self.max_gripper_delta > 0:
            gripper_delta = float(np.clip(target[6] - current[6], -self.max_gripper_delta, self.max_gripper_delta))
            target[6] = _clip_gripper(current[6] + gripper_delta)
        return _lowpass(target, previous, self.lowpass_alpha)

    def _smooth_absolute_eef(
        self,
        target: np.ndarray,
        current_obs: dict[str, np.ndarray],
        side: str,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32).copy()
        target[6] = _clip_gripper(target[6])
        current = np.asarray(current_obs[f"{side}_eef_pos"], dtype=np.float32)
        linear_delta = _clip_vector_delta(target[:3] - current[:3], self.max_eef_linear_delta_m)
        angular_delta = _clip_vector_delta(target[3:6] - current[3:6], self.max_eef_angular_delta_rad)
        target[:3] = current[:3] + linear_delta
        target[3:6] = current[3:6] + angular_delta
        if self.max_gripper_delta > 0:
            gripper_delta = float(np.clip(target[6] - current[6], -self.max_gripper_delta, self.max_gripper_delta))
            target[6] = _clip_gripper(current[6] + gripper_delta)
        return _lowpass(target, previous, self.lowpass_alpha)

    def _smooth_delta_eef(self, delta: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32).copy()
        delta[:3] = _clip_vector_delta(delta[:3], self.max_delta_eef_linear_m)
        delta[3:6] = _clip_vector_delta(delta[3:6], self.max_delta_eef_angular_rad)
        if self.max_gripper_delta > 0:
            delta[6] = float(np.clip(delta[6], -self.max_gripper_delta, self.max_gripper_delta))
        return _lowpass(delta, previous, self.lowpass_alpha)


class PolicyRunner:
    def __init__(
        self,
        env,
        policy: Any | Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray],
        action_mode: str = "absolute_joint",
        hz: float = 20.0,
        guard_motion_mode: bool = True,
        limiter: ActionLimiter | None = None,
        smoother: BimanualActionSmoother | None = None,
        smooth_actions: bool = True,
        execute: bool = True,
        print_every: int = 0,
    ):
        self.env = env
        self.policy = policy
        self.action_mode = action_mode
        self.hz = hz
        self.guard_motion_mode = guard_motion_mode
        if smoother is not None:
            self.smoother = smoother
        elif limiter is not None:
            self.smoother = BimanualActionSmoother(
                action_mode=action_mode,
                max_joint_delta_rad=limiter.max_joint_delta_rad,
                lowpass_alpha=limiter.lowpass_alpha,
            )
        elif smooth_actions:
            self.smoother = BimanualActionSmoother(action_mode=action_mode)
        else:
            self.smoother = None
        self.execute = execute
        self.print_every = print_every

    def run(self, max_steps: int | None = None) -> None:
        if self.execute and self.guard_motion_mode:
            self.env.guard_can_accept_motion()
        dt = 1.0 / self.hz
        previous_action: dict[str, np.ndarray | None] | None = None
        step = 0
        obs = self.env.get_observation()
        try:
            while max_steps is None or step < max_steps:
                t0 = time.monotonic()
                state = default_state(obs)
                images = default_images(obs)
                raw_action = predict_with_policy(self.policy, images, state)
                action = split_bimanual_action(raw_action)
                if self.smoother is not None:
                    action = self.smoother.filter(action, obs, previous_action)
                if self.execute:
                    obs = self.env.step(action, action_mode=self.action_mode, return_observation=True)
                else:
                    obs = self.env.get_observation()
                if self.print_every > 0 and (step == 0 or (step + 1) % self.print_every == 0):
                    left = action["left"]
                    right = action["right"]
                    print(
                        f"step={step + 1:06d} "
                        f"left={np.array2string(left, precision=4, suppress_small=True) if left is not None else None} "
                        f"right={np.array2string(right, precision=4, suppress_small=True) if right is not None else None}"
                    )
                previous_action = action
                step += 1
                sleep_s = dt - (time.monotonic() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            return
