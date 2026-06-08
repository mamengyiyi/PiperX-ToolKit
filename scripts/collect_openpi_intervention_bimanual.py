#!/usr/bin/env python3
"""Collect bimanual OpenPI rollouts with human interventions.

During policy control, the two leader arms can mirror the actual follower arms
so the operator can grab the leaders from the current rollout pose and take over
when a failure is about to happen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.collect.simple_zarr import append_episode, next_episode_id, require_array
from piperx_toolkit.deploy.openpi_remote import OpenPIRemotePolicy
from piperx_toolkit.deploy.policy_runner import default_images, default_state, predict_with_policy, split_bimanual_action
from piperx_toolkit.env.cameras import CameraConfig
from piperx_toolkit.env.dual_piper_env import DualPiperXEnvConfig
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.teleop.leader_follower import JointMapping, limit_joint_step, lowpass
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging

try:
    from piperx_toolkit.deploy.policy_runner import BimanualActionSmoother
except ImportError:  # Older robot-side checkout.
    BimanualActionSmoother = None  # type: ignore[assignment]


CAMERA_NAMES = ("front", "left_wrist", "right_wrist")
CONTROL_POLICY = 0
CONTROL_INTERVENTION = 1
UNKNOWN_SUCCESS = 255


class KeyboardListener:
    def __init__(self):
        import termios

        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)

    def start(self) -> None:
        import tty

        tty.setraw(sys.stdin.fileno())

    def stop(self) -> None:
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        import select

        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None


@dataclass
class ArmBundle:
    left_leader: PiperArm
    right_leader: PiperArm
    followers: DualPiperXEnv


@dataclass
class EpisodeStats:
    steps: int
    duration_s: float
    fps: float
    intervention_steps: int
    camera_failures: int


@dataclass
class SimpleJointSmoother:
    max_joint_delta_rad: float
    max_gripper_delta: float
    lowpass_alpha: float

    def filter(
        self,
        action: dict[str, np.ndarray | None],
        obs: dict[str, np.ndarray],
        previous: dict[str, np.ndarray | None] | None,
    ) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {"left": None, "right": None}
        for side in ("left", "right"):
            target = action.get(side)
            if target is None:
                continue
            current = np.asarray(obs[f"{side}_joint_pos"], dtype=np.float32)
            command = limit_joint_step(target, current, self.max_joint_delta_rad)
            if self.max_gripper_delta > 0:
                gripper_delta = float(
                    np.clip(command[6] - current[6], -self.max_gripper_delta, self.max_gripper_delta)
                )
                command[6] = float(np.clip(current[6] + gripper_delta, 0.0, 1.0))
            prev = previous.get(side) if previous is not None else None
            out[side] = lowpass(command, prev, self.lowpass_alpha)
        return out


def parse_vec7(text: str, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"{name} must contain 7 comma-separated values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def camera_device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def camera_devices(args: argparse.Namespace) -> dict[str, str]:
    return {
        "front": args.front_device,
        "left_wrist": args.left_wrist_device,
        "right_wrist": args.right_wrist_device,
    }


def ensure_rgb_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")
    if image.shape[0] == height and image.shape[1] == width:
        return image
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def build_follower_config(args: argparse.Namespace) -> DualPiperXEnvConfig:
    config = load_env_config(args.config, backend=args.follower_backend, camera_backend=args.camera_backend)
    config.left_can = args.left_follower_can
    config.right_can = args.right_follower_can
    config.enable_on_connect = args.enable_followers_on_connect
    config.speed_ratio = args.follower_speed_ratio
    config.high_follow = not args.no_high_follow
    for name, device in camera_devices(args).items():
        if str(device).strip() == "":
            config.cameras.pop(name, None)
            continue
        base = config.cameras.get(name, CameraConfig(name=name, backend=args.camera_backend))
        config.cameras[name] = CameraConfig(
            name=name,
            backend=args.camera_backend,
            device=camera_device_arg(device),
            width=args.width,
            height=args.height,
            fps=int(args.hz),
        )
        if base is not None:
            config.cameras[name].backend = args.camera_backend or base.backend
    if args.no_camera:
        config.cameras = {}
    return config


def connect_followers(args: argparse.Namespace) -> DualPiperXEnv:
    last_error: Exception | None = None
    for attempt in range(1, args.camera_open_retries + 1):
        try:
            env = DualPiperXEnv(build_follower_config(args))
            time.sleep(max(0.0, args.camera_warmup_s))
            return env
        except Exception as exc:
            last_error = exc
            try:
                env.close()  # type: ignore[name-defined]
            except Exception:
                pass
            if args.no_camera or attempt >= args.camera_open_retries:
                break
            print(
                f"Open followers/cameras failed on attempt {attempt}/{args.camera_open_retries}: {exc}. "
                f"Retrying in {args.camera_open_retry_s:.1f}s..."
            )
            time.sleep(max(0.0, args.camera_open_retry_s))
    if args.camera_fail_soft and last_error is not None:
        print(f"WARNING: cameras failed to open; retrying with cameras disabled. Last error: {last_error}")
        args.no_camera = True
        return DualPiperXEnv(build_follower_config(args))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not connect followers")


def make_leader(name: str, can_name: str, args: argparse.Namespace) -> PiperArm:
    return PiperArm(
        PiperArmConfig(
            name=name,
            can_name=can_name,
            backend=args.leader_backend,
            enable_on_connect=False,
            speed_ratio=args.leader_speed_ratio,
            high_follow=not args.no_high_follow,
        )
    )


def connect_arms(args: argparse.Namespace) -> ArmBundle:
    followers = connect_followers(args)
    left_leader = make_leader("left_leader", args.left_leader_can, args)
    right_leader = make_leader("right_leader", args.right_leader_can, args)
    try:
        left_leader.connect()
        right_leader.connect()
        if args.set_motion_output_role:
            followers.set_motion_output_role()
            left_leader.set_motion_output_role()
            right_leader.set_motion_output_role()
            time.sleep(max(0.0, args.startup_sleep_s))
        if args.execute and not args.allow_teaching_mode:
            followers.guard_can_accept_motion()
            if args.mirror_leaders:
                left_leader.guard_can_accept_motion()
                right_leader.guard_can_accept_motion()
        if args.execute:
            if args.enable_followers and not followers.left_arm.enable():
                raise RuntimeError("EnablePiper() timed out for left follower")
            if args.enable_followers and not followers.right_arm.enable():
                raise RuntimeError("EnablePiper() timed out for right follower")
            if args.mirror_leaders:
                if args.enable_leaders and not left_leader.enable():
                    raise RuntimeError("EnablePiper() timed out for left leader")
                if args.enable_leaders and not right_leader.enable():
                    raise RuntimeError("EnablePiper() timed out for right leader")
        return ArmBundle(left_leader=left_leader, right_leader=right_leader, followers=followers)
    except Exception:
        right_leader.disconnect()
        left_leader.disconnect()
        followers.close()
        raise


def close_arms(arms: ArmBundle) -> None:
    try:
        arms.followers.close()
    finally:
        try:
            arms.right_leader.disconnect()
        finally:
            arms.left_leader.disconnect()


def open_or_create_dataset(path: str, args: argparse.Namespace) -> tuple[Any, Any, int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required. Install with `uv pip install -e '.[hardware,data]'`.") from exc

    try:
        root = zarr.open(path, mode="a", zarr_format=2)
    except TypeError:
        root = zarr.open(path, mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")
    image_shape = (3, args.height, args.width)

    for cam in CAMERA_NAMES:
        require_array(data, f"rgb_{cam}", image_shape, np.uint8, chunks=(1, *image_shape))

    for side in ("left", "right"):
        for suffix in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort"):
            require_array(data, f"{side}_{suffix}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"leader_{side}_joint_pos", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"leader_{side}_eef_pos", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"policy_action_{side}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"expert_action_{side}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"executed_action_{side}", (7,), np.float32, chunks=(1024, 7))
        require_array(data, f"mirror_action_{side}", (7,), np.float32, chunks=(1024, 7))
        # Compatibility alias for replay; for DAgger training, prefer expert_action_* with intervention_mask.
        require_array(data, f"action_{side}", (7,), np.float32, chunks=(1024, 7))

    require_array(data, "intervention_mask", (), np.uint8, chunks=(4096,))
    require_array(data, "control_source", (), np.uint8, chunks=(4096,))
    require_array(data, "episode_success", (), np.uint8, chunks=(4096,))
    require_array(data, "timestamp", (), np.float64, chunks=(4096,))
    require_array(data, "episode", (), np.uint32, chunks=(4096,))
    if "episode_ends" not in meta:
        meta.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(1024,))

    meta.attrs["config"] = json.dumps(
        {
            "robot_type": "piperx_bimanual_openpi_intervention",
            "left_leader_can": args.left_leader_can,
            "left_follower_can": args.left_follower_can,
            "right_leader_can": args.right_leader_can,
            "right_follower_can": args.right_follower_can,
            "leader_backend": args.leader_backend,
            "follower_backend": args.follower_backend,
            "cameras": camera_devices(args),
            "camera_backend": args.camera_backend,
            "image_size": [args.width, args.height],
            "hz": args.hz,
            "task": args.task,
            "prompt": args.prompt,
            "action_mode": args.action_mode,
            "policy_host": args.host,
            "policy_port": args.port,
            "policy_observation_format": args.observation_format,
            "policy_chunk_size": args.chunk_size,
            "action_from": "executed_action_left/right; expert labels are expert_action_* where intervention_mask=1",
            "left_joint_signs": args.left_joint_signs,
            "left_joint_offsets": args.left_joint_offsets,
            "right_joint_signs": args.right_joint_signs,
            "right_joint_offsets": args.right_joint_offsets,
            "max_joint_delta_rad": args.max_joint_delta_rad,
            "lowpass_alpha": args.lowpass_alpha,
            "mirror_leaders": args.mirror_leaders,
        },
        ensure_ascii=True,
    )
    return data, meta, next_episode_id(data)


def empty_buffer() -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {f"rgb_{cam}": [] for cam in CAMERA_NAMES}
    for side in ("left", "right"):
        for suffix in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort"):
            out[f"{side}_{suffix}"] = []
        out[f"leader_{side}_joint_pos"] = []
        out[f"leader_{side}_eef_pos"] = []
        out[f"policy_action_{side}"] = []
        out[f"expert_action_{side}"] = []
        out[f"executed_action_{side}"] = []
        out[f"mirror_action_{side}"] = []
        out[f"action_{side}"] = []
    for key in ("intervention_mask", "control_source", "episode_success", "timestamp", "episode"):
        out[key] = []
    return out


def append_scalar(buffer: dict[str, list[np.ndarray]], key: str, value: int | float, dtype: Any) -> None:
    buffer[key].append(np.asarray([value], dtype=dtype))


def append_rgb_from_obs(buffer: dict[str, list[np.ndarray]], obs: dict[str, np.ndarray], args: argparse.Namespace) -> int:
    failures = 0
    for cam in CAMERA_NAMES:
        key = f"{cam}_color"
        if key in obs:
            try:
                image = ensure_rgb_size(obs[key], args.width, args.height)
            except Exception:
                failures += 1
                image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        else:
            image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        buffer[f"rgb_{cam}"].append(image.transpose(2, 0, 1)[None])
    return failures


def nan_action() -> np.ndarray:
    return np.full(7, np.nan, dtype=np.float32)


def inverse_mapping_target(follower_joint_pos: np.ndarray, mapping: JointMapping) -> np.ndarray:
    follower_joint_pos = np.asarray(follower_joint_pos, dtype=np.float32).reshape(7)
    target = (follower_joint_pos - mapping.offsets) / mapping.signs
    target[6] = float(np.clip(target[6], 0.0, 1.0))
    return target.astype(np.float32)


def read_leader_states(arms: ArmBundle) -> dict[str, Any]:
    return {
        "left": arms.left_leader.read_state(),
        "right": arms.right_leader.read_state(),
    }


def make_smoother(args: argparse.Namespace) -> Any | None:
    if args.no_smooth:
        return None
    if BimanualActionSmoother is None:
        return SimpleJointSmoother(
            max_joint_delta_rad=args.max_joint_delta_rad,
            max_gripper_delta=args.max_gripper_delta,
            lowpass_alpha=args.lowpass_alpha,
        )
    return BimanualActionSmoother(
        action_mode=args.action_mode,
        max_joint_delta_rad=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
        lowpass_alpha=args.lowpass_alpha,
    )


def instantiate_policy(args: argparse.Namespace) -> OpenPIRemotePolicy:
    kwargs = dict(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        prompt=args.prompt,
        observation_format=args.observation_format,
        resize=args.image_resize,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
    )
    try:
        return OpenPIRemotePolicy(**kwargs)
    except TypeError:
        kwargs.pop("chunk_size")
        return OpenPIRemotePolicy(**kwargs)


def read_follower_observation(arms: ArmBundle, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], int]:
    try:
        return arms.followers.get_observation(include_camera=True), 0
    except RuntimeError:
        if not args.camera_fail_soft:
            raise
        return arms.followers.get_observation(include_camera=False), 1


def policy_action(
    policy: OpenPIRemotePolicy,
    obs: dict[str, np.ndarray],
    smoother: Any | None,
    previous_action: dict[str, np.ndarray | None] | None,
) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
    state = default_state(obs)
    images = default_images(obs)
    raw = predict_with_policy(policy, images, state)
    action = split_bimanual_action(raw)
    if smoother is not None:
        action = smoother.filter(action, obs, previous_action)
    return action, np.asarray(raw, dtype=np.float32).reshape(-1)[:14].copy()


def expert_action_from_leaders(
    obs: dict[str, np.ndarray],
    leader_states: dict[str, Any],
    mappings: dict[str, JointMapping],
    previous_action: dict[str, np.ndarray | None] | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | None]:
    actions: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        target = mappings[side].apply(leader_states[side].joint_pos)
        current = np.asarray(obs[f"{side}_joint_pos"], dtype=np.float32)
        command = limit_joint_step(target, current, args.max_joint_delta_rad)
        previous = previous_action.get(side) if previous_action is not None else None
        command = lowpass(command, previous, args.lowpass_alpha)
        command[6] = target[6]
        actions[side] = command.astype(np.float32)
    return actions


def mirror_leaders_to_followers(
    arms: ArmBundle,
    obs: dict[str, np.ndarray],
    leader_states: dict[str, Any],
    mappings: dict[str, JointMapping],
    previous_mirror: dict[str, np.ndarray | None] | None,
    args: argparse.Namespace,
    execute: bool,
) -> dict[str, np.ndarray | None]:
    commands: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side, leader in (("left", arms.left_leader), ("right", arms.right_leader)):
        target = inverse_mapping_target(obs[f"{side}_joint_pos"], mappings[side])
        current = leader_states[side].joint_pos
        command = limit_joint_step(target, current, args.max_leader_mirror_delta_rad)
        previous = previous_mirror.get(side) if previous_mirror is not None else None
        command = lowpass(command, previous, args.leader_mirror_lowpass_alpha)
        command[6] = target[6]
        if execute:
            leader.send_joint_target(command)
        commands[side] = command.astype(np.float32)
    return commands


def append_step(
    buffer: dict[str, list[np.ndarray]],
    obs: dict[str, np.ndarray],
    leader_states: dict[str, Any],
    policy_act: dict[str, np.ndarray | None],
    expert_act: dict[str, np.ndarray | None],
    executed_act: dict[str, np.ndarray | None],
    mirror_act: dict[str, np.ndarray | None],
    control_source: int,
    episode_id: int,
    success_label: int,
    args: argparse.Namespace,
) -> int:
    for side in ("left", "right"):
        for suffix in ("joint_pos", "eef_pos", "joint_qvel", "joint_effort"):
            buffer[f"{side}_{suffix}"].append(np.asarray(obs[f"{side}_{suffix}"], dtype=np.float32).reshape(1, 7))
        buffer[f"leader_{side}_joint_pos"].append(leader_states[side].joint_pos.reshape(1, 7).astype(np.float32))
        buffer[f"leader_{side}_eef_pos"].append(leader_states[side].eef_pos.reshape(1, 7).astype(np.float32))
        p = policy_act.get(side)
        e = expert_act.get(side)
        x = executed_act.get(side)
        m = mirror_act.get(side)
        buffer[f"policy_action_{side}"].append((p if p is not None else nan_action()).reshape(1, 7).astype(np.float32))
        buffer[f"expert_action_{side}"].append((e if e is not None else nan_action()).reshape(1, 7).astype(np.float32))
        buffer[f"executed_action_{side}"].append((x if x is not None else nan_action()).reshape(1, 7).astype(np.float32))
        buffer[f"mirror_action_{side}"].append((m if m is not None else nan_action()).reshape(1, 7).astype(np.float32))
        buffer[f"action_{side}"].append((x if x is not None else nan_action()).reshape(1, 7).astype(np.float32))
    append_scalar(buffer, "intervention_mask", 1 if control_source == CONTROL_INTERVENTION else 0, np.uint8)
    append_scalar(buffer, "control_source", control_source, np.uint8)
    append_scalar(buffer, "episode_success", success_label, np.uint8)
    append_scalar(buffer, "timestamp", float(obs["timestamp"][0]), np.float64)
    append_scalar(buffer, "episode", episode_id, np.uint32)
    return append_rgb_from_obs(buffer, obs, args)


def finalize_buffer(
    buffer: dict[str, list[np.ndarray]],
    t0: float,
    camera_failures: int,
    success_label: int,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    episode = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
    if episode["episode_success"].size:
        episode["episode_success"][:] = np.asarray(success_label, dtype=np.uint8)
    duration = time.time() - t0
    steps = int(episode["timestamp"].shape[0])
    interventions = int(np.sum(episode["intervention_mask"]))
    return EpisodeStats(steps, duration, steps / max(duration, 1e-6), interventions, camera_failures), episode


def wait_for_start(kb: KeyboardListener, episode_id: int) -> None:
    print(
        f"\r\nEpisode {episode_id}: Space start. During recording: [I] intervention, [P] policy, Enter stop, Ctrl+C quit.\r\n"
    )
    while True:
        key = kb.get_key()
        if key == " ":
            return
        if key == "\x03":
            raise KeyboardInterrupt
        time.sleep(0.05)


def choose_save_label(kb: KeyboardListener, episode_id: int, stats: EpisodeStats) -> tuple[bool, int]:
    print(
        f"\r\nEpisode {episode_id}: {stats.steps} steps, {stats.duration_s:.1f}s, "
        f"{stats.fps:.1f} FPS, intervention_steps={stats.intervention_steps}, "
        f"camera_failures={stats.camera_failures}.\r\n"
        "[Y] save success / [N] save failure / [U] save unknown / [D] discard\r\n"
    )
    while True:
        key = kb.get_key()
        if key == "y":
            return True, 1
        if key == "n":
            return True, 0
        if key == "u":
            return True, UNKNOWN_SUCCESS
        if key == "d":
            return False, UNKNOWN_SUCCESS
        if key == "\x03":
            raise KeyboardInterrupt
        time.sleep(0.05)


def approach_leaders_to_followers(
    arms: ArmBundle,
    mappings: dict[str, JointMapping],
    args: argparse.Namespace,
) -> None:
    if not args.mirror_leaders or not args.execute:
        return
    print("Approaching leader arms to current follower poses...")
    deadline = time.monotonic() + args.leader_approach_timeout_s
    dt = 1.0 / max(args.leader_approach_hz, 1e-6)
    previous: dict[str, np.ndarray | None] | None = None
    while True:
        loop_t = time.monotonic()
        obs = arms.followers.get_observation(include_camera=False)
        leader_states = read_leader_states(arms)
        commands = mirror_leaders_to_followers(arms, obs, leader_states, mappings, previous, args, execute=True)
        previous = commands
        errs = []
        for side in ("left", "right"):
            target = inverse_mapping_target(obs[f"{side}_joint_pos"], mappings[side])
            errs.append(float(np.max(np.abs(target[:6] - leader_states[side].joint_pos[:6]))))
        max_err = max(errs)
        if max_err <= args.leader_start_tolerance_rad:
            print(f"Leader approach done: max_err={max_err:.4f} rad")
            return
        if loop_t >= deadline:
            raise TimeoutError(f"Leader approach timed out; final max_err={max_err:.4f} rad")
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)


def record_episode(
    arms: ArmBundle,
    policy: OpenPIRemotePolicy,
    smoother: Any | None,
    mappings: dict[str, JointMapping],
    episode_id: int,
    args: argparse.Namespace,
    kb: KeyboardListener,
) -> tuple[EpisodeStats, dict[str, np.ndarray] | None, int]:
    wait_for_start(kb, episode_id)
    print("\r\nRecording in POLICY mode. Press [I] to intervene, [P] to return to policy, Enter to stop.\r\n")
    if hasattr(policy, "reset"):
        policy.reset()
    buffer = empty_buffer()
    dt = 1.0 / max(args.hz, 1e-6)
    t0 = time.time()
    camera_failures = 0
    mode = CONTROL_POLICY
    previous_policy_action: dict[str, np.ndarray | None] | None = None
    previous_expert_action: dict[str, np.ndarray | None] | None = None
    previous_mirror_action: dict[str, np.ndarray | None] | None = None

    while True:
        loop_t = time.monotonic()
        key = kb.get_key()
        if key in ("\r", "\n"):
            break
        if key == "\x03":
            raise KeyboardInterrupt
        if key == "i" and mode != CONTROL_INTERVENTION:
            mode = CONTROL_INTERVENTION
            previous_expert_action = None
            previous_mirror_action = None
            if args.clear_policy_queue_on_intervention and hasattr(policy, "reset"):
                policy.reset()
            print("\r\nINTERVENTION mode: move leaders to correct the followers. Press [P] for policy.\r\n")
        if key == "p" and mode != CONTROL_POLICY:
            mode = CONTROL_POLICY
            previous_policy_action = None
            if hasattr(policy, "reset"):
                policy.reset()
            print("\r\nPOLICY mode: leaders mirror followers. Press [I] to intervene.\r\n")

        obs, obs_camera_failures = read_follower_observation(arms, args)
        camera_failures += obs_camera_failures
        leader_states = read_leader_states(arms)
        policy_act: dict[str, np.ndarray | None] = {"left": None, "right": None}
        expert_act: dict[str, np.ndarray | None] = {"left": None, "right": None}
        mirror_act: dict[str, np.ndarray | None] = {"left": None, "right": None}

        if mode == CONTROL_POLICY or args.predict_during_intervention:
            try:
                policy_act, _ = policy_action(policy, obs, smoother, previous_policy_action)
            except Exception:
                if mode == CONTROL_POLICY:
                    raise
                policy_act = {"left": None, "right": None}

        if mode == CONTROL_POLICY:
            executed = policy_act
            previous_policy_action = policy_act
            previous_expert_action = None
            if args.mirror_leaders:
                mirror_act = mirror_leaders_to_followers(
                    arms,
                    obs,
                    leader_states,
                    mappings,
                    previous_mirror_action,
                    args,
                    execute=args.execute,
                )
                previous_mirror_action = mirror_act
        else:
            expert_act = expert_action_from_leaders(obs, leader_states, mappings, previous_expert_action, args)
            executed = expert_act
            previous_expert_action = expert_act
            previous_mirror_action = None

        if args.execute:
            arms.followers.step(executed, action_mode=args.action_mode, return_observation=False)

        camera_failures += append_step(
            buffer,
            obs,
            leader_states,
            policy_act,
            expert_act,
            executed,
            mirror_act,
            mode,
            episode_id,
            UNKNOWN_SUCCESS,
            args,
        )

        if args.print_every > 0 and len(buffer["timestamp"]) % args.print_every == 0:
            src = "intervention" if mode == CONTROL_INTERVENTION else "policy"
            left = executed["left"]
            right = executed["right"]
            print(
                f"\rstep={len(buffer['timestamp']):06d} src={src} "
                f"left={np.array2string(left, precision=4, suppress_small=True) if left is not None else None} "
                f"right={np.array2string(right, precision=4, suppress_small=True) if right is not None else None}      ",
                end="",
            )

        if args.duration is not None and time.time() - t0 >= args.duration:
            break
        sleep_s = dt - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not buffer["timestamp"]:
        return EpisodeStats(0, 0.0, 0.0, 0, camera_failures), None, UNKNOWN_SUCCESS
    stats, episode = finalize_buffer(buffer, t0, camera_failures, UNKNOWN_SUCCESS)
    save, label = choose_save_label(kb, episode_id, stats)
    if not save:
        return stats, None, label
    episode["episode_success"][:] = np.asarray(label, dtype=np.uint8)
    return stats, episode, label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect OpenPI bimanual rollouts with human intervention labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    env = parser.add_argument_group("robot and cameras")
    env.add_argument("--config", default="configs/dual_piperx.yaml", help="Base follower/camera YAML config.")
    env.add_argument("--left-leader-can", default="can0")
    env.add_argument("--left-follower-can", default="can1")
    env.add_argument("--right-leader-can", default="can2")
    env.add_argument("--right-follower-can", default="can3")
    env.add_argument("--leader-backend", default="sdk", choices=["sdk", "mock"])
    env.add_argument("--follower-backend", default="sdk", choices=["sdk", "mock"])
    env.add_argument("--camera-backend", default="realsense", choices=["realsense", "mock", "opencv"])
    env.add_argument("--front-device", default="10")
    env.add_argument("--left-wrist-device", default="14")
    env.add_argument("--right-wrist-device", default="2")
    env.add_argument("--width", type=int, default=640)
    env.add_argument("--height", type=int, default=480)
    env.add_argument("--no-camera", action="store_true")
    env.add_argument("--camera-fail-soft", action="store_true")
    env.add_argument("--camera-open-retries", type=int, default=3)
    env.add_argument("--camera-open-retry-s", type=float, default=0.5)
    env.add_argument("--camera-warmup-s", type=float, default=2.0)

    openpi = parser.add_argument_group("openpi server")
    openpi.add_argument("--host", default="127.0.0.1")
    openpi.add_argument("--port", type=int, default=8000)
    openpi.add_argument("--api-key", default=None)
    openpi.add_argument("--prompt", default="")
    openpi.add_argument("--observation-format", default="aloha", choices=["aloha", "piperx"])
    openpi.add_argument("--image-resize", type=int, default=224)
    openpi.add_argument("--action-dim", type=int, default=14)
    openpi.add_argument("--chunk-size", type=int, default=60)
    openpi.add_argument("--predict-during-intervention", action="store_true")
    openpi.add_argument("--clear-policy-queue-on-intervention", action=argparse.BooleanOptionalAction, default=True)

    control = parser.add_argument_group("control")
    control.add_argument("--dataset", "-d", default="datasets/openpi_intervention_bimanual.zarr")
    control.add_argument("--episodes", "-n", type=int, default=1)
    control.add_argument("--duration", type=float, default=None)
    control.add_argument("--hz", type=float, default=30.0)
    control.add_argument("--task", default="")
    control.add_argument("--action-mode", default="absolute_joint", choices=["absolute_joint"])
    control.add_argument("--execute", action="store_true")
    control.add_argument("--allow-teaching-mode", action="store_true")
    control.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    control.add_argument("--startup-sleep-s", type=float, default=0.2)
    control.add_argument("--enable-followers", action=argparse.BooleanOptionalAction, default=True)
    control.add_argument("--enable-leaders", action=argparse.BooleanOptionalAction, default=True)
    control.add_argument("--enable-followers-on-connect", action="store_true")
    control.add_argument("--follower-speed-ratio", type=int, default=100)
    control.add_argument("--leader-speed-ratio", type=int, default=100)
    control.add_argument("--no-high-follow", action="store_true")
    control.add_argument("--print-every", type=int, default=20)

    smooth = parser.add_argument_group("follower action smoothing")
    smooth.add_argument("--no-smooth", action="store_true")
    smooth.add_argument("--lowpass-alpha", type=float, default=0.8)
    smooth.add_argument("--max-joint-delta-rad", type=float, default=0.08)
    smooth.add_argument("--max-gripper-delta", type=float, default=0.3)
    smooth.add_argument("--left-joint-signs", default="1,1,1,1,1,1,1")
    smooth.add_argument("--left-joint-offsets", default="0,0,0,0,0,0,0")
    smooth.add_argument("--right-joint-signs", default="1,1,1,1,1,1,1")
    smooth.add_argument("--right-joint-offsets", default="0,0,0,0,0,0,0")

    mirror = parser.add_argument_group("leader mirror")
    mirror.add_argument("--mirror-leaders", action=argparse.BooleanOptionalAction, default=True)
    mirror.add_argument("--approach-leaders-start", action="store_true")
    mirror.add_argument("--leader-approach-hz", type=float, default=30.0)
    mirror.add_argument("--leader-approach-timeout-s", type=float, default=20.0)
    mirror.add_argument("--leader-start-tolerance-rad", type=float, default=0.08)
    mirror.add_argument("--max-leader-mirror-delta-rad", type=float, default=0.08)
    mirror.add_argument("--leader-mirror-lowpass-alpha", type=float, default=0.8)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()

    os.makedirs(os.path.dirname(args.dataset) or ".", exist_ok=True)
    data, meta, start_episode = open_or_create_dataset(args.dataset, args)
    mappings = {
        "left": JointMapping(
            signs=parse_vec7(args.left_joint_signs, "--left-joint-signs"),
            offsets=parse_vec7(args.left_joint_offsets, "--left-joint-offsets"),
        ),
        "right": JointMapping(
            signs=parse_vec7(args.right_joint_signs, "--right-joint-signs"),
            offsets=parse_vec7(args.right_joint_offsets, "--right-joint-offsets"),
        ),
    }
    policy = instantiate_policy(args)
    arms = connect_arms(args)
    smoother = make_smoother(args)

    print(
        f"OpenPI intervention collection: host={args.host}:{args.port} hz={args.hz} "
        f"chunk_size={args.chunk_size} mirror_leaders={args.mirror_leaders}"
    )
    print("DRY-RUN: no arm commands will be sent." if not args.execute else "EXECUTE: robot commands will be sent.")

    kb = KeyboardListener()
    saved = 0
    try:
        if args.approach_leaders_start:
            approach_leaders_to_followers(arms, mappings, args)
        kb.start()
        for offset in range(args.episodes):
            episode_id = start_episode + offset
            stats, episode, label = record_episode(arms, policy, smoother, mappings, episode_id, args, kb)
            if episode is None:
                print(f"\r\nDiscarded episode {episode_id}.\r\n")
                continue
            append_episode(data, meta, episode)
            saved += 1
            label_text = {0: "failure", 1: "success", UNKNOWN_SUCCESS: "unknown"}.get(label, str(label))
            print(
                f"\r\nSaved episode {episode_id} ({label_text}) -> {args.dataset}; "
                f"steps={stats.steps}, intervention_steps={stats.intervention_steps}.\r\n"
            )
    except KeyboardInterrupt:
        print("\r\nInterrupted, closing intervention collector.\r\n")
    finally:
        try:
            kb.stop()
        except Exception:
            pass
        close_arms(arms)
        print(f"Collection done. Saved {saved} new episode(s) to {args.dataset}.")


if __name__ == "__main__":
    main()
