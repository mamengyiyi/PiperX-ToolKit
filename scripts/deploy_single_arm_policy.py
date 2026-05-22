#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit.env.cameras import CameraConfig, CameraManager
from piperx_toolkit.env.piper_arm import PiperArm, PiperArmConfig
from piperx_toolkit.utils.logging import setup_logging


class NumpySequencePolicy:
    def __init__(self, path: str):
        actions = np.load(path).astype(np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected .npy action sequence with shape (T, 7), got {actions.shape}")
        self.actions = actions
        self.index = 0

    def select_action(self, batch: dict[str, Any]) -> np.ndarray:
        if self.index >= len(self.actions):
            return self.actions[-1]
        action = self.actions[self.index]
        self.index += 1
        return action


class SingleArmPolicyAdapter:
    def __init__(
        self,
        policy_path: str,
        loader: str = "auto",
        device: str = "auto",
        camera_name: str = "front",
        task: str = "",
        local_files_only: bool = False,
    ):
        self.policy_path = policy_path
        self.loader = loader
        self.device = self._resolve_device(device)
        self.camera_name = camera_name
        self.task = task
        self.local_files_only = local_files_only
        self.policy = self._load_policy()
        self.action_queue: deque[np.ndarray] = deque()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            return "cpu"

    def _load_policy(self) -> Any:
        suffix = Path(self.policy_path).suffix.lower()
        if self.loader in {"auto", "npy"} and suffix == ".npy":
            return NumpySequencePolicy(self.policy_path)
        if self.loader == "npy":
            raise ValueError("--policy-loader npy requires a .npy file")

        if self.loader in {"auto", "lerobot"}:
            try:
                return self._load_lerobot_policy()
            except Exception as exc:
                if self.loader == "lerobot":
                    raise
                lerobot_error = exc
        else:
            lerobot_error = None

        if self.loader in {"auto", "torch"}:
            try:
                return self._load_torch_policy()
            except Exception as exc:
                if self.loader == "torch":
                    raise
                raise RuntimeError(
                    "Could not load policy automatically. Tried LeRobot and torch loaders. "
                    f"LeRobot error: {lerobot_error!r}. Torch error: {exc!r}."
                ) from exc
        raise ValueError(f"Unsupported policy loader: {self.loader}")

    def _load_torch_policy(self) -> Any:
        import torch

        policy = torch.load(self.policy_path, map_location=self.device)
        if isinstance(policy, dict):
            for key in ("policy", "model", "module"):
                if key in policy and hasattr(policy[key], "__call__"):
                    policy = policy[key]
                    break
            else:
                raise TypeError(
                    "Torch checkpoint is a dict but does not contain a callable 'policy', 'model', or 'module'. "
                    "For LeRobot checkpoints, use --policy-loader lerobot with the checkpoint directory."
                )
        if hasattr(policy, "to"):
            policy.to(self.device)
        if hasattr(policy, "eval"):
            policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        return policy

    def _load_lerobot_policy(self) -> Any:
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class
        except ImportError:
            from lerobot.common.policies.factory import get_policy_class  # type: ignore
            from lerobot.common.policies.pretrained import PreTrainedConfig  # type: ignore

        cfg = PreTrainedConfig.from_pretrained(
            self.policy_path,
            local_files_only=self.local_files_only,
        )
        if hasattr(cfg, "device"):
            cfg.device = self.device
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(
            self.policy_path,
            config=cfg,
            local_files_only=self.local_files_only,
        )
        if hasattr(policy, "to"):
            policy.to(self.device)
        if hasattr(policy, "eval"):
            policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        return policy

    def make_batch(self, state: np.ndarray, image: np.ndarray | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        import torch

        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device),
        }
        images_np: dict[str, np.ndarray] = {}
        if image is not None:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
            images_np[self.camera_name] = image
            image_chw = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch[f"observation.images.{self.camera_name}"] = torch.from_numpy(image_chw).unsqueeze(0).to(self.device)
        if self.task:
            batch["task"] = [self.task]
        return batch, images_np

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            for key in ("action", "actions", "prediction"):
                if key in value:
                    value = value[key]
                    break
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _split_actions(value: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 7:
                raise ValueError(f"Policy action must have 7 values, got {arr.shape}")
            return [arr.copy()]
        if arr.ndim == 2:
            if arr.shape[-1] != 7:
                raise ValueError(f"Policy action last dim must be 7, got {arr.shape}")
            if arr.shape[0] == 1:
                return [arr[0].copy()]
            return [row.copy() for row in arr]
        if arr.ndim == 3:
            if arr.shape[-1] != 7:
                raise ValueError(f"Policy action last dim must be 7, got {arr.shape}")
            return [row.copy() for row in arr[0]]
        raise ValueError(f"Unsupported policy action shape: {arr.shape}")

    def predict(self, state: np.ndarray, image: np.ndarray | None) -> np.ndarray:
        if self.action_queue:
            return self.action_queue.popleft()

        batch, images_np = self.make_batch(state, image)
        errors: list[str] = []

        try:
            import torch

            with torch.inference_mode():
                if hasattr(self.policy, "select_action"):
                    out = self.policy.select_action(batch)
                    actions = self._split_actions(self._to_numpy(out))
                    self.action_queue.extend(actions[1:])
                    return actions[0]
                if hasattr(self.policy, "predict"):
                    out = self.policy.predict(images_np, state)
                    actions = self._split_actions(self._to_numpy(out))
                    self.action_queue.extend(actions[1:])
                    return actions[0]
                if callable(self.policy):
                    try:
                        out = self.policy(batch)
                    except TypeError:
                        try:
                            out = self.policy(images_np, state)
                        except TypeError:
                            out = self.policy(state)
                    actions = self._split_actions(self._to_numpy(out))
                    self.action_queue.extend(actions[1:])
                    return actions[0]
        except Exception as exc:
            errors.append(repr(exc))

        raise RuntimeError(f"Policy inference failed. Errors: {errors}")


def camera_device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def make_camera(args: argparse.Namespace) -> CameraManager:
    return CameraManager(
        {
            args.camera_name: CameraConfig(
                name=args.camera_name,
                backend=args.camera_backend,
                device=camera_device_arg(args.camera_device),
                width=args.width,
                height=args.height,
                fps=int(args.hz),
            )
        }
    )


def connect_camera(args: argparse.Namespace) -> CameraManager | None:
    if args.no_camera:
        print("Camera disabled by --no-camera.")
        return None
    last_error: Exception | None = None
    for attempt in range(1, args.camera_open_retries + 1):
        cameras = make_camera(args)
        try:
            cameras.connect()
            time.sleep(max(0.0, args.camera_warmup_s))
            return cameras
        except Exception as exc:
            last_error = exc
            cameras.close()
            if attempt < args.camera_open_retries:
                print(f"Camera open failed on attempt {attempt}/{args.camera_open_retries}: {exc}")
                time.sleep(max(0.0, args.camera_open_retry_s))
    if args.camera_fail_soft:
        print(f"WARNING: could not open camera. Running with black frames. Last error: {last_error}")
        return None
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not open camera")


def read_camera(
    cameras: CameraManager | None,
    args: argparse.Namespace,
    last_good_image: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if cameras is None:
        image = np.zeros((args.height, args.width, 3), dtype=np.uint8) if not args.policy_no_image else None
        return image, last_good_image, 0

    failures = 0
    last_error: Exception | None = None
    for _ in range(max(1, args.camera_read_retries)):
        try:
            image = cameras.read_all()[args.camera_name]
            return image, image, failures
        except RuntimeError as exc:
            failures += 1
            last_error = exc
            time.sleep(0.02)
    if not args.camera_fail_soft and last_error is not None:
        raise last_error
    if last_good_image is not None:
        return last_good_image, last_good_image, failures
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8) if not args.policy_no_image else None
    return image, last_good_image, failures


def lowpass_action(action: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    if previous is None or alpha >= 1.0:
        return action
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * action + (1.0 - alpha) * previous).astype(np.float32)


def limit_joint_target(target: np.ndarray, current: np.ndarray, max_delta_rad: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    target[6] = float(np.clip(target[6], 0.0, 1.0))
    if max_delta_rad > 0:
        delta = np.clip(target[:6] - current[:6], -max_delta_rad, max_delta_rad)
        target[:6] = current[:6] + delta
    return target


def send_action(arm: PiperArm, action: np.ndarray, action_mode: str, args: argparse.Namespace) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(7)
    if action_mode == "absolute_joint":
        current = arm.read_state().joint_pos
        target = limit_joint_target(action, current, args.max_joint_delta_rad)
        arm.send_joint_target(target)
    elif action_mode == "absolute_eef":
        action[6] = float(np.clip(action[6], 0.0, 1.0))
        arm.send_eef_target(action)
    elif action_mode == "delta_eef":
        current = arm.read_state().eef_pos
        target = current.copy()
        target[:6] = current[:6] + action[:6]
        target[6] = float(np.clip(current[6] + action[6], 0.0, 1.0))
        arm.send_eef_target(target)
    elif action_mode == "smooth_eef":
        smooth_eef(arm, action, args)
    else:
        raise ValueError(f"Unsupported action mode: {action_mode}")


def smooth_eef(arm: PiperArm, target: np.ndarray, args: argparse.Namespace) -> None:
    start = arm.read_state().eef_pos
    target = np.asarray(target, dtype=np.float32).copy()
    target[6] = float(np.clip(target[6], 0.0, 1.0))
    linear_dist = float(np.linalg.norm(target[:3] - start[:3]))
    angular_dist = float(np.linalg.norm(target[3:6] - start[3:6]))
    gripper_dist = abs(float(target[6] - start[6]))
    duration = max(
        args.smooth_min_duration_s,
        linear_dist / max(args.smooth_linear_speed_m_s, 1e-6),
        angular_dist / max(args.smooth_angular_speed_rad_s, 1e-6),
        gripper_dist / max(args.smooth_gripper_speed_s, 1e-6),
    )
    duration = float(np.clip(duration, args.smooth_min_duration_s, args.smooth_max_duration_s))
    steps = max(2, int(round(duration * args.smooth_hz)))
    dt = 1.0 / max(args.smooth_hz, 1.0)
    for i in range(1, steps + 1):
        alpha = i / steps
        arm.send_eef_target((start + alpha * (target - start)).astype(np.float32))
        time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a single-arm PiperX policy with front RGB + joint state.")
    parser.add_argument("--policy", required=True, help=".npy action sequence, torch checkpoint, LeRobot checkpoint dir, or HF model id.")
    parser.add_argument("--policy-loader", default="auto", choices=["auto", "npy", "torch", "lerobot"])
    parser.add_argument("--local-files-only", action="store_true", help="Do not download LeRobot checkpoints from Hugging Face Hub.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="")
    parser.add_argument("--state-key", default="joint_pos", choices=["joint_pos", "eef_pos"])
    parser.add_argument("--policy-no-image", action="store_true", help="Do not pass observation.images.front to the policy.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    parser.add_argument("--set-motion-output-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-ratio", type=int, default=30)
    parser.add_argument("--no-high-follow", action="store_true")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-backend", default="opencv", choices=["mock", "opencv"])
    parser.add_argument("--camera-device", default="4")
    parser.add_argument("--camera-fail-soft", action="store_true")
    parser.add_argument("--camera-open-retries", type=int, default=3)
    parser.add_argument("--camera-open-retry-s", type=float, default=0.5)
    parser.add_argument("--camera-read-retries", type=int, default=10)
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--action-mode", default="absolute_joint", choices=["absolute_joint", "absolute_eef", "smooth_eef", "delta_eef"])
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means until Ctrl+C or --max-steps.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.08)
    parser.add_argument("--lowpass-alpha", type=float, default=0.8)
    parser.add_argument("--smooth-hz", type=float, default=50.0)
    parser.add_argument("--smooth-linear-speed-m-s", type=float, default=0.15)
    parser.add_argument("--smooth-angular-speed-rad-s", type=float, default=0.7)
    parser.add_argument("--smooth-gripper-speed-s", type=float, default=1.0)
    parser.add_argument("--smooth-min-duration-s", type=float, default=0.08)
    parser.add_argument("--smooth-max-duration-s", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the arm. Without this, inference is dry-run only.")
    args = parser.parse_args()
    setup_logging()

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    arm = PiperArm(
        PiperArmConfig(
            name=args.side,
            can_name=args.can,
            backend=args.backend,
            enable_on_connect=False,
            speed_ratio=args.speed_ratio,
            high_follow=not args.no_high_follow,
        )
    )
    cameras: CameraManager | None = None
    try:
        arm.connect()
        cameras = connect_camera(args)
        if args.set_motion_output_role:
            arm.set_motion_output_role()
            time.sleep(0.2)
        if args.execute and args.enable:
            if not arm.enable():
                raise RuntimeError("EnablePiper() timed out. Check power, emergency stop, and CAN link.")

        policy = SingleArmPolicyAdapter(
            args.policy,
            loader=args.policy_loader,
            device=args.device,
            camera_name=args.camera_name,
            task=args.task,
            local_files_only=args.local_files_only,
        )
        print(f"Loaded policy with loader={args.policy_loader}, device={policy.device}")
        print("Dry-run mode: actions will not be sent." if not args.execute else "EXECUTE mode: actions will be sent to the arm.")

        dt = 1.0 / max(args.hz, 1e-6)
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        step = 0
        previous_action: np.ndarray | None = None
        last_good_image: np.ndarray | None = None
        camera_failures = 0
        while args.max_steps is None or step < args.max_steps:
            if deadline is not None and time.monotonic() >= deadline:
                break
            loop_t = time.monotonic()
            state = arm.read_state()
            state_vec = state.joint_pos if args.state_key == "joint_pos" else state.eef_pos
            image, last_good_image, failures = read_camera(cameras, args, last_good_image)
            camera_failures += failures
            action = policy.predict(state_vec, None if args.policy_no_image else image)
            action = lowpass_action(action, previous_action, args.lowpass_alpha)
            previous_action = action.copy()

            if args.execute:
                send_action(arm, action, args.action_mode, args)

            step += 1
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print(
                    f"step={step} state={np.array2string(state_vec, precision=4)} "
                    f"action={np.array2string(action, precision=4)} camera_failures={camera_failures}"
                )

            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Interrupted, stopping deployment.")
    finally:
        if cameras is not None:
            cameras.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
