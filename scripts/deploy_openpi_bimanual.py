#!/usr/bin/env python3
"""Deploy an OpenPI bimanual policy via websocket to DualPiperXEnv."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.deploy.openpi_remote import OpenPIRemotePolicy
from piperx_toolkit.deploy.policy_runner import (
    BimanualActionSmoother,
    PolicyRunner,
    default_images,
    default_state,
    predict_with_policy,
    split_bimanual_action,
)
from piperx_toolkit.env.dual_piper_env import DualPiperXEnvConfig, SmoothEEFConfig
from piperx_toolkit.utils.config import load_env_config
from piperx_toolkit.utils.logging import setup_logging

ACTION_MODES = ("absolute_joint", "absolute_eef", "smooth_eef", "delta_eef")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy an OpenPI bimanual policy (local or remote websocket server).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    env = parser.add_argument_group("environment")
    env.add_argument("--config", default="configs/dual_piperx.yaml", help="Robot/camera YAML config.")
    env.add_argument("--backend", default="sdk", choices=["sdk", "mock"])
    env.add_argument("--camera-backend", default="realsense", choices=["realsense", "mock", "opencv"])
    env.add_argument("--speed-ratio", type=int, default=100, help="Override arm speed_ratio from config.")

    openpi = parser.add_argument_group("openpi server")
    openpi.add_argument("--host", default="127.0.0.1", help="OpenPI policy server host.")
    openpi.add_argument("--port", type=int, default=8000, help="OpenPI policy server port.")
    openpi.add_argument("--api-key", default=None, help="Optional websocket API key.")
    openpi.add_argument("--prompt", default="", help="Language prompt sent with each observation.")
    openpi.add_argument(
        "--observation-format",
        default="aloha",
        choices=["aloha", "piperx"],
        help="Observation dict layout expected by the policy server.",
    )
    openpi.add_argument("--image-resize", type=int, default=224, help="Policy image resize (square).")
    openpi.add_argument("--action-dim", type=int, default=14, help="Bimanual action dimension.")
    openpi.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Max actions to consume per server inference (OpenPI action horizon). "
        "Must be <= training action_horizon (current model: 100). "
        "Set to 1 to replan every control step.",
    )
    openpi.add_argument(
        "--exec-chunk-size",
        type=int,
        default=None,
        help="Only execute/cache this many actions from each policy inference. Defaults to --chunk-size.",
    )

    control = parser.add_argument_group("control loop")
    control.add_argument(
        "--action-mode",
        default="absolute_joint",
        choices=ACTION_MODES,
        help="How env interprets each 7-dim arm command.",
    )
    control.add_argument("--hz", type=float, default=20.0, help="Main control loop frequency.")
    control.add_argument("--duration", type=float, default=0.0, help="Run time in seconds. 0 = until Ctrl+C.")
    control.add_argument("--max-steps", type=int, default=None, help="Maximum control steps.")
    control.add_argument(
        "--execute",
        action="store_true",
        help="Send actions to the robot. Without this flag, dry-run only.",
    )
    control.add_argument(
        "--allow-teaching-mode",
        action="store_true",
        help="Skip ctrl_mode guard. For mock/debug only.",
    )
    control.add_argument("--print-every", type=int, default=20, help="Print status every N steps. 0 = silent.")
    control.add_argument(
        "--profile-timing",
        action="store_true",
        help="Print OpenPI timing only from this script: arm read, camera read, inference, control, loop.",
    )
    control.add_argument(
        "--profile-print-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include left/right actions in --profile-timing output.",
    )

    smooth = parser.add_argument_group("action smoothing")
    smooth.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable all action smoothing/limiting.",
    )
    smooth.add_argument("--lowpass-alpha", type=float, default=0.8, help="EMA weight on new action (1 = off).")
    smooth.add_argument("--max-joint-delta-rad", type=float, default=0.08, help="Per-step joint limit (absolute_joint).")
    smooth.add_argument("--max-eef-linear-delta-m", type=float, default=0.03, help="Per-step EEF position limit.")
    smooth.add_argument("--max-eef-angular-delta-rad", type=float, default=0.15, help="Per-step EEF orientation limit.")
    smooth.add_argument("--max-delta-eef-linear-m", type=float, default=0.02, help="Limit policy delta EEF translation.")
    smooth.add_argument("--max-delta-eef-angular-rad", type=float, default=0.10, help="Limit policy delta EEF rotation.")
    smooth.add_argument("--max-gripper-delta", type=float, default=0.3, help="Per-step gripper change limit.")

    seef = parser.add_argument_group("smooth_eef env interpolation (action-mode=smooth_eef)")
    seef.add_argument("--smooth-eef-hz", type=float, default=60.0)
    seef.add_argument("--smooth-eef-linear-speed-m-s", type=float, default=0.15)
    seef.add_argument("--smooth-eef-angular-speed-rad-s", type=float, default=0.7)
    seef.add_argument("--smooth-eef-gripper-speed-s", type=float, default=1.0)
    seef.add_argument("--smooth-eef-min-duration-s", type=float, default=0.08)
    seef.add_argument("--smooth-eef-max-duration-s", type=float, default=5.0)

    return parser


def resolve_max_steps(args: argparse.Namespace) -> int | None:
    if args.max_steps is not None:
        return args.max_steps
    if args.duration > 0:
        return max(1, int(round(args.duration * args.hz)))
    return None


def build_env_config(args: argparse.Namespace) -> DualPiperXEnvConfig:
    config = load_env_config(args.config, backend=args.backend, camera_backend=args.camera_backend)
    config.enable_on_connect = True
    if args.speed_ratio is not None:
        config.speed_ratio = args.speed_ratio
    config.smooth_eef = SmoothEEFConfig(
        hz=args.smooth_eef_hz,
        linear_speed_m_s=args.smooth_eef_linear_speed_m_s,
        angular_speed_rad_s=args.smooth_eef_angular_speed_rad_s,
        gripper_speed_s=args.smooth_eef_gripper_speed_s,
        min_duration_s=args.smooth_eef_min_duration_s,
        max_duration_s=args.smooth_eef_max_duration_s,
    )
    return config


def build_smoother(args: argparse.Namespace) -> BimanualActionSmoother | None:
    if args.no_smooth:
        return None
    return BimanualActionSmoother(
        action_mode=args.action_mode,
        max_joint_delta_rad=args.max_joint_delta_rad,
        max_eef_linear_delta_m=args.max_eef_linear_delta_m,
        max_eef_angular_delta_rad=args.max_eef_angular_delta_rad,
        max_delta_eef_linear_m=args.max_delta_eef_linear_m,
        max_delta_eef_angular_rad=args.max_delta_eef_angular_rad,
        max_gripper_delta=args.max_gripper_delta,
        lowpass_alpha=args.lowpass_alpha,
    )


def read_timed_observation(env: DualPiperXEnv) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    obs: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    timestamps = []

    arm_t0 = time.monotonic()
    for side, arm in (("left", env.left_arm), ("right", env.right_arm)):
        state = arm.read_state()
        obs[f"{side}_joint_pos"] = state.joint_pos.copy()
        obs[f"{side}_eef_pos"] = state.eef_pos.copy()
        obs[f"{side}_joint_qvel"] = state.joint_qvel.copy()
        obs[f"{side}_joint_effort"] = state.joint_effort.copy()
        obs[f"{side}_ctrl_mode"] = np.array([-1 if state.ctrl_mode is None else state.ctrl_mode], dtype=np.int32)
        timestamps.append(state.timestamp)
    metrics["arm_ms"] = (time.monotonic() - arm_t0) * 1000.0

    camera_t0 = time.monotonic()
    for name, camera in env.cameras.cameras.items():
        one_t0 = time.monotonic()
        obs[f"{name}_color"] = camera.read()
        metrics[f"{name}_ms"] = (time.monotonic() - one_t0) * 1000.0
    metrics["camera_ms"] = (time.monotonic() - camera_t0) * 1000.0
    metrics["obs_ms"] = metrics["arm_ms"] + metrics["camera_ms"]
    metrics["camera_fps"] = 1000.0 / metrics["camera_ms"] if metrics["camera_ms"] > 0 else 0.0
    obs["timestamp"] = np.array([time.time() if not timestamps else max(timestamps)], dtype=np.float64)
    return obs, metrics


def run_timing_profile(
    env: DualPiperXEnv,
    policy: OpenPIRemotePolicy,
    args: argparse.Namespace,
    smoother: BimanualActionSmoother | None,
    max_steps: int | None,
) -> None:
    if args.execute and not args.allow_teaching_mode and hasattr(env, "guard_can_accept_motion"):
        env.guard_can_accept_motion()

    dt = 1.0 / args.hz
    previous_action: dict[str, np.ndarray | None] | None = None
    previous_raw_action: np.ndarray | None = None
    step = 0

    try:
        while max_steps is None or step < max_steps:
            loop_t0 = time.monotonic()

            obs, obs_metrics = read_timed_observation(env)
            state = default_state(obs)
            images = default_images(obs)

            infer_t0 = time.monotonic()
            raw_action = predict_with_policy(policy, images, state)
            infer_ms = (time.monotonic() - infer_t0) * 1000.0
            raw_jump = (
                0.0
                if previous_raw_action is None
                else float(np.linalg.norm(np.asarray(raw_action, dtype=np.float32)[:14] - previous_raw_action[:14]))
            )

            action = split_bimanual_action(raw_action)
            if smoother is not None:
                action = smoother.filter(action, obs, previous_action)

            control_t0 = time.monotonic()
            if args.execute:
                env.step(action, action_mode=args.action_mode, return_observation=False)
            control_ms = (time.monotonic() - control_t0) * 1000.0

            elapsed_s = time.monotonic() - loop_t0
            sleep_s = dt - elapsed_s
            sleep_ms = max(0.0, sleep_s) * 1000.0
            loop_ms = elapsed_s * 1000.0

            if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0):
                source = getattr(policy, "last_predict_source", "policy")
                camera_parts = " ".join(
                    f"{key}={value:.1f}"
                    for key, value in obs_metrics.items()
                    if key.endswith("_ms") and key not in {"obs_ms", "arm_ms", "camera_ms"}
                )
                line = (
                    f"step={step + 1:06d} src={source} "
                    f"loop_ms={loop_ms:.1f} obs_ms={obs_metrics['obs_ms']:.1f} "
                    f"arm_ms={obs_metrics['arm_ms']:.1f} camera_ms={obs_metrics['camera_ms']:.1f} "
                    f"infer_ms={infer_ms:.1f} control_ms={control_ms:.1f} sleep_ms={sleep_ms:.1f} "
                    f"camera_fps={obs_metrics['camera_fps']:.1f} raw_jump={raw_jump:.4f}"
                )
                if camera_parts:
                    line = f"{line} {camera_parts}"
                if args.profile_print_actions:
                    left = action["left"]
                    right = action["right"]
                    line = (
                        f"{line} "
                        f"left={np.array2string(left, precision=4, suppress_small=True) if left is not None else None} "
                        f"right={np.array2string(right, precision=4, suppress_small=True) if right is not None else None}"
                    )
                print(line)

            previous_action = action
            previous_raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1).copy()
            step += 1
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        return


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()

    policy = OpenPIRemotePolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        prompt=args.prompt,
        observation_format=args.observation_format,
        resize=args.image_resize,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        exec_chunk_size=args.exec_chunk_size,
    )
    metadata = policy.get_server_metadata()
    if metadata:
        print(f"OpenPI server metadata: {metadata}")

    env = DualPiperXEnv(build_env_config(args))
    max_steps = resolve_max_steps(args)
    smoother = build_smoother(args)

    print(
        f"action_mode={args.action_mode} chunk_size={args.chunk_size} "
        f"exec_chunk_size={args.exec_chunk_size or args.chunk_size} hz={args.hz} "
        f"host={args.host}:{args.port} smooth={'off' if smoother is None else 'on'}"
    )
    print("DRY-RUN: actions will not be sent." if not args.execute else "EXECUTE: actions will be sent to the robot.")

    try:
        if args.profile_timing:
            run_timing_profile(env, policy, args, smoother=smoother, max_steps=max_steps)
        else:
            runner = PolicyRunner(
                env,
                policy=policy,
                action_mode=args.action_mode,
                hz=args.hz,
                guard_motion_mode=not args.allow_teaching_mode,
                smoother=smoother,
                smooth_actions=smoother is not None,
                execute=args.execute,
                print_every=args.print_every,
            )
            runner.run(max_steps=max_steps)
    except KeyboardInterrupt:
        print("Interrupted, stopping deployment.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
