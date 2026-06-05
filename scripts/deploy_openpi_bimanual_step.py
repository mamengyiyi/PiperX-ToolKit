#!/usr/bin/env python3
"""Step-wise OpenPI bimanual deployment with OpenCV camera confirmation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.deploy.openpi_remote import OpenPIRemotePolicy
from piperx_toolkit.deploy.policy_runner import (
    BimanualActionSmoother,
    default_images,
    default_state,
    split_bimanual_action,
)
from piperx_toolkit.utils.logging import setup_logging
from scripts.deploy_openpi_bimanual import ACTION_MODES, build_env_config, build_smoother

CAMERA_ORDER = ("front", "left_wrist", "right_wrist")
HIGHGUI_ERROR_MARKERS = (
    "The function is not implemented",
    "cvShowImage",
    "cvDestroyAllWindows",
    "GTK+",
    "Cocoa support",
)


def camera_device_arg(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step through OpenPI bimanual chunks after visually confirming camera frames.",
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
    openpi.add_argument("--observation-format", default="piperx", choices=["aloha", "piperx"])
    openpi.add_argument("--image-resize", type=int, default=224, help="Policy image resize (square).")
    openpi.add_argument("--action-dim", type=int, default=14, help="Bimanual action dimension.")
    openpi.add_argument("--chunk-size", type=int, default=10, help="Actions to keep from each OpenPI response chunk.")

    control = parser.add_argument_group("step control")
    control.add_argument("--action-mode", default="absolute_joint", choices=ACTION_MODES)
    control.add_argument("--hz", type=float, default=5.0, help="Execution frequency inside a confirmed chunk.")
    control.add_argument("--max-chunks", type=int, default=None, help="Stop after this many confirmed chunks.")
    control.add_argument("--execute", action="store_true", help="Actually send selected actions to the robot.")
    control.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Number of actions to execute from each confirmed chunk. 0 = infer and print only.",
    )
    control.add_argument(
        "--allow-teaching-mode",
        action="store_true",
        help="Skip ctrl_mode guard. For mock/debug only.",
    )

    display = parser.add_argument_group("opencv display")
    display.add_argument("--window-name", default="OpenPI step cameras")
    display.add_argument("--display-width", type=int, default=480, help="Per-camera display width.")
    display.add_argument("--wait-ms", type=int, default=30, help="OpenCV waitKey delay while refreshing preview.")
    display.add_argument(
        "--camera-save-dir",
        default="openpi_step_frames",
        help="Directory where confirmed chunk camera frames are saved as chunk_*/camera.png.",
    )
    display.add_argument(
        "--no-save-camera-frames",
        action="store_true",
        help="Disable saving confirmed camera frames to disk.",
    )

    report = parser.add_argument_group("chunk diagnostics")
    report.add_argument("--tail-fraction", type=float, default=0.5, help="Fraction of previous chunk tail to print.")
    report.add_argument("--print-rows", type=int, default=5, help="Rows to print from each chunk head/tail section.")
    report.add_argument(
        "--print-full-current-chunk",
        action="store_true",
        help="Print all rows from current chunk instead of compact head/tail rows.",
    )

    smooth = parser.add_argument_group("action smoothing")
    smooth.add_argument("--no-smooth", action="store_true", help="Disable all action smoothing/limiting.")
    smooth.add_argument("--lowpass-alpha", type=float, default=0.3, help="EMA weight on new action (1 = off).")
    smooth.add_argument("--max-joint-delta-rad", type=float, default=0.01, help="Per-step joint limit.")
    smooth.add_argument("--max-eef-linear-delta-m", type=float, default=0.03, help="Per-step EEF position limit.")
    smooth.add_argument("--max-eef-angular-delta-rad", type=float, default=0.15, help="Per-step EEF orientation limit.")
    smooth.add_argument("--max-delta-eef-linear-m", type=float, default=0.02, help="Limit policy delta EEF translation.")
    smooth.add_argument("--max-delta-eef-angular-rad", type=float, default=0.10, help="Limit policy delta EEF rotation.")
    smooth.add_argument("--max-gripper-delta", type=float, default=0.02, help="Per-step gripper change limit.")

    seef = parser.add_argument_group("smooth_eef env interpolation (action-mode=smooth_eef)")
    seef.add_argument("--smooth-eef-hz", type=float, default=60.0)
    seef.add_argument("--smooth-eef-linear-speed-m-s", type=float, default=0.15)
    seef.add_argument("--smooth-eef-angular-speed-rad-s", type=float, default=0.7)
    seef.add_argument("--smooth-eef-gripper-speed-s", type=float, default=1.0)
    seef.add_argument("--smooth-eef-min-duration-s", type=float, default=0.08)
    seef.add_argument("--smooth-eef-max-duration-s", type=float, default=5.0)

    return parser


def chunk_step_jumps(chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(arr[1:, :14] - arr[:-1, :14], axis=1).astype(np.float32)


def _chunk_tail(chunk: np.ndarray, fraction: float) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr.reshape(0, arr.shape[-1] if arr.ndim > 1 else 0)
    rows = max(1, int(np.ceil(arr.shape[0] * float(np.clip(fraction, 0.0, 1.0)))))
    return arr[-rows:]


def _array_block(name: str, arr: np.ndarray, print_rows: int, full: bool = False) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return f"{name} shape={arr.shape}\n[]"
    if full or arr.shape[0] <= print_rows * 2:
        shown = arr
    else:
        shown = np.concatenate([arr[:print_rows], arr[-print_rows:]], axis=0)
    return f"{name} shape={arr.shape}\n{np.array2string(shown, precision=4, suppress_small=True)}"


def is_highgui_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in HIGHGUI_ERROR_MARKERS)


def format_chunk_report(
    step_index: int,
    state: np.ndarray,
    current_chunk: np.ndarray,
    previous_chunk: np.ndarray | None,
    previous_state: np.ndarray | None = None,
    tail_fraction: float = 0.5,
    print_rows: int = 5,
    print_full_current_chunk: bool = False,
) -> str:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    current = np.asarray(current_chunk, dtype=np.float32)
    previous = None if previous_chunk is None else np.asarray(previous_chunk, dtype=np.float32)
    jumps = chunk_step_jumps(current)
    boundary_jump = float("nan")
    state_jump = float("nan")
    first_action_from_state = float("nan")
    prev_tail = np.zeros((0, current.shape[1] if current.ndim == 2 else 0), dtype=np.float32)
    if previous is not None and previous.size > 0 and current.size > 0:
        prev_tail = _chunk_tail(previous, tail_fraction)
        boundary_jump = float(np.linalg.norm(current[0, :14] - previous[-1, :14]))
    if previous_state is not None:
        prev_state = np.asarray(previous_state, dtype=np.float32).reshape(-1)
        if prev_state.shape[0] >= 14 and state.shape[0] >= 14:
            state_jump = float(np.linalg.norm(state[:14] - prev_state[:14]))
    if current.size > 0 and state.shape[0] >= 14:
        first_action_from_state = float(np.linalg.norm(current[0, :14] - state[:14]))

    jump_summary = "empty"
    if jumps.size:
        jump_summary = f"min={float(jumps.min()):.4f} mean={float(jumps.mean()):.4f} max={float(jumps.max()):.4f}"

    lines = [
        f"chunk_step={step_index}",
        f"state_left={np.array2string(state[:7], precision=4, suppress_small=True)}",
        f"state_right={np.array2string(state[7:14], precision=4, suppress_small=True)}",
        f"state_jump={state_jump:.4f}",
        f"first_action_from_state={first_action_from_state:.4f}",
        f"boundary_jump={boundary_jump:.4f}",
        f"current_step_jumps {jump_summary}",
        _array_block("prev_chunk_tail", prev_tail, print_rows=print_rows),
        _array_block(
            "current_chunk",
            current,
            print_rows=print_rows,
            full=print_full_current_chunk,
        ),
    ]
    return "\n".join(lines)


def make_camera_mosaic(obs: dict[str, np.ndarray], display_width: int) -> np.ndarray:
    import cv2

    panels = []
    for name in CAMERA_ORDER:
        key = f"{name}_color"
        if key not in obs:
            continue
        rgb = np.asarray(obs[key])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = display_width / max(1, bgr.shape[1])
        height = max(1, int(round(bgr.shape[0] * scale)))
        panel = cv2.resize(bgr, (display_width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        return np.zeros((360, display_width, 3), dtype=np.uint8)
    target_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < target_h:
            pad = np.zeros((target_h - panel.shape[0], panel.shape[1], 3), dtype=panel.dtype)
            panel = np.concatenate([panel, pad], axis=0)
        padded.append(panel)
    mosaic = np.concatenate(padded, axis=1)
    cv2.putText(
        mosaic,
        "Enter: infer/execute   q/Esc: quit",
        (12, target_h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return mosaic


def save_camera_frames(obs: dict[str, np.ndarray], output_root: Path | str, chunk_step: int) -> list[Path]:
    import cv2

    chunk_dir = Path(output_root) / f"chunk_{chunk_step:06d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name in CAMERA_ORDER:
        key = f"{name}_color"
        if key not in obs:
            continue
        rgb = np.asarray(obs[key])
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = chunk_dir / f"{name}.png"
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to save camera frame: {path}")
        saved.append(path)
    if saved:
        mosaic_path = chunk_dir / "mosaic.png"
        if cv2.imwrite(str(mosaic_path), make_camera_mosaic(obs, display_width=480)):
            saved.append(mosaic_path)
    return saved


def wait_for_confirmed_observation(
    env: DualPiperXEnv,
    window_name: str,
    display_width: int,
    wait_ms: int,
) -> dict[str, np.ndarray] | None:
    import cv2

    while True:
        obs = env.get_observation()
        try:
            cv2.imshow(window_name, make_camera_mosaic(obs, display_width=display_width))
            key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        except cv2.error as exc:
            if not is_highgui_error(exc):
                raise
            preview_path = "/tmp/openpi_step_cameras.jpg"
            cv2.imwrite(preview_path, make_camera_mosaic(obs, display_width=display_width))
            print(
                "OpenCV GUI is unavailable in this Python environment; "
                f"saved camera preview to {preview_path}."
            )
            text = input("Press Enter to infer with this observation, or type q then Enter to quit: ")
            return None if text.strip().lower() in {"q", "quit", "exit"} else obs
        if key in (10, 13):
            return obs
        if key in (27, ord("q")):
            return None


def infer_chunk(policy: OpenPIRemotePolicy, images: dict[str, np.ndarray], state: np.ndarray, chunk_size: int) -> np.ndarray:
    response = policy.client.infer(policy.make_observation(images, state))
    actions = policy._split_actions(response)
    if chunk_size > 0:
        actions = actions[:chunk_size]
    if not actions:
        raise ValueError("OpenPI server returned no actions")
    return np.stack(actions, axis=0).astype(np.float32)


def execute_chunk_steps(
    env: DualPiperXEnv,
    chunk: np.ndarray,
    args: argparse.Namespace,
    smoother: BimanualActionSmoother | None,
    initial_obs: dict[str, np.ndarray],
    previous_action: dict[str, np.ndarray | None] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | None] | None, int]:
    obs = initial_obs
    last_action = previous_action
    steps = min(max(0, args.execute_steps), chunk.shape[0])
    dt = 1.0 / max(args.hz, 1e-6)
    for idx, row in enumerate(chunk[:steps]):
        loop_t0 = time.monotonic()
        action = split_bimanual_action(row)
        if smoother is not None:
            action = smoother.filter(action, obs, last_action)
        env.step(action, action_mode=args.action_mode, return_observation=False)
        last_action = action
        print(f"executed_action_index={idx} left={action['left']} right={action['right']}")
        sleep_s = dt - (time.monotonic() - loop_t0)
        if sleep_s > 0:
            time.sleep(sleep_s)
        obs = env.get_observation()
    return obs, last_action, steps


def run_step_deploy(env: DualPiperXEnv, policy: OpenPIRemotePolicy, args: argparse.Namespace) -> None:
    import cv2

    if args.execute and not args.allow_teaching_mode:
        env.guard_can_accept_motion()

    smoother = build_smoother(args)
    previous_chunk: np.ndarray | None = None
    previous_state: np.ndarray | None = None
    previous_action: dict[str, np.ndarray | None] | None = None
    chunk_step = 0
    print("OpenCV step mode: press Enter in the camera window to infer; press q/Esc to quit.")
    try:
        while args.max_chunks is None or chunk_step < args.max_chunks:
            obs = wait_for_confirmed_observation(
                env,
                window_name=args.window_name,
                display_width=args.display_width,
                wait_ms=args.wait_ms,
            )
            if obs is None:
                break
            if not args.no_save_camera_frames:
                saved_paths = save_camera_frames(obs, args.camera_save_dir, chunk_step)
                if saved_paths:
                    print(f"saved_camera_frames={saved_paths[0].parent}")

            state = default_state(obs)
            images = default_images(obs)
            infer_t0 = time.monotonic()
            current_chunk = infer_chunk(policy, images, state, chunk_size=args.chunk_size)
            infer_ms = (time.monotonic() - infer_t0) * 1000.0
            print(f"\n=== OpenPI chunk {chunk_step} infer_ms={infer_ms:.1f} ===")
            print(
                format_chunk_report(
                    step_index=chunk_step,
                    state=state,
                    current_chunk=current_chunk,
                    previous_chunk=previous_chunk,
                    previous_state=previous_state,
                    tail_fraction=args.tail_fraction,
                    print_rows=args.print_rows,
                    print_full_current_chunk=args.print_full_current_chunk,
                )
            )

            if args.execute and args.execute_steps > 0:
                obs, previous_action, executed = execute_chunk_steps(
                    env,
                    current_chunk,
                    args,
                    smoother=smoother,
                    initial_obs=obs,
                    previous_action=previous_action,
                )
                print(f"executed_steps={executed}/{current_chunk.shape[0]}")
            else:
                print("DRY-RUN: chunk was not executed.")

            previous_chunk = current_chunk
            previous_state = state.copy()
            chunk_step += 1
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            if not is_highgui_error(exc):
                raise


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
        chunk_size=None,
    )
    metadata: dict[str, Any] = policy.get_server_metadata()
    if metadata:
        print(f"OpenPI server metadata: {metadata}")

    env = DualPiperXEnv(build_env_config(args))
    print(
        f"step_mode action_mode={args.action_mode} chunk_size={args.chunk_size} hz={args.hz} "
        f"host={args.host}:{args.port} execute_steps={args.execute_steps}"
    )
    print("DRY-RUN: actions will not be sent." if not args.execute else "EXECUTE: selected actions will be sent.")
    try:
        run_step_deploy(env, policy, args)
    except KeyboardInterrupt:
        print("Interrupted, stopping step deployment.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
