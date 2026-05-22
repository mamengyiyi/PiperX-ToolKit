from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from piperx_toolkit.collect.schema import (
    DEFAULT_CAMERAS,
    ZarrSchemaConfig,
    append_episode,
    open_or_create_dataset,
)
from piperx_toolkit.teleop.teaching_pendant import TeachingPendantTeleop

logger = logging.getLogger(__name__)


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
class EpisodeStats:
    steps: int
    duration_s: float
    fps: float


class Collector:
    def __init__(
        self,
        env,
        dataset_path: str,
        num_episodes: int = 1,
        hz: float = 30.0,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        image_size: tuple[int, int] = (640, 480),
        task: str = "",
        action_mode: str = "absolute_joint",
        action_shift_frames: int = 1,
        teleop_source: Any | None = None,
    ):
        self.env = env
        self.dataset_path = dataset_path
        self.num_episodes = num_episodes
        self.hz = hz
        self.cameras = cameras
        self.image_size = image_size
        self.task = task
        self.action_mode = action_mode
        self.action_shift_frames = action_shift_frames
        self.teleop_source = teleop_source or TeachingPendantTeleop(env)

    def run(self) -> None:
        config = ZarrSchemaConfig(
            cameras=self.cameras,
            image_size=self.image_size,
            task=self.task,
            hz=self.hz,
            action_mode=self.action_mode,
            action_shift_frames=self.action_shift_frames,
        )
        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        data, meta, start_episode = open_or_create_dataset(self.dataset_path, config)
        if start_episode >= self.num_episodes:
            print(f"Dataset already has {start_episode} episodes. Nothing to collect.")
            return

        self.teleop_source.start()
        kb = KeyboardListener()
        saved = 0
        try:
            kb.start()
            for episode_id in range(start_episode, self.num_episodes):
                stats, episode = self._record_episode_interactive(episode_id, kb)
                if episode is None:
                    continue
                print(
                    f"\r\nEpisode {episode_id}: {stats.steps} steps, "
                    f"{stats.duration_s:.1f}s, {stats.fps:.1f} FPS. [S] save / [D] discard\r\n"
                )
                while True:
                    key = kb.get_key()
                    if key == "s":
                        append_episode(data, meta, episode)
                        saved += 1
                        print(f"\r\nSaved episode {episode_id}.\r\n")
                        break
                    if key == "d":
                        print(f"\r\nDiscarded episode {episode_id}.\r\n")
                        break
                    if key == "\x03":
                        raise KeyboardInterrupt
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\r\nInterrupted, closing collector.\r\n")
        finally:
            kb.stop()
            self.teleop_source.stop()
            print(f"Collection done. Saved {saved} new episode(s) to {self.dataset_path}.")

    def collect_fixed_duration(self, duration_s: float, episode_id: int = 0) -> EpisodeStats:
        config = ZarrSchemaConfig(
            cameras=self.cameras,
            image_size=self.image_size,
            task=self.task,
            hz=self.hz,
            action_mode=self.action_mode,
            action_shift_frames=self.action_shift_frames,
        )
        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        data, meta, start_episode = open_or_create_dataset(self.dataset_path, config)
        ep_id = start_episode if start_episode > episode_id else episode_id
        self.teleop_source.start()
        try:
            stats, episode = self._record_episode_for_duration(ep_id, duration_s)
            append_episode(data, meta, episode)
            return stats
        finally:
            self.teleop_source.stop()

    def _record_episode_interactive(self, episode_id: int, kb: KeyboardListener) -> tuple[EpisodeStats, dict[str, np.ndarray] | None]:
        print(
            f"\r\nEpisode {episode_id}: Space start, Enter stop, Ctrl+C quit. "
            "Move both PiperX arms by hand in teaching/master input mode.\r\n"
        )
        while True:
            key = kb.get_key()
            if key == " ":
                break
            if key == "\x03":
                raise KeyboardInterrupt
            time.sleep(0.05)
        return self._record_until_enter(episode_id, kb)

    def _record_until_enter(self, episode_id: int, kb: KeyboardListener) -> tuple[EpisodeStats, dict[str, np.ndarray] | None]:
        buffer = self._empty_buffer()
        dt = 1.0 / self.hz
        t0 = time.time()
        steps = 0
        print(f"\r\nRecording episode {episode_id}. Press Enter to stop.\r\n")
        while True:
            loop_t = time.monotonic()
            key = kb.get_key()
            if key in ("\r", "\n"):
                break
            if key == "\x03":
                raise KeyboardInterrupt
            self._capture_step(buffer, episode_id)
            steps += 1
            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)
        if steps == 0:
            return EpisodeStats(0, 0.0, 0.0), None
        return self._finalize_buffer(buffer, t0, steps)

    def _record_episode_for_duration(self, episode_id: int, duration_s: float) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
        buffer = self._empty_buffer()
        dt = 1.0 / self.hz
        t0 = time.time()
        steps = 0
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            loop_t = time.monotonic()
            self._capture_step(buffer, episode_id)
            steps += 1
            sleep_s = dt - (time.monotonic() - loop_t)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return self._finalize_buffer(buffer, t0, steps)

    def _empty_buffer(self) -> dict[str, list[np.ndarray]]:
        buffer: dict[str, list[np.ndarray]] = {
            "left_eef_pos": [],
            "left_joint_pos": [],
            "left_joint_qvel": [],
            "left_joint_effort": [],
            "right_eef_pos": [],
            "right_joint_pos": [],
            "right_joint_qvel": [],
            "right_joint_effort": [],
            "action_left": [],
            "action_right": [],
            "timestamp": [],
            "episode": [],
        }
        for cam in self.cameras:
            buffer[f"rgb_{cam}"] = []
        return buffer

    def _capture_step(self, buffer: dict[str, list[np.ndarray]], episode_id: int) -> None:
        obs = self.env.get_observation(include_arm=True, include_camera=True)
        teleop_action = self.teleop_source.get_action(obs)
        width, height = self.image_size

        for key in (
            "left_eef_pos",
            "left_joint_pos",
            "left_joint_qvel",
            "left_joint_effort",
            "right_eef_pos",
            "right_joint_pos",
            "right_joint_qvel",
            "right_joint_effort",
        ):
            buffer[key].append(np.asarray(obs.get(key, np.zeros(7)), dtype=np.float32).reshape(1, 7))

        buffer["action_left"].append(
            np.asarray(teleop_action.get("left") if teleop_action.get("left") is not None else np.zeros(7), dtype=np.float32).reshape(1, 7)
        )
        buffer["action_right"].append(
            np.asarray(teleop_action.get("right") if teleop_action.get("right") is not None else np.zeros(7), dtype=np.float32).reshape(1, 7)
        )

        for cam in self.cameras:
            image = obs.get(f"{cam}_color")
            if image is None:
                image = np.zeros((height, width, 3), dtype=np.uint8)
            image = self._ensure_rgb_size(np.asarray(image), width=width, height=height)
            buffer[f"rgb_{cam}"].append(image.transpose(2, 0, 1)[None])

        timestamp = float(obs.get("timestamp", np.array([time.time()]))[0])
        buffer["timestamp"].append(np.array([timestamp], dtype=np.float64))
        buffer["episode"].append(np.array([episode_id], dtype=np.uint32))

    def _finalize_buffer(self, buffer: dict[str, list[np.ndarray]], t0: float, steps: int) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
        episode = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
        if self.action_mode == "absolute_joint" and self.action_shift_frames > 0:
            shift = self.action_shift_frames
            for side in ("left", "right"):
                state_key = f"{side}_joint_pos"
                action_key = f"action_{side}"
                source = episode[state_key]
                shifted = source.copy()
                if len(source) > shift:
                    shifted[:-shift] = source[shift:]
                    shifted[-shift:] = source[-1]
                episode[action_key] = shifted.astype(np.float32)
        duration = time.time() - t0
        stats = EpisodeStats(steps=steps, duration_s=duration, fps=steps / max(duration, 1e-6))
        return stats, episode

    @staticmethod
    def _ensure_rgb_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {image.shape}")
        if image.shape[0] == height and image.shape[1] == width:
            return image
        try:
            import cv2

            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except ModuleNotFoundError as exc:
            raise RuntimeError("opencv-python is required to resize camera images") from exc
