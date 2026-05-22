#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piperx_toolkit import DualPiperXEnv
from piperx_toolkit.env.dual_piper_env import DualPiperXEnvConfig


def main() -> None:
    env = DualPiperXEnv(DualPiperXEnvConfig(backend="mock"), auto_connect=True)
    try:
        obs = env.reset()
        print("obs keys:", sorted(obs.keys()))
        joint = np.array([0.0, 0.1, -0.1, 0.0, 0.2, 0.0, 0.5], dtype=np.float32)
        obs = env.step({"left": joint, "right": -joint}, action_mode="absolute_joint")
        print("left_joint_pos:", obs["left_joint_pos"])
        eef = np.array([0.06, 0.0, 0.22, 0.0, 1.4, 0.0, 0.2], dtype=np.float32)
        env.step({"left": eef, "right": eef}, action_mode="absolute_eef", return_observation=False)
        env.step({"left": eef + np.array([0, 0, 0.01, 0, 0, 0, 0], dtype=np.float32), "right": None}, action_mode="smooth_eef", return_observation=False)
        env.step({"left": np.array([0, 0, -0.01, 0, 0, 0, 0.1], dtype=np.float32), "right": None}, action_mode="delta_eef")
        print("mock env OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
