from __future__ import annotations

from typing import Protocol

import numpy as np

from piperx_toolkit.types import DualArmAction


class TeleopSource(Protocol):
    action_mode: str

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_action(self, obs: dict[str, np.ndarray]) -> DualArmAction:
        ...


def zero_action() -> DualArmAction:
    return {"left": np.zeros(7, dtype=np.float32), "right": np.zeros(7, dtype=np.float32)}

