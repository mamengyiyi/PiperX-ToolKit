from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

ActionMode = Literal["absolute_joint", "absolute_eef", "smooth_eef", "delta_eef"]
Side = Literal["left", "right"]
Backend = Literal["sdk", "mock"]


class DualArmAction(TypedDict):
    left: np.ndarray | None
    right: np.ndarray | None

