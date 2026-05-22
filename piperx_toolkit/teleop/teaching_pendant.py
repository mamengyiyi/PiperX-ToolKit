from __future__ import annotations

import logging

import numpy as np

from piperx_toolkit.types import DualArmAction

logger = logging.getLogger(__name__)


class TeachingPendantTeleop:
    """Read bimanual arm states as a teaching action source.

    This class does not send motion commands. In collection, the collector may
    shift these joint positions forward by one or more frames to create
    imitation-learning targets.
    """

    action_mode = "absolute_joint"

    def __init__(
        self,
        env,
        configure_roles: bool = False,
        set_motion_output_role: bool = False,
        configure_gripper_params: bool = False,
    ):
        self.env = env
        self.configure_roles = configure_roles
        self.set_motion_output_role = set_motion_output_role
        self.configure_gripper_params = configure_gripper_params
        self.running = False

    def start(self) -> None:
        if self.configure_gripper_params:
            logger.info("Configuring gripper/teaching-pendant parameters.")
            self.env.configure_gripper_teaching_pendants()
        if self.set_motion_output_role:
            logger.info("Setting both arms to motion/slave output role.")
            self.env.set_motion_output_role()
        if self.configure_roles:
            logger.warning(
                "Setting both arms to teaching/master input role. Piper SDK ordinary feedback "
                "may return zeros in this role; prefer set_motion_output_role for data collection."
            )
            self.env.set_teaching_input_role()
        self.running = True

    def stop(self) -> None:
        self.running = False

    def get_action(self, obs: dict[str, np.ndarray]) -> DualArmAction:
        left = obs.get("left_joint_pos")
        right = obs.get("right_joint_pos")
        return {
            "left": None if left is None else np.asarray(left, dtype=np.float32).copy(),
            "right": None if right is None else np.asarray(right, dtype=np.float32).copy(),
        }

