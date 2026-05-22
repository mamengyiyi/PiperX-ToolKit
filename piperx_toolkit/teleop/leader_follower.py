from __future__ import annotations


class LeaderFollowerTeleop:
    """Reserved leader-follower teleoperation extension point."""

    action_mode = "absolute_joint"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LeaderFollowerTeleop is reserved for the next hardware-backed iteration. "
            "Use TeachingPendantTeleop for the first data-collection workflow."
        )

