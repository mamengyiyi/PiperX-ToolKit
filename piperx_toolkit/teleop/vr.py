from __future__ import annotations


class VRTeleop:
    """Reserved VR teleoperation extension point."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VRTeleop is reserved for a future VR SDK/ROS/WebSocket integration. "
            "Use TeachingPendantTeleop for the first data-collection workflow."
        )

