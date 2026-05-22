from __future__ import annotations

import time
from types import SimpleNamespace


class MockPiperInterface:
    """Small Piper SDK stand-in used for offline tests."""

    def __init__(self, can_name: str = "mock", **_: object):
        self.can_name = can_name
        self.connected = False
        self.enabled = False
        self.ctrl_mode = 0x01
        self.motion_mode = (0x01, 0x01, 100, 0x00)
        self.joint_raw = [0, 0, 0, 0, 0, 0]
        self.gripper_raw = 70_000
        self.eef_raw = [57_000, 0, 215_000, 0, 85_000, 0]

    def ConnectPort(self, *args: object, **kwargs: object) -> bool:
        self.connected = True
        return True

    def DisconnectPort(self) -> bool:
        self.connected = False
        return True

    def EnablePiper(self) -> bool:
        self.enabled = True
        return True

    def DisableArm(self, *_: object) -> bool:
        self.enabled = False
        return True

    def MotionCtrl_1(self, mode: int, *_: object) -> bool:
        self.ctrl_mode = int(mode)
        return True

    def MotionCtrl_2(self, *args: object) -> bool:
        self.motion_mode = tuple(args)
        return True

    def JointCtrl(self, *joints: int) -> bool:
        self.joint_raw = [int(v) for v in joints[:6]]
        return True

    def EndPoseCtrl(self, *pose: int) -> bool:
        self.eef_raw = [int(v) for v in pose[:6]]
        return True

    def GripperCtrl(self, grippers_angle: int, *_: object) -> bool:
        self.gripper_raw = abs(int(grippers_angle))
        return True

    def MasterSlaveConfig(self, role: int, *_: object) -> bool:
        self.ctrl_mode = 0x02 if int(role) == 0xFA else 0x01
        return True

    def GripperTeachingPendantParamConfig(self, *_: object) -> bool:
        return True

    def ArmParamEnquiryAndConfig(self, *_: object) -> bool:
        return True

    def GetGripperTeachingPendantParamFeedback(self) -> SimpleNamespace:
        return SimpleNamespace(time_stamp=time.time(), max_range_config=70)

    def GetArmStatus(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            arm_status=SimpleNamespace(ctrl_mode=self.ctrl_mode, err_code=0),
        )

    def GetArmJointMsgs(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            joint_state=SimpleNamespace(
                joint_1=self.joint_raw[0],
                joint_2=self.joint_raw[1],
                joint_3=self.joint_raw[2],
                joint_4=self.joint_raw[3],
                joint_5=self.joint_raw[4],
                joint_6=self.joint_raw[5],
            ),
        )

    def GetArmJointCtrl(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            joint_ctrl=SimpleNamespace(
                joint_1=self.joint_raw[0],
                joint_2=self.joint_raw[1],
                joint_3=self.joint_raw[2],
                joint_4=self.joint_raw[3],
                joint_5=self.joint_raw[4],
                joint_6=self.joint_raw[5],
            ),
        )

    def GetArmEndPoseMsgs(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            end_pose=SimpleNamespace(
                X_axis=self.eef_raw[0],
                Y_axis=self.eef_raw[1],
                Z_axis=self.eef_raw[2],
                RX_axis=self.eef_raw[3],
                RY_axis=self.eef_raw[4],
                RZ_axis=self.eef_raw[5],
            ),
        )

    def GetArmGripperMsgs(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            gripper_state=SimpleNamespace(grippers_angle=self.gripper_raw, status_code=0x01),
        )

    def GetArmGripperCtrl(self) -> SimpleNamespace:
        return SimpleNamespace(
            time_stamp=time.time(),
            gripper_ctrl=SimpleNamespace(grippers_angle=self.gripper_raw),
        )

