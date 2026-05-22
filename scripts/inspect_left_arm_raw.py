#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def to_plain(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_plain(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {str(k): to_plain(v, depth + 1) for k, v in vars(value).items() if not str(k).startswith("_")}
    return repr(value)


def safe_call(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(obj, name)
    try:
        return method(*args, **kwargs)
    except TypeError:
        if kwargs:
            return method(*args)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump raw Piper SDK messages for one arm.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--out", default="sdk_left_raw.json")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--set-motion-output-role", action="store_true")
    parser.add_argument("--set-teaching-input-role", action="store_true")
    args = parser.parse_args()

    from piper_sdk import C_PiperInterface_V2

    arm = C_PiperInterface_V2(can_name=args.can, judge_flag=False, can_auto_init=True)
    safe_call(arm, "ConnectPort")
    if args.set_motion_output_role:
        arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
    if args.set_teaching_input_role:
        arm.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
    time.sleep(max(0.0, args.sleep))

    methods = [
        "isOk",
        "GetArmStatus",
        "GetArmJointMsgs",
        "GetArmGripperMsgs",
        "GetArmEndPoseMsgs",
        "GetArmJointCtrl",
        "GetArmGripperCtrl",
        "GetPiperFirmwareVersion",
        "GetAllMotorAngleLimitMaxSpd",
        "GetAllMotorMaxAccLimit",
    ]
    snapshot: dict[str, Any] = {"can": args.can}
    for name in methods:
        if not hasattr(arm, name):
            continue
        try:
            value = getattr(arm, name)()
            snapshot[name] = {"repr": repr(value), "plain": to_plain(value)}
        except Exception as exc:
            snapshot[name] = {"error": repr(exc)}

    Path(args.out).write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
