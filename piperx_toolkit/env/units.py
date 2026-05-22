from __future__ import annotations

import math

import numpy as np

JOINT_RAW_PER_RAD = 1000.0 * 180.0 / math.pi
RAD_PER_JOINT_RAW = math.pi / (180.0 * 1000.0)
LINEAR_RAW_PER_METER = 1_000_000.0
METER_PER_LINEAR_RAW = 1.0 / LINEAR_RAW_PER_METER
ANGLE_RAW_PER_RAD = JOINT_RAW_PER_RAD
RAD_PER_ANGLE_RAW = RAD_PER_JOINT_RAW


def rad_to_joint_raw(values: np.ndarray | list[float] | tuple[float, ...]) -> list[int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [int(round(v * JOINT_RAW_PER_RAD)) for v in arr]


def joint_raw_to_rad(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return (arr * RAD_PER_JOINT_RAW).astype(np.float32)


def meters_to_linear_raw(values: np.ndarray | list[float] | tuple[float, ...]) -> list[int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [int(round(v * LINEAR_RAW_PER_METER)) for v in arr]


def linear_raw_to_meters(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return (arr * METER_PER_LINEAR_RAW).astype(np.float32)


def rad_to_angle_raw(values: np.ndarray | list[float] | tuple[float, ...]) -> list[int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [int(round(v * ANGLE_RAW_PER_RAD)) for v in arr]


def angle_raw_to_rad(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return (arr * RAD_PER_ANGLE_RAW).astype(np.float32)


def gripper_norm_to_raw(normalized: float, open_raw: int, closed_raw: int) -> int:
    value = float(np.clip(normalized, 0.0, 1.0))
    return int(round(open_raw + value * (closed_raw - open_raw)))


def gripper_raw_to_norm(raw: float | int, open_raw: int, closed_raw: int) -> float:
    denom = float(closed_raw - open_raw)
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip((float(raw) - float(open_raw)) / denom, 0.0, 1.0))


def eef_to_raw(eef: np.ndarray | list[float], open_raw: int, closed_raw: int) -> tuple[list[int], int]:
    arr = np.asarray(eef, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 7:
        raise ValueError(f"EEF action must have 7 values, got {arr.shape}")
    xyz_raw = meters_to_linear_raw(arr[:3])
    rpy_raw = rad_to_angle_raw(arr[3:6])
    grip_raw = gripper_norm_to_raw(float(arr[6]), open_raw=open_raw, closed_raw=closed_raw)
    return xyz_raw + rpy_raw, grip_raw


def raw_to_eef(raw_xyz_rpy: list[float] | tuple[float, ...] | np.ndarray, gripper_raw: float, open_raw: int, closed_raw: int) -> np.ndarray:
    raw = np.asarray(raw_xyz_rpy, dtype=np.float64).reshape(-1)
    if raw.shape[0] < 6:
        padded = np.zeros(6, dtype=np.float64)
        padded[: raw.shape[0]] = raw
        raw = padded
    xyz = linear_raw_to_meters(raw[:3])
    rpy = angle_raw_to_rad(raw[3:6])
    grip = gripper_raw_to_norm(gripper_raw, open_raw=open_raw, closed_raw=closed_raw)
    return np.concatenate([xyz, rpy, np.array([grip], dtype=np.float32)]).astype(np.float32)

