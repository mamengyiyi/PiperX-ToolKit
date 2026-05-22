# PiperX ToolKit

PiperX ToolKit 是一套面向 **双松灵 PiperX 机械臂** 的模仿学习工具包，覆盖：

```text
环境控制 -> 遥操作接口 -> 示教数据采集 -> 数据格式转换 -> 模型部署
```

当前第一版重点支持 **双臂本体示教采集**：将两条 PiperX 本体切到
`teaching / master input` 模式，人手直接拖动两条机械臂完成任务，工具包同步记录双臂状态和三路 RGB 相机图像。后续的主从臂遥操作和 VR 遥操作接口已经预留，但暂未实现。

默认相机为三路 RGB：

- `front`
- `left_wrist`
- `right_wrist`

---

## 1. 系统要求

建议机器人端机器使用：

- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.10 或 3.11，推荐通过 `uv` 安装 Python 3.10
- 已配置 CAN 口，例如 `can0`、`can1`
- 已安装 Piper SDK 所需的系统依赖
- 三路 RGB 相机可被 OpenCV 打开

Ubuntu 24.04 默认系统 Python 通常是 3.12。机器人相关依赖（`piper_sdk`、
`opencv-python`、`lerobot` 等）不一定都在 3.12 上最稳，因此本项目推荐
**不要直接使用系统 Python**，而是使用 `uv` 单独安装 Python 3.10。

建议先安装一些机器人端常用检查工具：

```bash
sudo apt update
sudo apt install -y can-utils v4l-utils
```

其中：

- `can-utils` 用于检查 CAN，例如 `candump can0`
- `v4l-utils` 用于查看相机设备，例如 `v4l2-ctl --list-devices`

本仓库默认用 `uv` 管理 Python 环境。

---

## 2. 安装 uv

机器人端只需要安装一次：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后如果当前终端找不到 `uv`，重新打开一个终端，或执行：

```bash
source ~/.bashrc
```

检查：

```bash
uv --version
```

---

## 3. 创建 Python 环境

进入项目目录：

```bash
cd PiperX-ToolKit
```

创建虚拟环境并安装硬件采集依赖。Ubuntu 24.04 上也建议用这里的 Python 3.10，
不要直接使用系统 Python 3.12：

```bash
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[hardware,data]"
```

如果需要把 Zarr 数据转换成 LeRobot v3，或后续用 LeRobot 相关模型部署，再安装：

```bash
uv pip install -e ".[hardware,data,lerobot]"
```

以后每次重新打开终端，都先进入项目并激活环境：

```bash
cd PiperX-ToolKit
source .venv/bin/activate
```

`requirements-robot.txt` 只是备用依赖清单。推荐始终使用上面的 `uv pip install -e ...`。

---

## 4. 项目结构

```text
piperx_toolkit/
  env/                  # 双臂环境、Piper SDK 适配、单位转换、相机后端
  teleop/               # 遥操作接口；当前实现 teaching_pendant
  collect/              # Zarr 数据采集与 schema
  convert/              # Zarr -> LeRobot v3
  deploy/               # 策略部署主循环
scripts/
  doctor.py             # 环境体检
  inspect_sdk_msgs.py   # 导出 Piper SDK 真实消息结构
  smoke_read.py         # 只读双臂和相机
  collect_teaching.py   # 示教采集主脚本
  convert_to_lerobot_v3.py
  deploy_policy.py
  set_teaching_mode.py
  set_motion_mode.py
configs/
  dual_piperx.yaml      # 默认双臂 + 三相机配置
```

---

## 5. 离线 mock 自检

没有机械臂时也可以先跑 mock 后端，确认代码和 Python 环境基本正常：

```bash
python scripts/test_env.py
```

预期最后看到：

```text
mock env OK
```

再跑只读 mock：

```bash
python scripts/smoke_read.py --backend mock --camera-backend mock --duration 2
```

这会打印 observation 的 key、shape 和 dtype。

---

## 6. 机器人端环境体检

在连接机械臂的机器上，先运行：

```bash
python scripts/doctor.py
```

它会检查：

- Python 版本
- `piper_sdk`
- `python-can`
- `opencv-python`
- `zarr`
- `Pillow`
- `lerobot`
- CAN 网络接口

如果相机接在本机，还可以检查 OpenCV 是否能打开 0、1、2 三个相机：

```bash
python scripts/doctor.py --check-cameras
```

---

## 7. 配置 CAN 和相机

默认配置文件在：

```bash
configs/dual_piperx.yaml
```

默认内容假设：

- 左臂：`can0`
- 右臂：`can1`
- 三路相机：OpenCV device `0 / 1 / 2`

如果你的 CAN 或相机编号不同，直接修改这个文件：

```yaml
arms:
  left_can: can0
  right_can: can1

cameras:
  front:
    device: 0
  left_wrist:
    device: 1
  right_wrist:
    device: 2
```

也可以在命令行临时覆盖 CAN：

```bash
python scripts/smoke_read.py --backend sdk --left-can can0 --right-can can1
```

---

## 8. 导出 Piper SDK 消息结构

第一次上机时，强烈建议先跑：

```bash
python scripts/inspect_sdk_msgs.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1 \
  --out sdk_msgs.json
```

这个脚本不会控制机械臂运动，只会读取并保存 Piper SDK 返回的消息结构，包括：

- `GetArmStatus`
- `GetArmJointMsgs`
- `GetArmGripperMsgs`
- `GetArmEndPoseMsgs`
- `GetArmJointCtrl`
- `GetArmGripperCtrl`

如果后续读数不对，把 `sdk_msgs.json` 发给开发者即可快速适配 SDK 字段。

---

## 9. 只读烟测

确认双臂状态和三路相机都能读到：

```bash
python scripts/smoke_read.py \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --duration 10
```

成功时会打印类似：

```text
Read 300 observations in 10.00s (30.0 Hz)
front_color              shape=(480, 640, 3) dtype=uint8
left_joint_pos           shape=(7,) dtype=float32
right_joint_pos          shape=(7,) dtype=float32
...
```

如果相机打不开，先确认 `configs/dual_piperx.yaml` 里的 `device` 编号。

---

## 10. 切到示教模式

示教采集前，两条 PiperX 本体需要处于 `teaching / master input`，人手可以直接拖动机械臂。

如果你想通过脚本发送示教角色命令：

```bash
python scripts/set_teaching_mode.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1 \
  --configure-gripper-params
```

注意：Piper SDK / 固件可能要求发送主从角色配置后给机械臂重新上电，具体以你的硬件表现为准。

---

## 11. 采集示教数据

确认机械臂可以被手拖动后，运行：

```bash
python scripts/collect_teaching.py \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --dataset datasets/pick_cube.zarr \
  --episodes 5 \
  --hz 30 \
  --task "pick up the cube"
```

交互流程：

```text
Space  开始录制当前 episode
Enter  结束当前 episode
S      保存当前 episode
D      丢弃当前 episode
Ctrl+C 退出采集
```

采集时工具包不会下发动作命令，只会读取：

- 左右臂关节状态
- 左右臂末端位姿
- 关节速度
- 三路 RGB 图像
- 时间戳

### Action 的定义

示教阶段没有真实下发动作，因此数据集中默认写入：

```text
action_left[t]  = left_joint_pos[t + 1]
action_right[t] = right_joint_pos[t + 1]
```

也就是用未来一帧关节位置作为训练目标。默认 `--action-shift-frames 1`，可以按需要调整：

```bash
python scripts/collect_teaching.py ... --action-shift-frames 3
```

---

## 12. Zarr 数据格式

采集后目录类似：

```text
datasets/pick_cube.zarr/
  data/
    rgb_front              (N, 3, H, W) uint8
    rgb_left_wrist         (N, 3, H, W) uint8
    rgb_right_wrist        (N, 3, H, W) uint8
    left_joint_pos         (N, 7) float32
    right_joint_pos        (N, 7) float32
    left_eef_pos           (N, 7) float32
    right_eef_pos          (N, 7) float32
    left_joint_qvel        (N, 7) float32
    right_joint_qvel       (N, 7) float32
    left_joint_effort      (N, 7) float32
    right_joint_effort     (N, 7) float32
    action_left            (N, 7) float32
    action_right           (N, 7) float32
    timestamp              (N,) float64
    episode                (N,) uint32
  meta/
    episode_ends           (M,) uint32
    config                 JSON attrs
```

所有公开单位约定：

- 关节角：rad
- 末端 xyz：m
- 末端 rpy：rad
- gripper：`[0, 1]`，`0` 为全开，`1` 为全闭

---

## 13. 转换为 LeRobot v3

先查看 Zarr 内容：

```bash
python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/pick_cube.zarr \
  --dry-run
```

正式转换：

```bash
python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/pick_cube.zarr \
  --output lerobot_datasets/pick_cube \
  --repo-id mamengyiyi/piperx_pick_cube \
  --fps 30 \
  --task "pick up the cube" \
  --state left_joint_pos,right_joint_pos \
  --action action_left,action_right \
  --cameras front,left_wrist,right_wrist \
  --overwrite
```

默认：

- `observation.state = left_joint_pos + right_joint_pos`
- `action = action_left + action_right`
- `observation.images = front + left_wrist + right_wrist`

---

## 14. 策略部署

部署前，机械臂需要从示教模式切回可控制的 motion/follower 模式。

```bash
python scripts/set_motion_mode.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1 \
  --reset-after-teaching
```

如果固件要求，执行后给机械臂重新上电。

部署脚本：

```bash
python scripts/deploy_policy.py \
  --backend sdk \
  --camera-backend opencv \
  --policy path/to/policy.pt \
  --action-mode absolute_joint \
  --hz 20
```

当前部署 runner 支持：

- `.npy` 动作序列回放
- 可调用的 PyTorch policy
- 带 `predict(images, state)` 方法的 policy

### Action Mode

`DualPiperXEnv.step(action, action_mode=...)` 支持四种模式：

| action_mode | 输入 | 说明 |
|---|---|---|
| `absolute_joint` | `[j0, j1, j2, j3, j4, j5, gripper]` | 关节目标 |
| `absolute_eef` | `[x, y, z, roll, pitch, yaw, gripper]` | 末端目标 |
| `smooth_eef` | `[x, y, z, roll, pitch, yaw, gripper]` | 从当前 EEF 插值到目标 EEF，逐步下发 |
| `delta_eef` | `[dx, dy, dz, droll, dpitch, dyaw, gripper_delta]` | 当前 EEF 加增量后下发 |

部署时默认推荐先用 `absolute_joint`，和示教采集生成的 action 保持一致。

---

## 15. 推荐上机顺序

第一次上机建议严格按这个顺序：

```bash
source .venv/bin/activate

python scripts/doctor.py --check-cameras

python scripts/inspect_sdk_msgs.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1 \
  --out sdk_msgs.json

python scripts/smoke_read.py \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --duration 10

python scripts/collect_teaching.py \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --dataset datasets/test.zarr \
  --episodes 1 \
  --hz 30 \
  --task "test collection"

python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/test.zarr \
  --dry-run
```

如果任何一步报错，优先保存：

- 终端输出
- `sdk_msgs.json`
- `configs/dual_piperx.yaml`

这些信息足够定位大部分 CAN、SDK 字段、相机编号和数据 schema 问题。

---

## 16. 当前已实现与预留

已实现：

- 双 PiperX 环境
- Piper SDK 适配层
- mock 后端
- 三路 RGB 相机
- 示教 / master input 只读采集
- Zarr 数据保存
- Zarr -> LeRobot v3
- 策略部署主循环
- 上机诊断脚本

预留但暂未实现：

- 主从臂遥操作
- VR 遥操作
- 更完整的力矩 / 电流反馈解析
