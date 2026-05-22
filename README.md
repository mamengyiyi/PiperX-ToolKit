# PiperX ToolKit

PiperX ToolKit 是一套面向 **双松灵 PiperX 机械臂** 的模仿学习工具包，覆盖：

```text
环境控制 -> 遥操作接口 -> 示教数据采集 -> 数据格式转换 -> 模型部署
```

当前第一版重点支持 **本体拖动示教采集**：将 PiperX 本体切到
`motion / slave output` 角色，工具包只读取反馈，不下发运动指令；人手直接拖动机械臂完成任务，
同步记录机械臂状态和 RGB 相机图像。`teaching / master input`、主从臂遥操作和 VR 遥操作接口已经预留，
但当前推荐的实机采集路径是 `motion / slave output + 手拖本体`。

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
  probe_cameras.py      # 枚举并测试 OpenCV 相机设备
  inspect_sdk_msgs.py   # 导出 Piper SDK 真实消息结构
  smoke_read.py         # 只读双臂和相机
  smoke_left_arm.py     # 单臂 + 单相机烟测
  collect_single_arm.py # 单臂 + 单相机采集
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

RealSense 一类相机通常会暴露多个 `/dev/video*` 节点，其中只有部分节点能被
OpenCV 直接作为 RGB 图像读取。可以把 0 到 5 都探测一遍：

```bash
python scripts/probe_cameras.py --indices 0,1,2,3,4,5
```

如果你已经知道稳定设备路径，也可以直接探测 `/dev/v4l/by-id/...`：

```bash
python scripts/probe_cameras.py \
  --indices "" \
  --devices /dev/v4l/by-id/xxx-video-index2
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

如果上机调试时暂时只有一个 CAN 口或只接了一条机械臂，可以使用混合后端：

```bash
python scripts/smoke_read.py \
  --backend sdk \
  --left-can can0 \
  --left-backend sdk \
  --right-backend mock \
  --camera-backend mock \
  --duration 10
```

仓库也提供了一个单臂调试配置：

```bash
python scripts/smoke_read.py --config configs/debug_single_arm.yaml --duration 10
```

`configs/debug_single_arm.yaml` 默认假设：

- 左臂：真实 `sdk / can0`
- 右臂：`mock`
- `front` 相机：OpenCV device `4`，请按 `probe_cameras.py` 的结果改成你机器上可读的 RGB 节点
- 两个腕部相机：`mock`

这只是为了先把一条真实机械臂、Piper SDK 字段和相机读取链路测通。正式双臂采集前，
仍然建议使用真实的 `can0 + can1` 和三路真实 RGB 相机。

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

## 10. 采集前机械臂模式

实机测试发现：PiperX 在 `motion / slave output` 角色下可以被人手拖动，同时 Piper SDK
仍能稳定读取关节和末端反馈；而切到 `teaching / master input` 后，普通反馈读取可能变成
全 0。这个现象也和 Piper SDK 官方说明一致：读取关节反馈需要机械臂在 slave 模式。

所以当前推荐的示教采集方式是：

```text
motion/slave output 角色 + 工具包只读状态 + 人手拖动机械臂
```

采集前可以发送 motion/slave output 角色命令：

```bash
python scripts/set_motion_mode.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1
```

如果你在调试 master input 行为，也可以发送 teaching/master input 角色命令，但这个模式不作为
当前默认采集路径：

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

### 11.1 单臂 + 单相机采集

如果当前只接了一条 PiperX 和一路 `front` RGB 相机，先用这个脚本采数据：

```bash
python scripts/collect_single_arm.py \
  --can can0 \
  --backend sdk \
  --set-motion-output-role \
  --camera-backend opencv \
  --camera-device 4 \
  --camera-fail-soft \
  --camera-read-retries 10 \
  --camera-warmup-s 2 \
  --dataset datasets/single_arm_test.zarr \
  --episodes 1 \
  --duration 15 \
  --hz 30 \
  --task "single arm test"
```

这条命令会自动录制 1 个 15 秒 episode 并保存。你的 RealSense D405 当前实测可读 RGB
节点是 OpenCV device `4`；如果换相机或重新插拔后编号变化，先重新运行：

```bash
python scripts/probe_cameras.py --indices 0,1,2,3,4,5
```

也可以用交互模式采多个 episode。去掉 `--duration` 后，脚本会等待键盘控制：

```bash
python scripts/collect_single_arm.py \
  --can can0 \
  --backend sdk \
  --set-motion-output-role \
  --camera-backend opencv \
  --camera-device 4 \
  --camera-fail-soft \
  --camera-read-retries 10 \
  --camera-warmup-s 2 \
  --dataset datasets/single_arm_demo.zarr \
  --episodes 3 \
  --hz 30 \
  --task "single arm demo"
```

交互流程：

```text
Space  开始录制当前 episode
Enter  结束当前 episode
S      保存当前 episode
D      丢弃当前 episode
Ctrl+C 退出采集
```

`--camera-fail-soft` 会在相机偶发读帧失败时使用上一帧补齐；如果相机完全打不开，会写入黑图并打印警告。
正式采集前建议先用 `smoke_left_arm.py` 确认机械臂和相机都能稳定读取：

```bash
python scripts/smoke_left_arm.py \
  --can can0 \
  --backend sdk \
  --set-motion-output-role \
  --camera-backend opencv \
  --camera-device 4 \
  --camera-fail-soft \
  --camera-read-retries 10 \
  --camera-warmup-s 2 \
  --duration 10 \
  --hz 30
```

### 11.2 双臂 + 三相机采集

确认两条机械臂都可以被手拖动、三路相机都能读到后，运行：

```bash
python scripts/collect_teaching.py \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --set-motion-output-role \
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
单臂数据：action[t]       = joint_pos[t + 1]
双臂数据：action_left[t]  = left_joint_pos[t + 1]
双臂数据：action_right[t] = right_joint_pos[t + 1]
```

也就是用未来一帧关节位置作为训练目标。默认 `--action-shift-frames 1`，可以按需要调整：

```bash
python scripts/collect_teaching.py ... --action-shift-frames 3
```

---

## 12. Zarr 数据格式

单臂单相机采集后目录类似：

```text
datasets/single_arm_test.zarr/
  data/
    rgb_front              (N, 3, H, W) uint8
    joint_pos              (N, 7) float32
    eef_pos                (N, 7) float32
    joint_qvel             (N, 7) float32
    joint_effort           (N, 7) float32
    action                 (N, 7) float32
    timestamp              (N,) float64
    episode                (N,) uint32
  meta/
    episode_ends           (M,) uint32
    config                 JSON attrs
```

双臂三相机采集后目录类似：

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
  --zarr datasets/single_arm_test.zarr \
  --dry-run
```

单臂单相机数据正式转换：

```bash
mkdir -p lerobot_datasets/mamengyiyi

python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/single_arm_test.zarr \
  --output lerobot_datasets/mamengyiyi/piperx_single_arm_test \
  --repo-id mamengyiyi/piperx_single_arm_test \
  --robot-type piperx_single_arm \
  --fps 30 \
  --task "single arm test" \
  --state joint_pos \
  --action action \
  --cameras front \
  --overwrite
```

转换后可以用 LeRobot 自带的可视化工具打开本地 episode：

```bash
lerobot-dataset-viz \
  --repo-id mamengyiyi/piperx_single_arm_test \
  --root lerobot_datasets/mamengyiyi/piperx_single_arm_test \
  --mode local \
  --episode-index 0
```

这里的 `--root` 必须指向包含 `meta/info.json` 的数据集根目录。如果 `--root` 指到
`lerobot_datasets` 这一层，LeRobot 找不到本地 metadata 时会尝试去 Hugging Face Hub 下载同名数据集，
没有登录或远端数据集不存在时就会报 401。执行后会打开 Rerun 窗口，显示相机画面、状态和动作。
如果你的 `lerobot` 版本还没有
`lerobot-dataset-viz` 命令，先确认安装了完整依赖：

```bash
uv pip install -e ".[hardware,data,lerobot]"
lerobot-dataset-viz --help
```

双臂三相机数据正式转换：

```bash
mkdir -p lerobot_datasets/mamengyiyi

python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/pick_cube.zarr \
  --output lerobot_datasets/mamengyiyi/piperx_pick_cube \
  --repo-id mamengyiyi/piperx_pick_cube \
  --robot-type piperx_bimanual \
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

## 14. 单臂数据回放

如果想把采到的单臂轨迹回放到 PiperX 上，先做 dry-run 看 episode 范围和动作幅度：

```bash
python scripts/replay_single_arm.py \
  --zarr datasets/single_arm_test.zarr \
  --episode 0 \
  --key action \
  --can can0
```

确认第一帧、末帧、最大关节步长都合理后，再真实执行。建议第一次把空间清空、手放在急停附近，
并让机械臂尽量回到采集开始时附近的位置：

```bash
python scripts/replay_single_arm.py \
  --zarr datasets/single_arm_test.zarr \
  --episode 0 \
  --key action \
  --can can0 \
  --set-motion-output-role \
  --enable \
  --approach-start \
  --speed-ratio 30 \
  --max-joint-delta-rad 0.08 \
  --execute
```

说明：

- `--key action` 会回放采集时生成的下一帧关节目标；也可以改成 `--key joint_pos` 回放原始关节序列。
- 不加 `--execute` 时脚本只打印统计，不会给机械臂发控制命令。
- `--approach-start` 会低速插值到 episode 第一帧，避免第一条控制命令跨度太大。
- 如果 `EnablePiper()` 超时，先检查机械臂供电、急停、CAN 状态以及是否需要重新上电。

---

## 15. 策略部署

部署前，机械臂需要处于可控制的 motion/slave output 模式。

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

## 16. 推荐上机顺序

第一次上机建议严格按这个顺序：

```bash
source .venv/bin/activate

python scripts/doctor.py --check-cameras

python scripts/probe_cameras.py --indices 0,1,2,3,4,5

python scripts/inspect_sdk_msgs.py \
  --backend sdk \
  --left-can can0 \
  --right-can can1 \
  --out sdk_msgs.json

python scripts/smoke_left_arm.py \
  --can can0 \
  --backend sdk \
  --set-motion-output-role \
  --camera-backend opencv \
  --camera-device 4 \
  --camera-fail-soft \
  --camera-read-retries 10 \
  --camera-warmup-s 2 \
  --duration 10 \
  --hz 30

python scripts/collect_single_arm.py \
  --can can0 \
  --backend sdk \
  --set-motion-output-role \
  --camera-backend opencv \
  --camera-device 4 \
  --camera-fail-soft \
  --camera-read-retries 10 \
  --camera-warmup-s 2 \
  --dataset datasets/single_arm_test.zarr \
  --episodes 1 \
  --duration 15 \
  --hz 30 \
  --task "single arm test"

python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/single_arm_test.zarr \
  --dry-run

mkdir -p lerobot_datasets/mamengyiyi

python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/single_arm_test.zarr \
  --output lerobot_datasets/mamengyiyi/piperx_single_arm_test \
  --repo-id mamengyiyi/piperx_single_arm_test \
  --robot-type piperx_single_arm \
  --fps 30 \
  --task "single arm test" \
  --state joint_pos \
  --action action \
  --cameras front \
  --overwrite

lerobot-dataset-viz \
  --repo-id mamengyiyi/piperx_single_arm_test \
  --root lerobot_datasets/mamengyiyi/piperx_single_arm_test \
  --mode local \
  --episode-index 0
```

双臂和三路相机都接好后，再跑完整链路：

```bash
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
  --set-motion-output-role \
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

## 17. 当前已实现与预留

已实现：

- 双 PiperX 环境
- Piper SDK 适配层
- mock 后端
- 三路 RGB 相机
- motion/slave output 本体拖动只读采集
- 单臂单相机采集脚本
- 单臂 Zarr 轨迹回放脚本
- Zarr 数据保存
- Zarr -> LeRobot v3
- 策略部署主循环
- 上机诊断脚本

预留但暂未实现：

- 主从臂遥操作
- VR 遥操作
- 更完整的力矩 / 电流反馈解析
