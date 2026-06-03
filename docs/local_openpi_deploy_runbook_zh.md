# PiperX 本地部署与推理全流程（OpenPI + 实机）

这份文档把你当前这条链路整理成一套可复用的执行脚本，覆盖：

1. 模型/代码下载（从远端同步到本机）
2. 本机 OpenPI 环境搭建
3. 本机 policy server 启动
4. PiperX 硬件联调与相机探测
5. 本地推理（dry-run + execute）

---

## 0. 前置约定

- 本机：`/home/ruihao`
- PiperX 仓库：`/home/ruihao/PiperX-ToolKit`
- OpenPI 本地目录：`/opt/openpi`
- SSH 别名：`new_server`（对应远端训练机）
- 本地 checkpoint 路径：`/home/ruihao/PiperX-ToolKit/train_checkpoint/towel_bimanual_001_29999`

---

## 1. 同步 OpenPI 代码到本机

> 不拷远端 `.venv`，只拷代码。避免二进制依赖不兼容。

```bash
sudo mkdir -p /opt/openpi
sudo chown -R $USER:$USER /opt/openpi

rsync -av --delete \
  --exclude '.venv' \
  --exclude 'wandb_offline' \
  --exclude 'checkpoints' \
  new_server:/root/data/my/piperx/openpi/ /opt/openpi/
```

---

## 2. 同步模型 checkpoint 到本机

如果你本机还没有目标 checkpoint，执行：

```bash
mkdir -p /home/ruihao/PiperX-ToolKit/train_checkpoint

rsync -av \
  new_server:/root/data/my/piperx/openpi/checkpoints/pi05_piperx_bimanual_fold_towel_20260527/towel_bimanual_001_29999/ \
  /home/ruihao/PiperX-ToolKit/train_checkpoint/towel_bimanual_001_29999/
```

如果本机已经有该目录，可跳过本节。

---

## 3. 本机 OpenPI 环境搭建

```bash
cd /opt/openpi

# 推荐使用 uv
uv sync

# 验证 JAX/GPU
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /opt/openpi/.venv/bin/python - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY
```

---

## 4. 处理 norm_stats 路径对齐（本模型必需）

当前配置 `pi05_piperx_bimanual_fold_towel_20260527` 期望：

`assets/ruio248/fold_towel_20260527_merged_v21/norm_stats.json`

而 checkpoint 内现有路径通常是：

`assets/mamengyiyi/piperx_towel_bimanual/norm_stats.json`

用软链接对齐：

```bash
mkdir -p /home/ruihao/PiperX-ToolKit/train_checkpoint/towel_bimanual_001_29999/assets/ruio248/fold_towel_20260527_merged_v21

ln -sfn \
  /home/ruihao/PiperX-ToolKit/train_checkpoint/towel_bimanual_001_29999/assets/mamengyiyi/piperx_towel_bimanual/norm_stats.json \
  /home/ruihao/PiperX-ToolKit/train_checkpoint/towel_bimanual_001_29999/assets/ruio248/swing_fold_towel_20260527_merged_v21/norm_stats.json
```

---

## 5. 启动本地 Policy Server

开一个新终端 A：

```bash
cd /opt/openpi

env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /opt/openpi/.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "fold the towel" \
  policy:checkpoint \
  --policy.config=pi05_piperx_bimanual_fold_towel_20260527 \
  --policy.dir=/home/ruihao/PiperX-ToolKit/train_checkpoint/fold_towel_parts123_horizon100_8gpu_bs64_150k_001
```

看到以下日志说明服务成功：

```text
server listening on 0.0.0.0:8000
```

---

## 6. PiperX 硬件体检与相机探测

开新终端 B：

```bash
cd /home/ruihao/PiperX-ToolKit
source .venv/bin/activate

python scripts/doctor.py --check-cameras
python scripts/probe_cameras.py --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
```

当前实测稳定可用建议：

- `front: 4`
- `left_wrist: 10`
- `right_wrist: 16`

更新配置文件 `configs/dual_piperx.yaml`：

```yaml
cameras:
  front:
    backend: opencv
    device: 4
  left_wrist:
    backend: opencv
    device: 10
  right_wrist:
    backend: opencv
    device: 16
```

再做烟测：

```bash
python scripts/smoke_read.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend opencv \
  --left-can can0 \
  --right-can can1 \
  --duration 10
```

---

## 7. 本地推理（dry-run）

> 不加 `--execute` 不会下发动作。

```bash
cd /home/ruihao/PiperX-ToolKit
source .venv/bin/activate

python scripts/deploy_openpi_bimanual.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend opencv \
  --host 127.0.0.1 \
  --port 8000 \
  --observation-format piperx \
  --action-mode absolute_joint \
  --hz 20 \
  --duration 20 \
  --print-every 10
```

预期看到持续输出：

```text
step=000010 left=[...] right=[...]
```

---

## 8. 本地推理（实机执行）

确认 dry-run 正常后再执行：

```bash
cd /home/ruihao/PiperX-ToolKit
source .venv/bin/activate

python scripts/deploy_openpi_bimanual.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend realsense \
  --host 127.0.0.1 \
  --port 8000 \
  --observation-format piperx \
  --action-mode absolute_joint \
  --hz 30 \
  --duration 600 \
  --print-every 10 \
  --max-joint-delta-rad 0.2 \
  --lowpass-alpha 0.8 \
  --execute
```

首次建议：

- 把 `--duration` 控制在 10~20s
- 清空工作空间
- 人在急停旁

---

## 9. 常见报错与处理

### 9.1 `Could not open camera front: 0`

相机 index 不对，重新 `probe_cameras.py`，并更新 `dual_piperx.yaml`。

### 9.2 `KeyError: 'observation/image'`

服务端和客户端观测 key 不一致。当前仓库已做兼容发送，部署时加：

```bash
--observation-format piperx
```

### 9.3 `Norm stats file not found`

执行本文件第 4 节软链接步骤，对齐 `norm_stats.json` 路径。

### 9.4 `Failed to connect to 127.0.0.1:7897`（安装依赖时）

是代理变量影响下载，临时去掉代理后再执行：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy uv sync
```

---

## 10. 两终端最小化运行清单

终端 A（server）：

```bash
cd /opt/openpi
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /opt/openpi/.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "fold the towel" \
  policy:checkpoint \
  --policy.config=pi05_piperx_bimanual_fold_towel_20260527 \
  --policy.dir=/home/ruihao/PiperX-ToolKit/train_checkpoint/fold_towel_parts123_horizon100_8gpu_bs64_150k_001/15000
```

终端 B（deploy）：

```bash
cd /home/ruihao/PiperX-ToolKit
source .venv/bin/activate
python scripts/deploy_openpi_bimanual.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend opencv \
  --host 127.0.0.1 \
  --port 8000 \
  --observation-format piperx \
  --action-mode absolute_joint \
  --hz 30 \
  --duration 600 \
  --print-every 10
```

