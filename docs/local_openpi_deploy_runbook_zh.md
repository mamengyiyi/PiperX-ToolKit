# PiperX 本地部署与推理全流程（OpenPI + 实机）

这份文档把 OpenPI + PiperX 实机部署整理成可复用流程，覆盖：

1. 模型/代码下载（从远端同步到本机）
2. 本机 OpenPI 环境搭建
3. norm_stats 路径对齐
4. 本机 policy server 启动
5. PiperX 硬件联调与相机探测
6. 本地推理（dry-run + execute + timing）

---

## 0. 前置约定

先在本机 shell 里设置这些变量（后续章节都会引用）：

```bash
# 路径
export PIPERX_ROOT="${PIPERX_ROOT:-$HOME/PiperX-ToolKit}"
export OPENPI_ROOT="${OPENPI_ROOT:-/opt/openpi}"

# 远端同步（rsync / SSH 别名，按你的环境改）
export REMOTE_HOST="${REMOTE_HOST:-new_server}"
export REMOTE_OPENPI="${REMOTE_OPENPI:-/root/data/my/piperx/openpi}"

# OpenPI policy 配置名（serve_policy.py 里的 --policy.config）
export POLICY_CONFIG="${POLICY_CONFIG:-pi05_piperx_bimanual_fold_towel_20260527}"

# checkpoint：训练 run 名 + step
export CKPT_RUN_NAME="${CKPT_RUN_NAME:-towel_bimanual_001_29999}"
export CKPT_STEP="${CKPT_STEP:-29999}"
export LOCAL_CKPT="${PIPERX_ROOT}/train_checkpoint/${CKPT_RUN_NAME}/${CKPT_STEP}"

# norm_stats：OpenPI config 期望的路径 vs checkpoint 内实际路径
export NORM_STATS_EXPECTED="${NORM_STATS_EXPECTED:-assets/ruio248/fold_towel_20260527_merged_v21}"
export NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-assets/mamengyiyi/piperx_towel_bimanual}"

# 语言 prompt（与训练一致）
export POLICY_PROMPT="${POLICY_PROMPT:-fold the towel}"
```

说明：

- `train_checkpoint/` 已在 `.gitignore` 中，不会进 Git；每人本机路径可不同。
- 若 checkpoint 目录结构是 `train_checkpoint/<run_name>/`（无 step 子目录），可设 `LOCAL_CKPT="${PIPERX_ROOT}/train_checkpoint/${CKPT_RUN_NAME}"` 并忽略 `CKPT_STEP`。

---

## 1. 同步 OpenPI 代码到本机

> 不拷远端 `.venv`，只拷代码，避免二进制依赖不兼容。

```bash
sudo mkdir -p "${OPENPI_ROOT}"
sudo chown -R "$USER:$USER" "${OPENPI_ROOT}"

rsync -av --delete \
  --exclude '.venv' \
  --exclude 'wandb_offline' \
  --exclude 'checkpoints' \
  "${REMOTE_HOST}:${REMOTE_OPENPI}/" "${OPENPI_ROOT}/"
```

---

## 2. 同步模型 checkpoint 到本机

若本机还没有目标 checkpoint：

```bash
mkdir -p "${PIPERX_ROOT}/train_checkpoint"

# 远端路径按你的训练输出目录修改
rsync -av \
  "${REMOTE_HOST}:${REMOTE_OPENPI}/checkpoints/${POLICY_CONFIG}/${CKPT_RUN_NAME}/${CKPT_STEP}/" \
  "${LOCAL_CKPT}/"
```

若 checkpoint 没有 step 子目录，去掉 rsync 源/目标里的 `${CKPT_STEP}/` 即可。

同步完成后确认目录存在：

```bash
ls "${LOCAL_CKPT}"
```

---

## 3. 本机 OpenPI 环境搭建

```bash
cd "${OPENPI_ROOT}"

# 推荐使用 uv
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy uv sync

# 验证 JAX/GPU
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  "${OPENPI_ROOT}/.venv/bin/python" - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY
```

---

## 4. 处理 norm_stats 路径对齐

OpenPI config（`${POLICY_CONFIG}`）会按固定 asset 路径加载 norm stats，例如：

```text
${LOCAL_CKPT}/${NORM_STATS_EXPECTED}/norm_stats.json
```

而 checkpoint 内实际文件可能在 `${NORM_STATS_SOURCE}/norm_stats.json`。  
用软链接把「期望路径」指到「实际路径」。

### 方式 A：只链 norm_stats.json（最常见）

```bash
mkdir -p "${LOCAL_CKPT}/${NORM_STATS_EXPECTED}"

ln -sfn \
  "${LOCAL_CKPT}/${NORM_STATS_SOURCE}/norm_stats.json" \
  "${LOCAL_CKPT}/${NORM_STATS_EXPECTED}/norm_stats.json"
```

### 方式 B：整个 assets 子目录已是正确 stats，链整个目录

```bash
mkdir -p "$(dirname "${LOCAL_CKPT}/${NORM_STATS_EXPECTED}")"

ln -sfn \
  "${LOCAL_CKPT}/${NORM_STATS_SOURCE}" \
  "${LOCAL_CKPT}/${NORM_STATS_EXPECTED}"
```

验证：

```bash
ls -l "${LOCAL_CKPT}/${NORM_STATS_EXPECTED}/norm_stats.json"
```

---

## 5. 启动本地 Policy Server

开终端 A：

```bash
cd "${OPENPI_ROOT}"

env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  "${OPENPI_ROOT}/.venv/bin/python" scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "${POLICY_PROMPT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${LOCAL_CKPT}"
```

看到以下日志说明服务成功：

```text
server listening on 0.0.0.0:8000
```

---

## 6. PiperX 硬件体检与相机探测

开终端 B：

```bash
cd "${PIPERX_ROOT}"
source .venv/bin/activate

python scripts/doctor.py --check-cameras
python scripts/probe_cameras.py --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
```

### 相机配置建议

| 阶段 | 建议 backend | 说明 |
|------|--------------|------|
| 探测 / 烟测 | `opencv` | 用 `probe_cameras.py` 找到可用 index，写入 `configs/dual_piperx.yaml` |
| 实机部署 | `realsense` | 使用 RealSense 序列号（`device:` 字段），稳定性更好 |

`configs/dual_piperx.yaml` 示例（OpenCV 探测阶段）：

```yaml
cameras:
  front:
    backend: opencv
    device: 4        # 按 probe 结果修改
  left_wrist:
    backend: opencv
    device: 10
  right_wrist:
    backend: opencv
    device: 16
```

CAN 接口按本机接线修改（常见为 `can0` / `can1`）：

```yaml
arms:
  left_can: can0
  right_can: can1
```

烟测：

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
cd "${PIPERX_ROOT}"
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

### 7.1 可选：timing 诊断

排查 infer 延迟、相机帧率、chunk 边界跳变时：

```bash
python scripts/deploy_openpi_bimanual.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend realsense \
  --host 127.0.0.1 \
  --port 8000 \
  --observation-format piperx \
  --action-mode absolute_joint \
  --hz 30 \
  --chunk-size 15 \
  --duration 30 \
  --print-every 5 \
  --profile-timing
```

输出含 `infer_ms`、`camera_fps`、`raw_jump`（相邻推理 action 跳变）等字段。

逐步确认 chunk（人工看相机帧后再执行）：

```bash
python scripts/deploy_openpi_bimanual_step.py \
  --config configs/dual_piperx.yaml \
  --backend sdk \
  --camera-backend realsense \
  --host 127.0.0.1 \
  --port 8000 \
  --observation-format piperx \
  --action-mode absolute_joint \
  --chunk-size 10 \
  --execute-steps 5
```

---

## 8. 本地推理（实机执行）

确认 dry-run 正常后再执行：

```bash
cd "${PIPERX_ROOT}"
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
  --chunk-size 30 \
  --duration 600 \
  --print-every 10 \
  --execute
```

脚本默认已开启 action smoothing（`--lowpass-alpha 0.8`，`--max-joint-delta-rad 0.08`）。  
若需更保守或关闭平滑，可显式覆盖：

```bash
  --lowpass-alpha 0.6 \
  --max-joint-delta-rad 0.04 \
  # 或完全关闭：--no-smooth
```

首次实机建议：

- 把 `--duration` 控制在 10~20s
- 清空工作空间
- 人在急停旁

---

## 9. 常见报错与处理

### 9.1 `Could not open camera front: 0`

相机 index / 序列号不对。重新运行 `probe_cameras.py` 或检查 RealSense 序列号，更新 `configs/dual_piperx.yaml`。

### 9.2 `KeyError: 'observation/image'`

服务端和客户端观测 key 不一致。部署时加：

```bash
--observation-format piperx
```

### 9.3 `Norm stats file not found`

执行第 4 节软链接，并确认 `${LOCAL_CKPT}`、`${NORM_STATS_EXPECTED}` 变量正确。

### 9.4 `Failed to connect to 127.0.0.1:7897`（安装依赖时）

代理变量影响下载，临时去掉代理：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy uv sync
```

---

## 10. 两终端最小化运行清单

终端 A 启动前，确保第 0 节变量已 export（至少 `LOCAL_CKPT`、`POLICY_CONFIG`、`POLICY_PROMPT`）。

终端 A（server）：

```bash
cd "${OPENPI_ROOT}"
env -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  "${OPENPI_ROOT}/.venv/bin/python" scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "${POLICY_PROMPT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${LOCAL_CKPT}"
```

终端 B（deploy）：

```bash
cd "${PIPERX_ROOT}"
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
  --chunk-size 30 \
  --duration 600 \
  --print-every 10 \
  --execute
```
