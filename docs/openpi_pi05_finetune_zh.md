# 使用自采 LeRobot 数据微调 OpenPI pi0.5

本文档说明如何把 PiperX ToolKit 采集的数据转换为 LeRobot 格式，并在训练服务器上用
[OpenPI](https://github.com/Physical-Intelligence/openpi) 微调已经下载好的官方 `pi05_base`
模型。

当前训练服务器已准备好的关键路径：

```bash
OPENPI_DIR=/root/data/my/piperx/openpi
OPENPI_DATA_HOME=/root/data/my/piperx/openpi_cache
PI05_BASE=/root/data/my/piperx/openpi_cache/openpi-assets/checkpoints/pi05_base
HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets
```

官方 `pi05_base` checkpoint 已经从下面这个官方 GS 地址下载：

```bash
gs://openpi-assets/checkpoints/pi05_base
```

本流程只使用这个官方模型作为初始化权重。

---

## 1. 总体流程

```text
机器人端采集 Zarr
  -> 转成 LeRobot v3 数据集
  -> 拷贝到训练服务器
  -> 在 OpenPI 中注册 PiperX 数据配置
  -> 计算 normalization statistics
  -> 从 pi05_base 微调
  -> 启动 OpenPI policy server
  -> 机器人端连接 policy server 部署
```

OpenPI 不能只靠一个 LeRobot 数据目录直接训练。它需要一个训练配置，明确告诉模型：

- LeRobot 数据集在哪里
- 哪些 key 是图像、状态和动作
- 图像如何映射到 OpenPI 的三路视觉输入
- 状态和动作维度是多少
- 是否使用数据集里的 `task` 作为语言 prompt
- 使用哪个 checkpoint 初始化

所以除了准备 LeRobot 数据，还需要在 OpenPI 里加一个 PiperX 专用配置。

---

## 2. 机器人端：把 Zarr 转成 LeRobot

### 2.1 单臂主从采集数据

如果数据来自 `scripts/collect_leader_follower_single_arm.py`，推荐用下面方式转换。
这里假设右臂是主臂，左臂是从臂，因此数据里存的是：

- `front`
- `left_wrist`
- `joint_pos`
- `action`

其中 `joint_pos` 是从臂当前状态，`action` 是实际发送给从臂的目标关节动作。

```bash
python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/leader_follower_single_arm_front_leftwrist.zarr \
  --output lerobot_datasets/mamengyiyi/piperx_lf_single \
  --repo-id mamengyiyi/piperx_lf_single \
  --robot-type piperx_single_arm \
  --fps 30 \
  --state joint_pos \
  --action action \
  --cameras front,left_wrist \
  --overwrite
```

### 2.2 双臂主从采集数据

如果数据来自 `scripts/collect_leader_follower_bimanual.py`，推荐用：

```bash
python scripts/convert_to_lerobot_v3.py \
  --zarr datasets/leader_follower_bimanual.zarr \
  --output lerobot_datasets/mamengyiyi/piperx_lf_bimanual \
  --repo-id mamengyiyi/piperx_lf_bimanual \
  --robot-type piperx_bimanual \
  --fps 30 \
  --state left_joint_pos,right_joint_pos \
  --action action_left,action_right \
  --cameras front,left_wrist,right_wrist \
  --overwrite
```

### 2.3 转换后先本地检查

注意 `--root` 必须指向真正包含 `meta/info.json` 的目录：

```bash
lerobot-dataset-viz \
  --repo-id mamengyiyi/piperx_lf_single \
  --root lerobot_datasets/mamengyiyi/piperx_lf_single \
  --mode local \
  --episode-index 0
```

如果是双臂数据，把 `repo-id` 和 `root` 换成：

```bash
--repo-id mamengyiyi/piperx_lf_bimanual
--root lerobot_datasets/mamengyiyi/piperx_lf_bimanual
```

---

## 3. 拷贝 LeRobot 数据到训练服务器

OpenPI 默认会从 `HF_LEROBOT_HOME / repo_id` 读取本地 LeRobot 数据。为了让
`repo_id=mamengyiyi/piperx_lf_single` 能被直接读到，服务器目录必须长这样：

```text
/root/data/my/piperx/lerobot_datasets/
  mamengyiyi/
    piperx_lf_single/
      meta/
      data/
      videos/   # 如果转换时使用了视频
```

从机器人端拷贝单臂数据：

```bash
rsync -av --progress -e "ssh -p 38044" \
  lerobot_datasets/mamengyiyi/piperx_lf_single/ \
  my@ssh-cn-huabei1.ebcloud.com:/root/data/my/piperx/lerobot_datasets/mamengyiyi/piperx_lf_single/
```

拷贝双臂数据：

```bash
rsync -av --progress -e "ssh -p 38044" \
  lerobot_datasets/mamengyiyi/piperx_lf_bimanual/ \
  my@ssh-cn-huabei1.ebcloud.com:/root/data/my/piperx/lerobot_datasets/mamengyiyi/piperx_lf_bimanual/
```

---

## 4. 训练服务器环境变量

登录训练服务器：

```bash
ssh -p 38044 my@ssh-cn-huabei1.ebcloud.com
```

进入 OpenPI：

```bash
cd /root/data/my/piperx/openpi
```

每次训练前建议先执行：

```bash
export PATH="/root/data/my/piperx/openpi/.venv/bin:$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/data/my/piperx/openpi_cache
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets
export HF_HOME=/root/data/my/piperx/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/data/my/piperx/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

其中：

- `OPENPI_DATA_HOME` 用于 OpenPI checkpoint cache
- `HF_LEROBOT_HOME` 用于本地 LeRobot 数据
- `HF_ENDPOINT` 只影响 HuggingFace 数据或依赖下载，不影响官方 GS checkpoint
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 允许 JAX 使用更多 GPU 显存

确认 OpenPI 和 GPU：

```bash
.venv/bin/python -c "import jax, torch; print(jax.devices()); print(torch.cuda.is_available())"
```

确认官方 `pi05_base` 在本机：

```bash
du -sh /root/data/my/piperx/openpi_cache/openpi-assets/checkpoints/pi05_base
```

---

## 5. 在 OpenPI 中新增 PiperX 数据适配

OpenPI 的训练代码需要一个 policy transform，把 LeRobot 中的字段映射成模型输入。

新增文件：

```text
/root/data/my/piperx/openpi/src/openpi/policies/piperx_policy.py
```

内容如下：

```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    has_left_wrist: bool = True
    has_right_wrist: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        left_wrist = (
            _parse_image(data["observation/left_wrist_image"])
            if self.has_left_wrist and "observation/left_wrist_image" in data
            else np.zeros_like(base_image)
        )
        right_wrist = (
            _parse_image(data["observation/right_wrist_image"])
            if self.has_right_wrist and "observation/right_wrist_image" in data
            else np.zeros_like(base_image)
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.has_left_wrist else np.False_,
                "right_wrist_0_rgb": np.True_ if self.has_right_wrist else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
```

### 5.1 修改 OpenPI 训练配置

打开：

```text
/root/data/my/piperx/openpi/src/openpi/training/config.py
```

在已有 policy imports 附近加入：

```python
from openpi.policies import piperx_policy
```

在 `LeRobotAlohaDataConfig` 或 `LeRobotLiberoDataConfig` 后面新增：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotPiperXDataConfig(DataConfigFactory):
    action_dim: int = 14
    has_left_wrist: bool = True
    has_right_wrist: bool = True
    default_prompt: str | None = None
    use_delta_joint_actions: bool = False

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_structure = {
            "observation/image": "observation.images.front",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.has_left_wrist:
            repack_structure["observation/left_wrist_image"] = "observation.images.left_wrist"
        if self.has_right_wrist:
            repack_structure["observation/right_wrist_image"] = "observation.images.right_wrist"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                piperx_policy.PiperXInputs(
                    model_type=model_config.model_type,
                    has_left_wrist=self.has_left_wrist,
                    has_right_wrist=self.has_right_wrist,
                )
            ],
            outputs=[piperx_policy.PiperXOutputs(action_dim=self.action_dim)],
        )

        if self.use_delta_joint_actions:
            if self.action_dim == 7:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
            elif self.action_dim == 14:
                delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            else:
                raise ValueError(f"Unsupported PiperX action_dim for delta actions: {self.action_dim}")
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```

说明：

- PiperX ToolKit 当前主从采集的 `action` 是 `absolute_joint`
- 所以默认 `use_delta_joint_actions=False`
- 如果以后采集的是 delta action，再改为 `True`
- 单臂 `action_dim=7`
- 双臂 `action_dim=14`
- `PadStatesAndActions` 会把状态和动作 pad 到 pi0.5 的内部 `action_dim=32`

---

## 6. 添加训练配置

继续在 `src/openpi/training/config.py` 的 `_CONFIGS` 列表里增加配置。

### 6.1 单臂配置

适用于：

- `repo_id=mamengyiyi/piperx_lf_single`
- 图像：`front + left_wrist`
- 状态：7 维
- 动作：7 维

```python
TrainConfig(
    name="pi05_piperx_single_lf",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotPiperXDataConfig(
        repo_id="mamengyiyi/piperx_lf_single",
        action_dim=7,
        has_left_wrist=True,
        has_right_wrist=False,
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/root/data/my/piperx/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    save_interval=1000,
    keep_period=5000,
    wandb_enabled=False,
)
```

### 6.2 双臂配置

适用于：

- `repo_id=mamengyiyi/piperx_lf_bimanual`
- 图像：`front + left_wrist + right_wrist`
- 状态：14 维
- 动作：14 维

```python
TrainConfig(
    name="pi05_piperx_bimanual_lf",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotPiperXDataConfig(
        repo_id="mamengyiyi/piperx_lf_bimanual",
        action_dim=14,
        has_left_wrist=True,
        has_right_wrist=True,
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/root/data/my/piperx/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    save_interval=1000,
    keep_period=5000,
    wandb_enabled=False,
)
```

`batch_size` 必须能被 GPU 数整除。当前训练服务器有 8 张 GPU，所以 64、128、256 都可以。
第一次建议从 64 开始，确认显存和 loss 都正常后再增大。

---

## 7. 检查 LeRobot 数据能被 OpenPI 读到

在训练服务器执行：

```bash
cd /root/data/my/piperx/openpi

export PATH="/root/data/my/piperx/openpi/.venv/bin:$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets

.venv/bin/python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_id = "mamengyiyi/piperx_lf_single"
ds = LeRobotDataset(repo_id)
print("frames:", len(ds))
print("fps:", ds.meta.fps)
print("features:", ds.features)
print("sample keys:", ds[0].keys())
PY
```

双臂数据把 `repo_id` 改成：

```python
repo_id = "mamengyiyi/piperx_lf_bimanual"
```

如果这里报 `meta/info.json` 不存在，通常是数据目录没有放到：

```text
$HF_LEROBOT_HOME/<repo_id>
```

例如单臂必须是：

```text
/root/data/my/piperx/lerobot_datasets/mamengyiyi/piperx_lf_single/meta/info.json
```

---

## 8. 计算 normalization statistics

OpenPI 在训练前必须先为当前数据集计算 normalization statistics。单臂：

```bash
cd /root/data/my/piperx/openpi

export PATH="/root/data/my/piperx/openpi/.venv/bin:$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/data/my/piperx/openpi_cache
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets
export HF_HOME=/root/data/my/piperx/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/data/my/piperx/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_piperx_single_lf
```

双臂：

```bash
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_piperx_bimanual_lf
```

成功后会写到：

```text
assets/pi05_piperx_single_lf/mamengyiyi/piperx_lf_single/
assets/pi05_piperx_bimanual_lf/mamengyiyi/piperx_lf_bimanual/
```

如果训练时报：

```text
Normalization stats not found
```

说明这一步没跑成功，或者 `repo_id` 和配置里的 `repo_id` 不一致。

---

## 9. 开始微调

### 9.1 单臂训练

```bash
cd /root/data/my/piperx/openpi

export PATH="/root/data/my/piperx/openpi/.venv/bin:$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/data/my/piperx/openpi_cache
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets
export HF_HOME=/root/data/my/piperx/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/data/my/piperx/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

.venv/bin/python scripts/train.py pi05_piperx_single_lf \
  --exp-name piperx_single_001 \
  --overwrite
```

### 9.2 双臂训练

```bash
.venv/bin/python scripts/train.py pi05_piperx_bimanual_lf \
  --exp-name piperx_bimanual_001 \
  --overwrite
```

checkpoint 会保存在：

```text
checkpoints/pi05_piperx_single_lf/piperx_single_001/
checkpoints/pi05_piperx_bimanual_lf/piperx_bimanual_001/
```

例如第 20000 步：

```text
checkpoints/pi05_piperx_single_lf/piperx_single_001/20000/
```

### 9.3 中断后继续训练

如果中途断了，用同一个 `exp-name` 加 `--resume`：

```bash
.venv/bin/python scripts/train.py pi05_piperx_single_lf \
  --exp-name piperx_single_001 \
  --resume
```

不要同时加 `--overwrite`，否则会覆盖旧 checkpoint。

---

## 10. 启动微调后的 policy server

单臂例子：

```bash
cd /root/data/my/piperx/openpi

export PATH="/root/data/my/piperx/openpi/.venv/bin:$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/data/my/piperx/openpi_cache
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets

.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piperx_single_lf \
  --policy.dir=checkpoints/pi05_piperx_single_lf/piperx_single_001/20000 \
  --port=8000 \
  --default-prompt="put the object into the bowl"
```

双臂例子：

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piperx_bimanual_lf \
  --policy.dir=checkpoints/pi05_piperx_bimanual_lf/piperx_bimanual_001/20000 \
  --port=8000 \
  --default-prompt="complete the bimanual manipulation task"
```

`default-prompt` 可以换成采集时 `--task` 使用的任务描述。若 LeRobot 数据中已经有
`task`，训练时会使用 `prompt_from_task=True` 自动取出。

---

## 11. 机器人端部署时的数据对齐

训练时的输入和部署时的输入必须一致：

| 场景 | 状态 | 动作 | 图像 |
| --- | --- | --- | --- |
| 单臂 | `joint_pos`，7 维 | `absolute_joint`，7 维 | `front + follower_wrist` |
| 双臂 | `left_joint_pos + right_joint_pos`，14 维 | `absolute_joint`，14 维 | `front + left_wrist + right_wrist` |

如果训练用的是 `absolute_joint`，部署也要用：

```bash
--action-mode absolute_joint
```

如果训练数据是 30 Hz，部署可以先用 20 到 30 Hz。第一次上真机建议：

- 先不加 `--execute` 做 dry-run
- 确认 policy server 正常返回动作
- 再低速加 `--execute`
- 设置较小的 `--max-joint-delta-rad`
- 保留急停和人工监控

---

## 12. 常见问题

### 12.1 OpenPI 找不到本地 LeRobot 数据

检查：

```bash
echo $HF_LEROBOT_HOME
ls /root/data/my/piperx/lerobot_datasets/mamengyiyi/piperx_lf_single/meta/info.json
```

目录必须等于：

```text
$HF_LEROBOT_HOME/mamengyiyi/piperx_lf_single
```

### 12.2 `Normalization stats not found`

先运行：

```bash
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piperx_single_lf
```

然后确认：

```bash
find assets/pi05_piperx_single_lf -maxdepth 4 -type f
```

### 12.3 显存不够

优先改训练配置里的：

```python
batch_size=32
```

如果还不够，再考虑：

```python
fsdp_devices=8
```

### 12.4 loss 正常下降但真机动作很差

优先检查四件事：

1. `state` 是否是从臂状态，而不是主臂状态
2. `action` 是否是发送给从臂的目标动作
3. 训练和部署的相机顺序是否一致
4. 训练和部署都使用同一种 action mode，例如都是 `absolute_joint`

### 12.5 单臂数据只有一个 wrist camera

这是允许的。配置里设置：

```python
has_left_wrist=True
has_right_wrist=False
```

缺失的 wrist 图像会用黑图补齐，并通过 `image_mask` 告诉模型该路无效。

---

## 13. 推荐的第一次实验设置

单臂先从小规模开始：

```text
episodes: 20 到 50 条
fps: 30
action_mode: absolute_joint
state: joint_pos
images: front + follower wrist
batch_size: 64
num_train_steps: 10k 到 20k
```

先确认：

- LeRobot 可视化正常
- norm stats 能计算
- train loss 能下降
- policy server 能返回 `(horizon, 7)` 动作
- dry-run 中动作数值没有跳变

这些都通了之后，再扩大数据量并尝试双臂训练。
