# PiperX DAgger / IWR 流程实现

本文档说明 PiperX DAgger 数据从 OpenPI 人类接管轨迹转换为 LeRobot 数据集、清洗、构造 IWR 权重、计算归一化统计量，并用于加权行为克隆训练的通用实现流程。

文档中只使用占位符，不包含机器名、私有路径、具体数据集日期或实验超参。具体实验的路径、batch size、训练步数和显卡配置应放在单独的 launch 脚本中。

## 1. 总体链路

```text
OpenPI intervention Zarr
  -> 转换为两份 LeRobot v3 视图：full 和 intervention
  -> 合并多批 LeRobot v3 数据为 LeRobot v2.1
  -> 物理清洗无效帧、长停顿和不可训练片段
  -> 写入 IWR 风格的 piperx.sample_weight
  -> 从 parquet 中快速计算 state/action norm stats
  -> 用带 sample weight 的 supervised behavior cloning 训练策略
```

当前实现的训练目标是加权行为克隆，不是 AWR，也不做 value learning 或 advantage weighting：

```text
loss = sum_i weight_i * BC(policy(observation_i), action_i) / sum_i weight_i
```

训练时读取的权重字段是：

```text
piperx.sample_weight
```

## 2. 数据语义

转换后保留标准策略训练输入：

```text
observation.state              float32[14]
action                         float32[14]
observation.images.front       video
observation.images.left_wrist  video
observation.images.right_wrist video
```

`observation.state` 不是 RGB 图像，而是双臂关节状态拼接：

```text
left_joint_pos[7] + right_joint_pos[7]
```

标准 `action` 始终使用实际执行到从臂上的动作：

```text
executed_action_left[7] + executed_action_right[7]
```

因此：

- policy 段的 `action` 是模型实际执行动作。
- intervention 段的 `action` 是人类接管后经过限幅和平滑、实际执行到机械臂上的动作。
- 训练目标始终对齐真实执行动作，而不是单独使用模型原始输出或专家原始输入。

## 3. DAgger 附加字段

除标准字段外，转换和清洗流程会保留以下 PiperX 元数据：

```text
piperx.policy_action
piperx.human_action
piperx.executed_action
piperx.control_source
piperx.intervention_mask
piperx.episode_success
piperx.policy_action_valid
piperx.human_action_valid
piperx.original_episode_index
piperx.original_frame_index
piperx.intervention_segment_index
piperx.source_id
piperx.towel_type_id
```

这些字段的作用：

- `piperx.policy_action`：模型原本希望执行的动作。
- `piperx.human_action`：人类接管时给出的专家动作。
- `piperx.executed_action`：最终实际执行动作，也会写入标准 `action`。
- `piperx.control_source`：当前帧来自 policy 还是 human intervention。
- `piperx.intervention_mask`：当前帧是否处在人类接管状态。
- `piperx.episode_success`：原始 rollout 的成功、失败或未知标签。
- `piperx.original_episode_index` 和 `piperx.original_frame_index`：回溯到原始 Zarr/LeRobot 数据的位置。
- `piperx.source_id` 和 `piperx.towel_type_id`：区分不同采集批次和任务物体类型。

## 4. Full / Intervention 双视图

转换阶段会生成两份 LeRobot v3 数据：

```text
full view:         完整 rollout，包含 policy 段和 intervention 段
intervention view: 只包含连续人类接管片段
```

full view 的 episode 与原始 rollout 一一对应，不主动切分。

intervention view 会按照 `intervention_mask` 查找连续接管区间，每个连续区间独立成为一个 episode。不同 rollout 中的接管段不会合并，同一 rollout 中不连续的接管段也不会拼接。

训练时不要简单地把 full view 和 intervention view 直接拼在一起，因为 intervention 帧已经存在于 full view 中。否则会重复计算人类接管帧。推荐做法是：

- full view 用作主训练来源。
- intervention view 用于检查、清洗参考和权重构造。
- 最终训练只使用清洗后的 full 数据集。

## 5. Zarr 转 LeRobot v3

脚本：

```text
scripts/convert_openpi_intervention_to_lerobot_v3_fast.py
```

通用命令：

```bash
python scripts/convert_openpi_intervention_to_lerobot_v3_fast.py \
  --zarr <RAW_INTERVENTION_ZARR> \
  --output <LEROBOT_V3_FULL_OUTPUT> \
  --repo-id <LEROBOT_V3_FULL_REPO_ID> \
  --towel-type <square|small_rectangle|large_rectangle> \
  --source-id <SOURCE_ID> \
  --task "<TASK_DESCRIPTION>" \
  --fps <FPS> \
  --image-writer-threads <N> \
  --overwrite
```

输出：

```text
full view:         <LEROBOT_V3_FULL_OUTPUT>
intervention view: <LEROBOT_V3_FULL_OUTPUT>_intervention
```

单元测试或 smoke test 可以关闭视频写入：

```bash
python scripts/convert_openpi_intervention_to_lerobot_v3_fast.py \
  --zarr <RAW_INTERVENTION_ZARR> \
  --output <TEST_LEROBOT_V3_FULL_OUTPUT> \
  --repo-id <TEST_REPO_ID> \
  --towel-type <square|small_rectangle|large_rectangle> \
  --source-id <SOURCE_ID> \
  --episodes 2 \
  --no-videos \
  --overwrite
```

每个独立采集批次应使用唯一 `source_id`。如果任务中存在不同物体类别或形态，应通过 `towel_type` 等字段保留下来，便于后续分析和采样。

## 6. 合并多批 LeRobot v3

脚本：

```text
scripts/merge_dagger_lerobot_v3_parts_to_v21.py
```

合并阶段的目标是把多批转换后的 v3 数据整理成两份 v2.1 数据集：

```text
merged full v2.1
merged intervention v2.1
```

合并时应保持 full 和 intervention 两条视图分离。典型输入包括：

```text
多个 full v3 repo path
多个 intervention v3 repo path
full v2.1 输出 repo id/path
intervention v2.1 输出 repo id/path
```

运行前可查看脚本参数：

```bash
python scripts/merge_dagger_lerobot_v3_parts_to_v21.py --help
```

合并后需要检查：

```text
episode 数量是否符合预期
frame 数量是否符合预期
所有 piperx.* 元数据字段是否保留
intervention 数据是否能映射回 full 数据中的人类接管帧
```

## 7. 物理清洗与 IWR 权重

推荐脚本：

```text
scripts/prepare_dagger_iwr_dataset_physical.py
```

该脚本采用“物理清洗”路线：直接裁掉不可训练帧或片段，并写出新的 LeRobot 数据集。这样训练 dataloader 不需要额外理解 `piperx.train_mask`。

清洗阶段通常处理：

```text
非有限数值
长时间停顿
无效尾帧
过短片段
控制源切换边界
episode success 标签
原始 episode/frame 索引
```

输出中会写入：

```text
piperx.sample_weight
```

一个常用的简化 IWR 权重策略是让 policy 样本和 intervention 样本的总加权质量接近：

```text
policy_weight = 1.0
intervention_weight = num_policy_frames / num_intervention_frames
```

实际实现可以根据任务需要调整权重，但应保证：

```text
所有参与训练的 sample_weight 都是有限正数
policy 和 intervention 的权重含义明确
权重构造不重复计算 intervention 帧
```

旧脚本：

```text
scripts/prepare_dagger_iwr_dataset.py
```

这条路线保留完整 episode，并用 `piperx.train_mask` 标记可训练帧。除非训练 loader 明确支持并正确使用 `piperx.train_mask`，否则不推荐作为默认训练输入。

## 8. Inspection Package

脚本：

```text
scripts/build_dagger_iwr_inspection_package.py
```

用于导出小规模人工检查包，对比：

```text
original full
cleaned full
original intervention
cleaned intervention
```

通用命令：

```bash
python scripts/build_dagger_iwr_inspection_package.py \
  --original-full <ORIGINAL_FULL_V21> \
  --cleaned-full <CLEANED_FULL_V21> \
  --original-intervention <ORIGINAL_INTERVENTION_V21> \
  --cleaned-intervention <CLEANED_INTERVENTION_V21> \
  --output <INSPECTION_OUTPUT_DIR> \
  --overwrite
```

inspection package 只用于人工检查，不参与训练。

## 9. 快速计算 Norm Stats

脚本：

```text
scripts/compute_piperx_norm_stats_from_lerobot_parquet.py
```

该脚本只读取 parquet 中的数值字段：

```text
observation.state
action
```

它不会解码三路 RGB 视频，因此适合大规模 PiperX 数据集。对于 DAgger/IWR 训练，如果图像不需要重新统计，优先使用该脚本。

运行前可查看参数：

```bash
python scripts/compute_piperx_norm_stats_from_lerobot_parquet.py --help
```

输出的 `norm_stats.json` 应放在策略配置或 checkpoint loader 能找到的资产目录下。

## 10. 加权行为克隆训练

### 10.1 训练入口

当前推荐入口分为两层：

```bash
# 一键流程脚本：数据检查、norm_stats、dry-run、正式训练
scripts/run_dagger_iwr_4gpu_normstats_train20k.sh

# 底层训练脚本：实际执行 policy-train
scripts/train_with_rl.py
```

一键流程脚本主要做这些事情：

```text
检查 LeRobot 数据集是否存在
检查 observation.state、action、piperx.sample_weight、piperx.control_source、piperx.executed_action 字段
检查训练配置中 discrete_state_input=True
检查 baseline checkpoint 是否存在
按需计算 norm_stats
为部署阶段建立 assets alias
先执行 train_with_rl.py --dry-run
正式启动 weighted BC / 简化 IWR 训练
```

### 10.2 当前训练命令结构

底层训练命令的核心结构如下：

```bash
.venv/bin/python scripts/train_with_rl.py "$CONFIG" \
  --repo-id "$DATA_REPO" \
  --exp-name "$EXP_NAME" \
  --stage policy-train \
  --training-mode full \
  --weight-loader-path "$BASELINE_PARAMS" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --num-workers "$TRAIN_NUM_WORKERS" \
  --fsdp-devices "$FSDP_DEVICES" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --log-interval "$LOG_INTERVAL" \
  --no-acp \
  --policy-sample-weight-field piperx.sample_weight \
  --overwrite \
  --wandb-enabled
```

训练前建议先跑同一命令的 dry-run 版本：

```bash
.venv/bin/python scripts/train_with_rl.py "$CONFIG" \
  --repo-id "$DATA_REPO" \
  --exp-name "$EXP_NAME" \
  --stage policy-train \
  --training-mode full \
  --weight-loader-path "$BASELINE_PARAMS" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --num-workers "$TRAIN_NUM_WORKERS" \
  --fsdp-devices "$FSDP_DEVICES" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --log-interval "$LOG_INTERVAL" \
  --no-acp \
  --policy-sample-weight-field piperx.sample_weight \
  --dry-run
```

### 10.3 当前实验默认参数

一键脚本中的当前默认参数如下：

| 参数 | 当前默认值 | 说明 |
|---|---:|---|
| `CONFIG` | `pi05_piperx_bimanual_swing_fold_towel_20260531` | OpenPI / PiperX 模型和数据配置 |
| `DATA_REPO` | cleaned DAgger IWR LeRobot repo id | 清洗后、带 `piperx.sample_weight` 的训练集 |
| `EXP_NAME` | `dagger_multi_towel_iwr_weighted_bc_4gpu_bs128_steps20000_save5000_001` | checkpoint 实验名 |
| `BATCH_SIZE` | `128` | 全局 batch size |
| `NUM_TRAIN_STEPS` | `20000` | 总训练步数 |
| `FSDP_DEVICES` | `4` | FSDP 使用的 GPU 数 |
| `TRAIN_NUM_WORKERS` | `8` | 训练 dataloader worker 数 |
| `NORM_NUM_WORKERS` | `8` | norm_stats 计算 worker 数 |
| `SAVE_INTERVAL` | `5000` | 每 5000 step 保存一次 checkpoint |
| `KEEP_PERIOD` | `5000` | checkpoint 周期性保留间隔 |
| `LOG_INTERVAL` | `10` | 日志打印间隔 |
| `POLICY_SAMPLE_WEIGHT_FIELD` | `piperx.sample_weight` | supervised BC loss 使用的样本权重字段 |
| `ACP` | disabled by `--no-acp` | 关闭 ACP 条件提示 / value / advantage 路径 |

这组参数是当前实验设置，不是算法本身的硬性要求。四卡训练使用：

```text
batch 128 * 20000 steps = 2,560,000 samples
```

这是为了对齐之前八卡 baseline 的训练样本量：

```text
batch 256 * 10000 steps = 2,560,000 samples
```

### 10.4 `--no-acp` 为什么可以实现当前 IWR

这里的 IWR 指 simplified Intervention Weighted Regression，也就是把人类接管样本作为更高权重的 supervised BC 样本；它不是 AWR，也不依赖 value function 或 advantage。

关键逻辑是：

```text
prepare_dagger_iwr_dataset_physical.py
  -> 根据 control_source / intervention_mask 写入 piperx.sample_weight

train_with_rl.py 的 RLDataset
  -> 从 parquet 读取 piperx.sample_weight
  -> 放入 batch["rl"]["sample_weight"]

train_with_rl.py --stage policy-train --policy-sample-weight-field piperx.sample_weight
  -> 在 policy_train_step 里用该权重加权 supervised BC loss
```

对应的训练代码等价于：

```python
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
per_sample = mean(chunked_loss, axis=-1)
weights = batch["rl"]["sample_weight"]
loss = sum(per_sample * weights) / max(sum(weights), 1e-6)
```

因此优化目标可以写成：

```text
loss = sum_i w_i * BC(policy(observation_i), action_i) / sum_i w_i
```

其中：

```text
w_i = piperx.sample_weight_i
action_i = executed_action_i
```

对 DAgger 数据来说，intervention 段的 `action` 是人类接管后实际执行到机械臂上的动作；policy 段的 `action` 是模型实际执行的动作。只要清洗脚本把 intervention 帧设置为更高 `piperx.sample_weight`，`policy-train` 的 supervised BC loss 就会更重视人类纠偏样本。

`--no-acp` 的作用是关闭 ACP 条件提示、value learning 和 advantage conditioning。它不会关闭 `policy_train_step` 中的 `piperx.sample_weight` 加权分支。所以当前命令：

```bash
--stage policy-train \
--no-acp \
--policy-sample-weight-field piperx.sample_weight
```

表示：

```text
不走 ACP / advantage 路径
只做 policy supervised BC
但 BC loss 按 piperx.sample_weight 加权
```

这就是当前代码路径下的 DAgger / IWR weighted BC 实现。

### 10.5 当前实现不是完整 AWR

当前流程的核心不是重新实现一个完整 RL/AWR 算法，而是在标准 BC 训练中提高人类接管样本的监督权重。这样可以让模型更多学习失败恢复、纠偏和关键接触阶段的人类动作。

当前路径不做：

```text
value function 训练
advantage 估计
exponential advantage weighting
ACP 条件行为克隆
在线强化学习更新
```

如果后续要做真正的 AWR 或 value-based IWR，需要额外定义 return / value / advantage，并在训练目标中显式使用这些量，而不能只依赖 `piperx.sample_weight`。

## 11. 部署注意事项

部署前需要确认 checkpoint 能找到训练数据对应的 `norm_stats.json`。

如果 serving config 仍然使用旧的数据集 asset id，应采取以下一种方式：

```text
更新 serving config 的数据集 asset id
或创建明确记录的 assets alias
```

部署前至少做一次短测试：

```text
检查 robot/CAN 映射
检查相机可用性
检查 policy server 连接
检查 observation/action 维度
检查动作限幅和平滑参数
检查 dry-run 或短时 execute 是否正常
```

## 12. 最小验证清单

正式训练前应确认：

```text
full episode 数量与原始 rollout 数一致
intervention episode 只按连续接管段切分
不同 rollout 的接管段没有被拼接
observation.state 和 action 都是 14 维
三路视频帧数与 parquet 帧数对齐
episode success 标签被保留
original episode/frame 索引被保留
piperx.sample_weight 有限且为正
norm stats 来自最终清洗后的训练数据集
训练时没有把 full 和 intervention 直接重复拼接
```

## 13. 当前实现边界

当前 IWR 实现可以理解为：

```text
DAgger 数据收集 + intervention 样本重加权 + supervised BC
```

它不包含：

```text
value function 训练
advantage 估计
AWR/AC-style policy improvement
在线强化学习更新
```

如果后续要做真正的 AWR 或 value-based IWR，需要在数据中额外定义 return/value/advantage，并在训练目标中显式使用这些量，而不是只读取 `piperx.sample_weight`。
