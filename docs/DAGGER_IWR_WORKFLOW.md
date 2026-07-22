# PiperX DAgger / IWR Workflow

This document records the current PiperX DAgger data pipeline and the training path used for weighted behavior cloning / simplified IWR.

## 1. Pipeline Overview

```text
OpenPI intervention zarr
  -> PiperX-ToolKit: convert OpenPI intervention zarr to LeRobot v3 full/intervention views
  -> openpi-piperx: merge four LeRobot v3 parts into LeRobot v2.1 full/intervention datasets
  -> openpi-piperx: physically clean invalid/stall frames and write piperx.sample_weight
  -> openpi-piperx: compute fast state/action norm stats from parquet
  -> openpi-piperx: run policy-train --no-acp with piperx.sample_weight
```

The current training is not AWR and does not use advantage weighting. It uses supervised BC loss weighted by `piperx.sample_weight`:

```text
loss = sum_i w_i * BC(policy(obs_i), executed_action_i) / sum_i w_i
```

## 2. Convert Zarr To LeRobot V3 On JG

Run from:

```bash
cd /home/ruihao/PiperX-ToolKit
source .venv/bin/activate
```

Main conversion script:

```text
scripts/convert_openpi_intervention_to_lerobot_v3_fast.py
```

Example for the square towel batch:

```bash
python scripts/convert_openpi_intervention_to_lerobot_v3_fast.py \
  --zarr /towel_data/PiperX-ToolKit/datasets/dagger_multi_towel_openpi_intervention_20260719_v1.zarr \
  --output /towel_data/PiperX-ToolKit/lerobot_datasets/ruio248/dagger_multi_towel_iwr_v3_20260722/dagger_square_20260719_v1_full_v3 \
  --repo-id ruio248/dagger_multi_towel_iwr_v3_20260722/dagger_square_20260719_v1_full_v3 \
  --towel-type square \
  --source-id 0 \
  --task "swing fold the towel" \
  --fps 30 \
  --image-writer-threads 8 \
  --overwrite
```

The script writes two datasets:

```text
full view:         ${output}
intervention view: ${output}_intervention
```

Four source batches used in the current run:

| source_id | towel_type | zarr |
|---:|---|---|
| 0 | `square` | `/towel_data/PiperX-ToolKit/datasets/dagger_multi_towel_openpi_intervention_20260719_v1.zarr` |
| 1 | `small_rectangle` | `/home/ruihao/PiperX-ToolKit/datasets/dagger_multi_towel_openpi_intervention_20260720_v2.zarr` |
| 2 | `large_rectangle` | `/home/ruihao/PiperX-ToolKit/datasets/dagger_multi_towel_openpi_intervention_20260720_v3.zarr` |
| 3 | `large_rectangle` | `/towel_data/PiperX-ToolKit/datasets/dagger_multi_towel_openpi_intervention_20260721_v2.zarr` |

## 3. Merge And Clean On The Training Server

Run from:

```bash
ssh new_server_my_2
cd /root/data/my/piperx/openpi
export HF_LEROBOT_HOME=/root/data/my/piperx/lerobot_datasets
```

Merge four v3 full/intervention parts into v2.1:

```text
scripts/merge_dagger_lerobot_v3_parts_to_v21.py
```

Current recommended physical cleaning and IWR sample-weight generation:

```text
scripts/prepare_dagger_iwr_dataset_physical.py
```

`scripts/prepare_dagger_iwr_dataset.py` is the older annotation route that keeps complete episodes and marks trainable frames with `piperx.train_mask`. The current training uses the physical route because it removes rejected frames directly and avoids changing the training loader.

Current full v2.1 output:

```text
ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_full_v21
```

Current intervention v2.1 output:

```text
ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_intervention_v21
```

Current cleaned training dataset:

```text
ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_physically_cleaned_iwr_s60
```

Current cleaned intervention reference:

```text
ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_physically_cleaned_iwr_s60_intervention
```

Important: train only on the cleaned full dataset. Do not append the `_intervention` dataset to training, because those human frames already exist in the full view.

## 4. Inspection Package

Use this script when manually checking original vs cleaned trajectories:

```text
scripts/build_dagger_iwr_inspection_package.py
```

It extracts aligned snippets from:

```text
original full
cleaned full
original intervention
cleaned intervention
```

This is only for visual/manual validation and is not required for training.

## 5. Fast Norm Stats

Use:

```text
scripts/compute_piperx_norm_stats_from_lerobot_parquet.py
```

Do not use the original `scripts/compute_norm_stats.py` for this DAgger dataset unless you intentionally want to decode videos. The fast script reads only parquet columns:

```text
observation.state
action
```

It expands `action` with `action_horizon=60`, matching the OpenPI training loader's action chunking, without decoding RGB videos.

Current norm stats path:

```text
/root/data/my/piperx/openpi/assets/pi05_piperx_bimanual_swing_fold_towel_20260531/ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_physically_cleaned_iwr_s60/norm_stats.json
```

## 6. Training

The current training uses `train_with_rl.py`:

```text
stage: policy-train
ACP: disabled via --no-acp
sample weight field: piperx.sample_weight
```

Current four-GPU setting:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
BATCH_SIZE=128
NUM_TRAIN_STEPS=20000
FSDP_DEVICES=4
SAVE_INTERVAL=5000
KEEP_PERIOD=5000
```

This matches the previous 8-GPU baseline training amount:

```text
8 GPU baseline: batch 256 * 10000 steps = 2,560,000 samples
4 GPU IWR:      batch 128 * 20000 steps = 2,560,000 samples
```

Current checkpoint output:

```text
/root/data/my/piperx/openpi/checkpoints/pi05_piperx_bimanual_swing_fold_towel_20260531/dagger_multi_towel_iwr_weighted_bc_4gpu_bs128_steps20000_save5000_001
```

Monitor:

```bash
tail -f /root/data/my/piperx/logs/dagger_multi_towel_iwr_weighted_bc_4gpu_bs128_steps20000_save5000_001_20260722_191144/train.log
```

Expected healthy metrics:

```text
actor_weight_mean around 1.3
policy_loss finite and decreasing
policy_grad_norm finite
GPU 4-7 busy
```

## 7. Deployment Asset Alias

`serve_policy.py` may look for norm stats using the original config repo id:

```text
ruio248/swing_fold_towel_20260531_my_merged_v21_parts123456789_rebuilt
```

The trained DAgger repo id is:

```text
ruio248/dagger_multi_towel_openpi_intervention_20260719_20260721_physically_cleaned_iwr_s60
```

Before deployment, make sure each deployable checkpoint has an assets alias from the expected old repo id to the DAgger norm stats directory.
