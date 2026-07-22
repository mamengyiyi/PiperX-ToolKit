# PiperX DAgger / IWR Workflow

This document describes a reusable PiperX DAgger data pipeline for converting
OpenPI human-intervention rollouts into LeRobot datasets, cleaning them, and
training a weighted behavior cloning policy.

The workflow is intentionally written with placeholders instead of machine names,
private paths, or experiment-specific hyperparameters.

## 1. Pipeline Overview

```text
OpenPI intervention Zarr
  -> convert to two LeRobot v3 views: full and intervention
  -> merge multiple LeRobot v3 parts into LeRobot v2.1
  -> physically clean invalid or stalled frames
  -> assign IWR-style sample weights
  -> compute normalization statistics from state/action parquet data
  -> train a policy with weighted supervised behavior cloning
```

The training objective is weighted behavior cloning, not advantage-weighted RL:

```text
loss = sum_i weight_i * BC(policy(observation_i), action_i) / sum_i weight_i
```

The intended weight field is:

```text
piperx.sample_weight
```

## 2. Data Semantics

The converter keeps the standard policy inputs unchanged:

```text
observation.state              float32[14]
action                         float32[14]
observation.images.front       video
observation.images.left_wrist  video
observation.images.right_wrist video
```

`observation.state` is not an image. It is the concatenation of:

```text
left_joint_pos[7] + right_joint_pos[7]
```

`action` is always the executed action:

```text
executed_action_left[7] + executed_action_right[7]
```

Additional PiperX fields are preserved for DAgger/IWR analysis:

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

The full view contains complete rollout episodes. The intervention view contains
only continuous human-intervention segments, with each segment stored as a
separate episode.

Do not train by simply concatenating full and intervention views. Intervention
frames already exist inside the full view, so concatenating them directly would
duplicate those frames. Use the intervention view for inspection, filtering, or
weight construction.

## 3. Convert OpenPI Intervention Zarr To LeRobot V3

Script:

```text
scripts/convert_openpi_intervention_to_lerobot_v3_fast.py
```

Example:

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

The command writes two outputs:

```text
full view:         <LEROBOT_V3_FULL_OUTPUT>
intervention view: <LEROBOT_V3_FULL_OUTPUT>_intervention
```

For smoke tests or unit tests, use:

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

Use a unique `source_id` for each independently collected data batch. Use
`towel_type` or a similar categorical label to preserve task/domain metadata.

## 4. Merge LeRobot V3 Parts

Script:

```text
scripts/merge_dagger_lerobot_v3_parts_to_v21.py
```

The merge step combines multiple converted parts into one LeRobot v2.1 full
dataset and one LeRobot v2.1 intervention dataset. Keep full and intervention
views separate.

Use the script help for the exact argument names in your checkout:

```bash
python scripts/merge_dagger_lerobot_v3_parts_to_v21.py --help
```

A typical merge should provide:

```text
input full v3 repo paths
input intervention v3 repo paths
output full v2.1 repo id/path
output intervention v2.1 repo id/path
```

After merging, verify that all `piperx.*` metadata fields are still present.

## 5. Physically Clean And Assign IWR Weights

Recommended script:

```text
scripts/prepare_dagger_iwr_dataset_physical.py
```

This route physically removes rejected frames or segments and writes
`piperx.sample_weight` for training. It avoids requiring the training dataloader
to understand a separate `piperx.train_mask` field.

The older script:

```text
scripts/prepare_dagger_iwr_dataset.py
```

keeps complete episodes and marks trainable samples with `piperx.train_mask`.
Use it only if the downstream training code is designed to consume that mask.

Use the script help for the exact argument names in your checkout:

```bash
python scripts/prepare_dagger_iwr_dataset_physical.py --help
```

The cleaning policy should be chosen explicitly for the project. Common checks
include:

```text
remove invalid numeric values
trim long stalls
remove or trim unusable camera/state tails
preserve episode success labels
preserve original episode/frame indices
write piperx.sample_weight
```

The cleaned full dataset is the primary training dataset. The cleaned
intervention dataset is useful for auditing human corrections and weight logic.

## 6. Build An Inspection Package

Script:

```text
scripts/build_dagger_iwr_inspection_package.py
```

Use this script to export a small visual package for manual checking:

```bash
python scripts/build_dagger_iwr_inspection_package.py \
  --original-full <ORIGINAL_FULL_V21> \
  --cleaned-full <CLEANED_FULL_V21> \
  --original-intervention <ORIGINAL_INTERVENTION_V21> \
  --cleaned-intervention <CLEANED_INTERVENTION_V21> \
  --output <INSPECTION_OUTPUT_DIR> \
  --overwrite
```

The package is for visual validation only. It is not part of the training input.

## 7. Compute Norm Stats

Script:

```text
scripts/compute_piperx_norm_stats_from_lerobot_parquet.py
```

This script reads only parquet numeric columns:

```text
observation.state
action
```

It avoids decoding videos and is therefore preferred for large PiperX datasets
when image statistics are not needed.

Use the script help for the exact argument names in your checkout:

```bash
python scripts/compute_piperx_norm_stats_from_lerobot_parquet.py --help
```

The resulting norm stats should be placed where the policy config or checkpoint
loader expects dataset assets.

## 8. Train With Weighted Behavior Cloning

The training code should read:

```text
observation.state
observation.images.*
action
piperx.sample_weight
```

and apply `piperx.sample_weight` to the supervised behavior cloning loss.

Choose batch size, number of steps, save interval, and device count according to
the dataset size and available compute. Those values are experiment-specific and
should live in the training launch script, not in this general workflow document.

If the repository also supports RL, ACP, or advantage-estimation paths, disable
or bypass them unless the experiment intentionally uses those objectives.

## 9. Deployment Notes

Before deployment, confirm that the checkpoint can find the norm stats associated
with the training dataset. If the serving config expects a different dataset
asset id, create a documented assets alias or update the config so that the
loader resolves the correct `norm_stats.json`.

Run a dry-run or short execution test before long deployments:

```text
check robot/CAN mapping
check camera availability
check policy server connectivity
check observation/action dimensions
check action limits and smoothing settings
```

## 10. Minimal Validation Checklist

Before using a converted dataset for training:

```text
full episode count matches source rollout count
intervention segments are split only at control-source boundaries
state/action dimensions are 14
video frame counts align with parquet frame counts
episode success labels are preserved
original episode/frame indices are preserved
piperx.sample_weight is finite and positive for trainable samples
norm stats are computed from the intended cleaned training dataset
```
