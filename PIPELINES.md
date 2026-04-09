# Pipeline Guide For AI Agents

This document is the authoritative guide to the registered training and evaluation pipelines in this repository.

If you are an AI agent operating on this codebase, read this file before changing any training or evaluation command, dataset logic, or prompt logic.

The pipeline system is implemented in:

- [gr00t/experiment/pipelines.py](/data1/yfl_data/DynaHOI/gr00t/experiment/pipelines.py)
- [gr00t/experiment/data_config.py](/data1/yfl_data/DynaHOI/gr00t/experiment/data_config.py)
- [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py)
- [gr00t/model/transforms.py](/data1/yfl_data/DynaHOI/gr00t/model/transforms.py)
- [gr00t/utils/eval.py](/data1/yfl_data/DynaHOI/gr00t/utils/eval.py)
- [scripts/finetune_policy.py](/data1/yfl_data/DynaHOI/scripts/finetune_policy.py)
- [scripts/eval_policy.py](/data1/yfl_data/DynaHOI/scripts/eval_policy.py)

## Overview

There are currently 4 registered pipelines:

1. `baseline`
2. `Local`
3. `Global`
4. `LoGo`

They are all routed through the same unified CLI entrypoints:

- training: [scripts/finetune_policy.py](/data1/yfl_data/DynaHOI/scripts/finetune_policy.py)
- evaluation: [scripts/eval_policy.py](/data1/yfl_data/DynaHOI/scripts/eval_policy.py)

The pipeline name selects:

- argument validation
- dataset construction
- VLM prompt structure
- rollout behavior during Unity evaluation
- result tag naming

## Global Rules

These rules apply to all pipelines unless explicitly stated otherwise.

### Model loading

Training always loads the model through:

- [GR00T_N1_5.from_pretrained(...)](/data1/yfl_data/DynaHOI/scripts/finetune_policy.py)

The pipeline does not change model class selection. It only changes:

- dataset assembly
- transform behavior
- action head horizon/dimension configuration

### State and action space

All current hand pipelines use:

- `action_dim = 18`
- single video key: `video.ego_view`
- single state key: `state.left_hand`
- single action key: `action.left_hand`

The relevant data configs are:

- `baseline`
- `Local`
- `Global`
- `LoGo`

### Unified CLI parameters

The main user-facing pipeline switches are:

- `--pipeline`
- `--data-config`
- `--window-length`
- `--motion-hint-ratio`
- `--motion-hint-num-frames`

Not every pipeline supports every parameter. Unsupported combinations are rejected explicitly in [gr00t/experiment/pipelines.py](/data1/yfl_data/DynaHOI/gr00t/experiment/pipelines.py).

### Evaluation uses per-trajectory length, not `--steps`

In [scripts/eval_policy.py](/data1/yfl_data/DynaHOI/scripts/eval_policy.py), the `steps` argument is documented as unused. Real rollout length comes from:

- `dataset.trajectory_lengths[traj_id]`

### Motion hint cache is a hard prerequisite

Any pipeline with `use_motion_hint=True` requires a precomputed cache under:

- `meta/motion_hint_farneback/ratio_xxx`

This is validated in [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py).

If the cache is missing or inconsistent, dataset initialization fails explicitly.

The cache algorithm is expected to be:

- `farneback_weighted_uv_magnitude_v1`

The cache should be created before training or evaluation using:

- [scripts/precompute_motion_hints.py](/data1/yfl_data/DynaHOI/scripts/precompute_motion_hints.py)

## Pipeline 1: `baseline`

### Purpose

This is the plain single-frame policy path. It does not use adjacent history windows and does not use motion hints.

### Canonical data config

- `baseline`

### Transform type

Uses:

- `vlm_type="base"`

in [gr00t/experiment/data_config.py](/data1/yfl_data/DynaHOI/gr00t/experiment/data_config.py).

### Video input seen by the model

The model sees only the current frame from the dataset transform pipeline.

There is no extra frame concatenation in [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py).

### Prompt semantics

This path uses the generic GR00T conversation path from [gr00t/model/transforms.py](/data1/yfl_data/DynaHOI/gr00t/model/transforms.py).

There is no baseline-style prompt explaining multiple image roles because there is no extra image structure here.

### Training dataset behavior

Training uses:

- `build_default_train_dataset(...)`

This means:

- no `add_observe_frames`
- no `use_motion_hint`
- no extra availability filtering beyond normal dataset length

### Evaluation behavior

Evaluation uses:

- `get_and_send_action(...)`

in [gr00t/utils/eval.py](/data1/yfl_data/DynaHOI/gr00t/utils/eval.py).

Rollout starts at:

- frame `0`

At every `action_horizon` boundary:

- Unity sends current observation
- Python adds task text
- policy predicts a chunk of actions
- actions are sent back to Unity

### Parameter constraints

Allowed:

- `action_dim = 18`
- positive `action_horizon`

Must remain unset or default:

- `window_length = 0`
- `motion_hint_ratio = 0.25`
- `motion_hint_num_frames = 6`

If these are changed, validation fails.

### When to use

Use this pipeline when you want:

- no temporal RGB context
- no precomputed motion prior
- plain current-frame policy behavior

## Pipeline 2: `Local`

### Purpose

This is the short-term temporal RGB baseline.

It prepends `K` history frames before the current observation, where:

- `K = window_length`
- if `observe_frame_offsets` is omitted, those frames are the default contiguous window
- if `observe_frame_offsets` is provided, those frames are sampled at the explicit offsets

### Canonical data config

- `Local`

### Transform type

Uses:

- `vlm_type="baseline"`

### Video input seen by the model

At dataset time, the video tensor is assembled as:

1. history frames in the configured offset order
2. `current`

This is done in [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py) through:

- `get_adjacent_observe_frames(...)`
- `get_video(...)`

### Prompt semantics

The baseline prompt in [gr00t/model/transforms.py](/data1/yfl_data/DynaHOI/gr00t/model/transforms.py) explicitly says:

- the first `window_length` images are history frames sampled from earlier timesteps before the current observation
- the last image is the current frame

This is the correct interpretation of the baseline window.

### Training dataset behavior

Training uses:

- `build_baseline_train_dataset(...)`

This sets:

- `add_observe_frames=True`
- `observe_frame_num=window_length`
- `observe_frame_offsets=observe_frame_offsets` when provided

Dataset validity rule:

- only steps with `base_index >= max(history_offsets)` are included

This filtering is done in:

- `LeRobotSingleDataset._get_all_steps()`

So the early steps of each trajectory are dropped for training.

### Evaluation behavior

Evaluation uses:

- `get_and_send_action_baseline(...)`

Rollout start:

- `start_frame_idx = max(history_offsets)`

This means:

- no actions are produced before enough adjacent history exists
- first action chunk starts only after `window_length` past frames are available

At each inference point:

1. Unity provides the current frame only
2. Python fetches the configured history frames from the dataset video
3. those frames are resized to Unity frame resolution
4. final sequence becomes `history + current`
5. sequence is sent into the policy

### Parameter constraints

Allowed:

- positive `action_dim`
- positive `action_horizon`
- positive `window_length`
- optional explicit `observe_frame_offsets` with the same length as `window_length`

Unsupported:

- changing `motion_hint_ratio`
- changing `motion_hint_num_frames`

For this pipeline, motion hint parameters are supposed to stay at defaults and are not used.

### When to use

Use this pipeline when you want:

- short-term causal RGB history only
- no precomputed long-term motion prior

## Pipeline 3: `Global`

### Purpose

This is the long-term precomputed motion-prior pipeline.

It does not use adjacent RGB history. Instead, it feeds:

1. one precomputed motion hint image
2. one current frame

### Canonical data config

- `Global`

### What the motion hint actually is

The motion hint is a precomputed RGB image stored on disk.

It is derived from the first `motion_hint_ratio` portion of the trajectory using Farneback optical flow aggregation.

Key implementation:

- [compute_motion_hint_from_frames(...)](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py)

High-level algorithm:

1. convert prefix frames to grayscale
2. compute Farneback optical flow between consecutive prefix frames
3. aggregate all flow fields with increasing temporal weights
4. encode:
   - weighted horizontal flow into one channel
   - weighted vertical flow into one channel
   - magnitude into one channel

This produces one RGB motion-hint image per episode.

### Transform type

Uses:

- `vlm_type="motion_hint"`

### Video input seen by the model

At dataset time, the video tensor becomes:

1. `motion_hint`
2. `current`

This is assembled in:

- `LeRobotSingleDataset.get_video(...)`

### Prompt semantics

The prompt explicitly says:

- first image is a precomputed motion hint from the first 20% of the trajectory
- second image is the current observation frame

Note:

- the code text says “first 20%�? but the real ratio is controlled by `motion_hint_ratio`
- agents should treat the prompt wording as descriptive, and the actual source of truth as the CLI/config value

### Training dataset behavior

Training uses:

- `build_motion_hint_train_dataset(...)`

This sets:

- `use_motion_hint=True`
- `motion_hint_ratio=config.motion_hint_ratio`
- `motion_hint_num_frames=config.motion_hint_num_frames`

Dataset validity rule:

- a trajectory is kept only if a valid motion hint cache file exists
- its prefix length is large enough

The filtering is handled by:

- `_validate_motion_hint_cache()`
- `_resolve_valid_motion_hint_trajectory_ids()`

Step-level start rule:

- training starts from `start_index = get_prefix_frame_count(trajectory_length, motion_hint_ratio)`

So the policy only acts after the observation-only prefix segment.

### Evaluation behavior

Evaluation uses:

- `get_and_send_action_motion_hint(...)`

Rollout start:

- `start_frame_idx = dataset.get_motion_hint_start_index(traj_id)`

This equals the prefix-frame count implied by `motion_hint_ratio`.

At each inference point:

1. Unity provides the current frame
2. Python loads the precomputed motion hint once
3. the hint is resized to the current frame resolution
4. final sequence becomes `motion_hint + current`
5. the sequence is sent into the policy

### Parameter constraints

Allowed:

- `action_dim = 18`
- positive `action_horizon`
- positive `motion_hint_num_frames`
- `0 < motion_hint_ratio < 1`

Unsupported:

- `window_length != 0`

### When to use

Use this pipeline when you want:

- long-term global motion prior from the beginning of the trajectory
- no short-term adjacent RGB window

## Pipeline 4: `LoGo`

### Purpose

This is the fused pipeline.

It combines:

- short-term adjacent RGB history
- long-term precomputed Farneback motion hint
- current frame

### Canonical data config

- `LoGo`

### Transform type

Uses:

- `vlm_type="baseline_motion_hint"`

### Video input seen by the model

The final image order is:

1. `prevK`
2. `prevK-1`
3. ...
4. `prev1`
5. `motion_hint`
6. `current`

More generally:

- `window_length` history frames
- optional explicit `observe_frame_offsets` for the local history branch
- then 1 motion hint image
- then 1 current frame

This is assembled in [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py).

### Prompt semantics

The fused prompt in [gr00t/model/transforms.py](/data1/yfl_data/DynaHOI/gr00t/model/transforms.py) says:

- first `window_length` images are the adjacent frames immediately before the current observation
- next image is a precomputed motion hint summarizing the moving object trajectory during the first part of the episode
- last image is the current frame

This is the most important semantic contract for the fused pipeline.

### Training dataset behavior

Training uses:

- `build_baseline_motion_hint_train_dataset(...)`

This sets both:

- `add_observe_frames=True`
- `use_motion_hint=True`

Dataset validity rules:

- the trajectory must have a valid motion-hint cache
- the step index must satisfy both requirements:
  - `base_index >= max(history_offsets)`
  - `base_index >= motion_hint_start_index`

The actual start index is:

- `max(max(history_offsets), motion_hint_start_index)`

This is enforced in:

- `LeRobotSingleDataset._get_all_steps()`

### Evaluation behavior

Evaluation uses:

- `get_and_send_action_baseline_motion_hint(...)`

Rollout start:

- `start_frame_idx = max(max(history_offsets), motion_hint_start_index)`

This means the fused pipeline is causal in both senses:

- it waits until enough local adjacent history exists
- it also waits until the prefix segment used by the motion hint has been fully observed

At each inference point:

1. Unity provides the current frame
2. Python fetches the configured local history frames
3. Python loads the precomputed motion hint
4. everything is resized to the current frame resolution
5. final sequence becomes `history + motion_hint + current`
6. sequence is sent into the policy

### Parameter constraints

Allowed:

- `action_dim = 18`
- positive `action_horizon`
- positive `window_length`
- optional explicit `observe_frame_offsets` with the same length as `window_length`
- positive `motion_hint_num_frames`
- `0 < motion_hint_ratio < 1`

Unlike the other specialized pipelines, this one requires both temporal parameter families at once.

### When to use

Use this pipeline when you want:

- local short-term RGB dynamics
- long-term global motion prior
- current-frame appearance

This is the richest hand pipeline currently registered.

## Training-Side Model Configuration Differences

Pipeline-specific action-head configuration is controlled in [gr00t/experiment/pipelines.py](/data1/yfl_data/DynaHOI/gr00t/experiment/pipelines.py).

### `baseline`

Uses:

- `configure_our_model_for_train(...)`

This:

1. recreates action head if horizon differs from the current model
2. replaces the action head while keeping the existing DiT
3. forces `action_dim=18`

### `Local`

Uses:

- `configure_baseline_model_for_train(...)`

This is similar to `baseline` but allows `action_dim=config.action_dim`.

In practice, the hand setup is still 18-dimensional.

### `Global`

Uses:

- `configure_our_model_for_train(...)`

So it follows the same action-head handling as `baseline`.

### `LoGo`

Also uses:

- `configure_our_model_for_train(...)`

So the fused pipeline:

- does not introduce any new trainable modules
- only changes the visual input packaging and prompt semantics

This is important when continuing from an existing checkpoint such as `ObAct`.

## Output Directory Behavior

There are two output-dir policies.

### Identity output dir

Used by:

- `Local`

Output path is used exactly as provided.

### Timestamped output dir

Used by:

- `baseline`
- `Global`
- `LoGo`

The actual directory becomes:

- `output_dir/<MM><DD>:<HH>_<dataset_suffix>`

This is implemented by:

- `resolve_output_dir_with_timestamp(...)`

## Result Tag Naming

Eval outputs are named per pipeline.

### `baseline`

Result tag:

- `<pipeline>:<dataset_tag>`

### `Local`

Result tag:

- `<pipeline>:window_<window_length>:<dataset_tag>` for default contiguous history
- `<pipeline>:offsets_<offsets>:<dataset_tag>` for explicit history offsets

### `Global`

Result tag:

- `<pipeline>:<dataset_tag>`

### `LoGo`

Result tag:

- `<pipeline>:window_<window_length>:ratio_<motion_hint_ratio>:frames_<motion_hint_num_frames>:<dataset_tag>` for default contiguous history
- `<pipeline>:offsets_<offsets>:ratio_<motion_hint_ratio>:frames_<motion_hint_num_frames>:<dataset_tag>` for explicit history offsets

## Recommended Valid Pairings

Use these pairings unless you have a concrete reason not to.

### Single-frame policy

- `--pipeline baseline`
- `--data-config baseline`

### Adjacent RGB baseline

- `--pipeline Local`
- `--data-config Local`

### Farneback motion-hint pipeline

- `--pipeline Global`
- `--data-config Global`

### Fused adjacent-window + motion-hint pipeline

- `--pipeline LoGo`
- `--data-config LoGo`

## Recommended Commands

### Train fused pipeline from an existing checkpoint

```bash
python /data1/yfl_data/DynaHOI/scripts/finetune_policy.py \
  --pipeline LoGo \
  --data-config LoGo \
  --base-model-path /data1/yfl_data/DynaHOI/gr00t/checkpoints/ObAct \
  --window-length 5 \
  --motion-hint-ratio 0.2 \
  --motion-hint-num-frames 6
```

### Evaluate fused pipeline

```bash
python /data1/yfl_data/DynaHOI/scripts/eval_policy.py \
  --pipeline LoGo \
  --data-config LoGo \
  --model-path /path/to/checkpoint \
  --window-length 5 \
  --observe-frame-offsets 10 5 3 2 1 \
  --motion-hint-ratio 0.2 \
  --motion-hint-num-frames 6
```

## Common Failure Modes

### Unknown pipeline

Cause:

- pipeline name not registered in [gr00t/experiment/pipelines.py](/data1/yfl_data/DynaHOI/gr00t/experiment/pipelines.py)

### Empty eval dataset

Cause:

- no valid steps remain after start-index filtering

Common triggers:

- `window_length` too large
- `motion_hint_ratio` too large
- motion hint availability removes too many trajectories

### Motion hint cache missing

Cause:

- missing manifest or image files in `meta/motion_hint_farneback`

### Wrong data-config / pipeline pairing

Cause:

- using a transform whose `vlm_type` does not match the pipeline’s expected image layout

Examples:

- `Local` with `Global`
- `Global` with `Local`

### Invalid parameter family for a pipeline

Cause:

- setting `window_length` for `Global`
- setting motion-hint parameters for `Local`

Validation is intentionally strict and should not be bypassed.

## Agent Checklist

Before modifying a command, pipeline, or dataset path, an agent should verify:

1. Which pipeline is intended.
2. Which data config matches that pipeline.
3. Whether motion hint cache is required.
4. What the model will actually see as image order.
5. What the rollout start index is.
6. Whether the current checkpoint was trained on the same input structure.

If any of these are unclear, inspect:

- [gr00t/experiment/pipelines.py](/data1/yfl_data/DynaHOI/gr00t/experiment/pipelines.py)
- [gr00t/data/dataset.py](/data1/yfl_data/DynaHOI/gr00t/data/dataset.py)
- [gr00t/model/transforms.py](/data1/yfl_data/DynaHOI/gr00t/model/transforms.py)

Do not infer pipeline semantics from old scripts or old experiment names alone.


