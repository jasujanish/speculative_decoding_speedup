# Qwen3 LTD Usage

This document describes the LTD Modal workflow in
`Learning-to-Draft-main/modal_qwen3.py`.

## Models

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

## 1. Setup

Create the Modal virtual environment from the repository root, then run the
workflow commands from `Learning-to-Draft-main`:

```bash
uv venv .venv/modal
source .venv/modal/bin/activate
uv pip install modal
modal setup
cd Learning-to-Draft-main
```

## 2. Iterative Training

The paper-style workflow runs one explicit stage per Modal execution and stores
state in `/results/<preset>/iterative/iterative_state.json`.

Check the next required stage:

```bash
modal run modal_qwen3.py \
  --action iterative-status \
  --model-preset qwen3_8b
```

Run that exact stage:

```bash
modal run modal_qwen3.py \
  --action iterative-train \
  --model-preset qwen3_8b \
  --current-stage iter0_size \
  --size-total-timesteps 10000 \
  --depth-total-timesteps 10000 \
  --batch-size 256 \
  --n-steps 2048 \
  --lr 0.001 \
  --validation-fraction 0.2 \
  --split-seed 42
```
Here `iter0_size` is the first stage. This is the order of stages:

- `iter0_size`
- `iter0_depth`
- `iter1_size`
- `iter2_depth`
- `iter3_size`
- `iter4_depth`

Defaults:

- dataset: `humaneval`
- deterministic HumanEval split: 80/20, seed `42`
- size training: `100000` timesteps
- depth training: `1000000` timesteps
- PPO: `batch_size=256`, `n_steps=2048`, `n_epochs=20`, `lr=1e-3`
- size policy: `gamma=0.9`, `pi_arch=[1024, 256]`, `vf_arch=[1024, 256]`
- depth policy: `gamma=0.999`, `pi_arch=[1024]`, `vf_arch=[1024, 256]`
- checkpoint frequency: every `10000` timesteps

**Important Note**: there are 2 supported models, `qwen3_8b` and `qwen3_14b`

### 3a. Stage Outputs

Each stage directory is stored under `/results/<preset>/iterative/<stage>/`:

- `training_metrics.jsonl`: PPO step, rollout, checkpoint, and best-model events.
- `training_summary.json`: latest checkpoint step, best training-reward step,
  canonical final checkpoint, metrics log path, and `dataset_manifest.json` path.
- `dataset_manifest.json`: training drafter/target call counts, validation
  drafter/target call counts after checkpoint selection, FLOP estimates, model
  paths, prompt file, and phase metadata.
- `time_summary.jsonl`: one JSONL summary record with `event`, `stage_name`,
  `phase_type`, `total_timesteps`, `total_time_seconds`,
  `training_time_seconds`, `validation_time_seconds`, `training_flops`,
  `validation_flops`, and `total_flops`.
- `validation_metrics.jsonl`: one record per validation candidate plus the
  selected checkpoint.
- `validation_selection.json`: final validation summary and promoted checkpoint.
- `ppo_speculative_decoder_controller_step_<N>.zip`: periodic checkpoints.
- `ppo_speculative_decoder_controller_best.zip`: best checkpoint by training
  reward.
- canonical promoted checkpoint:
  - size stage: `ppo_speculative_decoder_controller_rebuttal.zip`
  - depth stage: `ppo_speculative_decoder_controller_v1_single_action.zip`

Validation selection is always based on the HumanEval validation split. The
checkpoint with the highest validation speedup is promoted and used by the next
stage.

`time_summary.jsonl` is overwritten with one summary record for the stage. In
iterative training, it includes both training and validation wall-clock time. In
standalone `train-size` and `train-depth` runs, `validation_time_seconds` and
`validation_flops` are `0` because those actions do not run validation
selection.

### 3b. Final LTD Checkpoints

After `iter3_size` and `iter4_depth` complete:

- token model: `/results/<preset>/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip`
- depth model: `/results/<preset>/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip`

When passing checkpoint paths back to `modal_qwen3.py`, use either absolute
remote paths or paths relative to `/results`.

### 3c. Repeated running
This command is run repeatedely with multiple stages. The first 2 runs are done to train the initial policies. The last 4 runs perform co-optimization.

## 4. Benchmark and CSV

With co-optimization (`iter3_size` and `iter4_depth`):

```bash
modal run modal_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_14b \
  --token-model-path qwen3_14b/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip \
  --model-label "Qwen3 14B"
```

Without co-optimization (`iter0_size` and `iter0_depth`):

```bash
modal run modal_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_14b \
  --token-model-path qwen3_14b/iterative/iter0_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter0_depth/ppo_speculative_decoder_controller_v1_single_action.zip \
  --output-subdir qwen3_14b/iterative/benchmark_outputs_iter0 \
  --model-label "Qwen3 14B iter0"
```

Default benchmark datasets are `mt_bench` and `gsm8k`.

Outputs:

- `/results/<preset>/benchmark_outputs/baseline/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/eagle3/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/ltd/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/summary.csv`

The summary CSV has `model`, `method`, per-dataset speedup/tau columns, and
`mean_speedup`/`mean_tau`.

## 5. Download Result Files Locally

`modal volume get` downloads directories recursively when the remote path is a
folder.

Download one stage:
```bash
modal volume get ltd-qwen3-results qwen3_8b/iterative/iter0_size ./local_path
```

Download the full iterative training tree:
```bash
modal volume get ltd-qwen3-results qwen3_8b/iterative ./local_path
```

## 6. Remove Results on Modal

Use cleanup commands only after downloading any artifacts you need:

```bash
modal volume rm -r ltd-qwen3-results qwen3_8b/iterative
```
