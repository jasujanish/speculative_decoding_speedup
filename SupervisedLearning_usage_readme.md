# Qwen3 Supervised Depth Usage

This document describes the supervised-depth Modal workflow in
`Learning-to-Draft-main/supervised_depth_modal/modal_supervised_depth_qwen3.py`.

The workflow replaces LTD's PPO depth policy with a supervised regressor. It
does not train or use a size policy: verification size stays fixed at `60`, and
the supervised model decides whether Eagle3 should continue expanding the draft
tree.

## Models

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

## Policy Variants

Use `--policy-variant` when collecting or training. The two intended variants
are `base` and `titan`.

### Base

The default `base` policy uses the original 129-dimensional observation and the
original greedy objective:

```text
target_delta(depth) = throughput(depth + 1) - throughput(depth)
```

Collection is sweep-based internally, but `--total-timesteps` means the minimum
number of labeled depth-state records to collect. The collector keeps complete
sweeps, so `collected_examples` can be slightly larger than `total_timesteps`.

### Titan

The `titan` policy uses the 135-dimensional observation with score summary
features and a larger MLP:

```text
135 -> 1664 -> 384 -> 128 -> 1
```

This is 915,073 trainable parameters.

Titan uses Q-style best-future labels:

```text
Q_stop(depth) = throughput(depth)
Q_continue(depth) = max future throughput over deeper depths
target_delta(depth) = Q_continue(depth) - Q_stop(depth)
```

Titan also uses sweep-based collection internally. Each sweep is expanded up to
`max_draft_depth`, then every depth state in that sweep receives a label.

Base outputs keep the original directory names. Titan adds a suffix:

- `dataset_t<timesteps>` and `train_t<timesteps>` for base.
- `dataset_t<timesteps>_titan` and `train_t<timesteps>_titan` for Titan.

## 1. Setup

Create the Modal virtual environment from the repository root, then run the
workflow commands from `Learning-to-Draft-main`:

```bash
uv venv .venv/modal_supervised_depth
source .venv/modal_supervised_depth/bin/activate
uv pip install modal
modal setup
cd Learning-to-Draft-main
```

## 2. Collect Dataset

Collect base data:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action collect-depth-dataset \
  --model-preset qwen3_8b \
  --total-timesteps 10000
```

Collect Titan data:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action collect-depth-dataset \
  --model-preset qwen3_8b \
  --total-timesteps 10000 \
  --policy-variant titan
```

Default output root:

- `/results/<preset>/supervised_depth`

Generated files:

- `split_data/humaneval_train/question.jsonl`: deterministic HumanEval training split.
- `split_data/humaneval_val/question.jsonl`: deterministic HumanEval validation split.
- `dataset_t<timesteps>/dataset.jsonl`: base dataset.
- `dataset_t<timesteps>_titan/dataset.jsonl`: Titan dataset.
- `dataset_manifest.json`: collection metadata in each dataset directory.

Important `dataset.jsonl` fields:

- `observation`: 129-dimensional for base, 135-dimensional for Titan.
- `target_delta`: supervised regression target.
- `continue_label`: `1` if `target_delta > 0`, else `0`.
- `target_type`: `greedy_next_depth` for base, `q_best_future` for Titan.
- `draft_depth`, `context_length`, `entropy`.
- `stop_throughput`, `next_throughput`.
- `stop_accept_length`, `next_accept_length`.

Additional Titan fields:

- `score_max`, `score_mean`, `score_std`, `score_min`.
- `entropy_delta`.
- `eos_logprob_from_last_drafter_logits`.
- `q_stop_throughput`.
- `q_continue_throughput`.
- `best_future_depth`.
- `best_future_accept_length`.

Important `dataset_manifest.json` fields:

- `total_timesteps`: requested minimum number of labeled depth-state records.
- `collected_sweeps`, `collected_examples`.
- `draft_model_calls`, `target_model_calls`.
- `fixed_total_token`: fixed verification size, currently `60`.
- `max_draft_depth`: maximum explored draft depth, currently `12`.
- `observation_dim`: `129` for base; `135` for Titan.
- `policy_variant`: `base` or `titan`.
- `target_type`: `greedy_next_depth` for base, `q_best_future` for Titan.
- `collection_target_flops`, `collection_draft_flops`, `collection_total_flops`.
- `flop_estimate_method`: `2 * parameter_count * forward_calls`.

If the matching manifest already exists for the requested `total_timesteps` and
`policy_variant`, the workflow reuses it instead of recollecting.

## 3. Train

Train base:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action train-depth \
  --model-preset qwen3_8b \
  --total-timesteps 10000 \
  --epochs 20 \
  --checkpoint-epochs 20 \
  --batch-size 256 \
  --lr 0.001
```

Train Titan:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action train-depth \
  --model-preset qwen3_8b \
  --total-timesteps 10000 \
  --epochs 20 \
  --checkpoint-epochs 20 \
  --policy-variant titan \
  --batch-size 256 \
  --lr 0.001
```

`--epochs` is the number of full passes over the supervised training split. The
trainer derives `optimizer_steps` as `epochs * train_batches_per_epoch`.
`--checkpoint-epochs` checkpoints at epoch boundaries, plus the initial step-1
checkpoint and the final-epoch checkpoint.

Training output directories:

- `/results/<preset>/supervised_depth/train_t<timesteps>/`
- `/results/<preset>/supervised_depth/train_t<timesteps>_titan/`

Generated files:

- `training_metrics.jsonl`
- `training_summary.json`
- `time_summary.jsonl`
- `supervised_depth_model_step_<N>.pt`
- `supervised_depth_model_best.pt`
- `validation/<checkpoint_stem>.jsonl`
- `validation_metrics.jsonl`
- `validation_selection.json`
- `supervised_depth_model.pt`

Validation selection uses the deterministic HumanEval validation split. The
checkpoint with the highest validation speedup is promoted, with average
acceptance length and checkpoint step used as tie-breakers.

## 4. Benchmark

The benchmark suite now defaults to only:

- `mt_bench`
- `gsm8k`

Benchmark base:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_8b \
  --depth-model-path qwen3_8b/supervised_depth/train_t10000/supervised_depth_model.pt \
  --output-subdir qwen3_8b/supervised_depth/benchmark_outputs_base \
  --model-label "Qwen3 8B Base"
```

Benchmark Titan:

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_8b \
  --depth-model-path qwen3_8b/supervised_depth/train_t10000_titan/supervised_depth_model.pt \
  --output-subdir qwen3_8b/supervised_depth/benchmark_outputs_titan \
  --model-label "Qwen3 8B Titan"
```

The benchmark script reads checkpoint metadata, so no `--policy-variant` is
needed for benchmarking.

Outputs:

- `<output-subdir>/baseline/<dataset>.jsonl`
- `<output-subdir>/eagle3/<dataset>.jsonl`
- `<output-subdir>/supervised_depth/<dataset>.jsonl`
- `<output-subdir>/summary.csv`
- sidecar manifests beside each benchmark JSONL

## 5. Download Result Files Locally

```bash
modal volume get ltd-qwen3-results qwen3_8b/supervised_depth ./local_path
```

## 6. Remove Files on Modal

```bash
modal volume rm -r ltd-qwen3-results qwen3_8b/supervised_depth
```
You should clear files before starting the next run.