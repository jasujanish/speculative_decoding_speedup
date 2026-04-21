# Qwen3 Supervised Depth Usage

This document describes the supervised-depth Modal workflow in
`Learning-to-Draft-main/supervised_depth_modal/modal_supervised_depth_qwen3.py`.

The workflow replaces the LTD PPO depth policy with a supervised regressor. It
does not train or use a size policy: verification size stays fixed at `60`, and
the supervised model decides whether tree growth should continue at the next
depth.

## Models

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

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

## 2. Download Models

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action download-models \
  --model-preset qwen3_8b

modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action download-models \
  --model-preset qwen3_14b
```

Weights are cached in the `ltd-qwen3-models` Modal volume. These weights are
shared by the LTD workflow and the supervised-depth workflow.

## 3. Collect Dataset

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action collect-depth-dataset \
  --model-preset qwen3_8b \
  --total-timesteps 20000
```

Default output root:

- `/results/<preset>/supervised_depth`

Generated files:

- `split_data/humaneval_train/question.jsonl`: deterministic HumanEval training split.
- `split_data/humaneval_val/question.jsonl`: deterministic HumanEval validation split.
- `dataset_t<timesteps>/dataset.jsonl`: one supervised training record per
  collected depth-decision state.
- `dataset_t<timesteps>/dataset_manifest.json`: collection metadata.

Important `dataset.jsonl` fields:

- `observation`: 129-dimensional depth observation vector. The final entry is
  the entropy feature added by this workflow.
- `target_delta`: `throughput(next depth) - throughput(stop now)`.
- `continue_label`: `1` if `target_delta > 0`, else `0`.
- `draft_depth`, `context_length`, `entropy`.
- `stop_throughput`, `next_throughput`.
- `stop_accept_length`, `next_accept_length`.

Important `dataset_manifest.json` fields:

- `total_timesteps`, `collected_examples`, `episodes_started`.
- `draft_model_calls`, `target_model_calls`.
- `fixed_total_token`: fixed verification size, currently `60`.
- `max_draft_depth`: maximum explored draft depth, currently `12`.
- `observation_dim`: currently `129`.
- `dataset_path`, `question_file`, `base_model_path`, `ea_model_path`.
- `total_collection_time_seconds`.
- `target_model_parameters`, `draft_model_parameters`.
- `collection_target_flops`, `collection_draft_flops`,
  `collection_total_flops`.
- `flop_estimate_method`: `2 * parameter_count * forward_calls`.

If `dataset_manifest.json` already exists for the requested `total_timesteps`,
`collect-depth-dataset` returns the existing manifest instead of recollecting.

## 4. Train the Supervised Depth Policy

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action train-depth \
  --model-preset qwen3_8b \
  --total-timesteps 20000 \
  --epochs 1 \
  --checkpoint-epochs 1 \
  --batch-size 256 \
  --lr 0.001
```

`--epochs` is the number of full passes over the supervised training split. The
trainer derives `optimizer_steps` as `epochs * train_batches_per_epoch` and
records both values in the training summary. `--checkpoint-epochs` controls how
often to checkpoint at epoch boundaries; the trainer always also writes an
initial step-1 checkpoint and a final-epoch checkpoint. `train-depth` also
collects the dataset first if `dataset_t<timesteps>/dataset_manifest.json` does
not already exist.

Training output directory:

- `/results/<preset>/supervised_depth/train_t<timesteps>/`

Generated files:

- `training_metrics.jsonl`: `step`, `checkpoint`, and `best_model` events.
  Step events include train loss, internal validation loss, and validation sign
  accuracy.
- `training_summary.json`: dataset path, optimizer settings, internal
  train/validation example counts, best validation loss, checkpoint paths,
  supervised model parameter count, examples seen, and estimated training FLOPs.
- `time_summary.jsonl`: one summary record with `phase_type`,
  `total_timesteps`, `epochs`, `checkpoint_epochs`, derived
  `optimizer_steps`, total/training/validation wall-clock time,
  training/validation/total FLOPs, validation call counts, and FLOP estimate
  method.
- `supervised_depth_model_step_<N>.pt`: checkpoints from optimizer step `1`,
  the end of every `checkpoint_epochs` epochs, and the final epoch. `N` is
  still the optimizer-step number.
- `supervised_depth_model_best.pt`: best checkpoint by internal validation loss.
- `validation/<checkpoint_stem>.jsonl`: HumanEval validation generations for
  each candidate checkpoint.
- `validation_metrics.jsonl`: one `candidate` record per checkpoint plus the
  final `selected` record.
- `validation_selection.json`: validation summary, candidate results, best
  result, and promoted checkpoint path.
- `supervised_depth_model.pt`: canonical promoted checkpoint selected by
  HumanEval validation speedup.

Validation selection uses the deterministic HumanEval validation split. The
checkpoint with the highest validation speedup is promoted, with average
acceptance length and checkpoint step used as tie-breakers.

The workflow root also gets a cached validation baseline:

- `validation/baseline/humaneval.jsonl`
- `validation/baseline/humaneval.jsonl.manifest.json`

Answer JSONL files generated during validation also receive sidecar manifests
named `<answer_file>.manifest.json` with `answer_file`, `question_count`,
`draft_model_calls`, and `target_model_calls`. These call counts feed into
`time_summary.jsonl`.

The final promoted checkpoint is:

- `/results/<preset>/supervised_depth/train_t<timesteps>/supervised_depth_model.pt`

When passing checkpoint paths back to the workflow, use either absolute remote
paths or paths relative to `/results`.

## 5. Run the Benchmark Suite

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_8b \
  --depth-model-path qwen3_8b/supervised_depth/train_t20000/supervised_depth_model.pt \
  --model-label "Qwen3 8B"
```

Default benchmark datasets are `mt_bench`, `gsm8k`, `alpaca`, and `qa`.

Outputs:

- `/results/<preset>/supervised_depth/benchmark_outputs/baseline/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/eagle3/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/supervised_depth/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/summary.csv`
- sidecar manifests beside each benchmark JSONL:
  `<dataset>.jsonl.manifest.json`

The summary CSV has `model`, `method`, per-dataset speedup/tau columns, and
`mean_speedup`/`mean_tau`. Method labels are `Baseline`, `Eagle3`, and
`Eagle3+SupervisedDepth`.

The benchmark output format is intentionally aligned with the Qwen3 LTD
workflow so comparisons remain clean.

## 6. Download Result Files Locally
Download the training run:

```bash
modal volume get ltd-qwen3-results qwen3_8b/supervised_depth ./local_path
```

## 7. Remove Files on Modal
Clear the files for the next run
```bash
modal volume rm -r ltd-qwen3-results qwen3_8b/supervised_depth
```
