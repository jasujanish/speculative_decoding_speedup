# Qwen3 Supervised Depth on Modal

This folder contains a separate Modal workflow for a supervised replacement of
the LTD depth policy. It does not modify the original LTD codepath.

Main entrypoint: [modal_supervised_depth_qwen3.py](./modal_supervised_depth_qwen3.py)

## What Is Different From LTD

- LTD trains the depth policy with PPO.
- This workflow replaces that depth policy with a supervised regressor.
- The model predicts a one-step throughput delta:
  `delta = throughput(next depth) - throughput(stop now)`.
- During inference, expansion continues while `delta > 0` and stops otherwise.
- The size policy is removed. Verification size stays fixed at `60`, matching the
  original depth-policy training setup.

## Fairness and `total_timesteps`

`total_timesteps` is the collection budget for supervised data. It limits the
number of depth-decision states collected from the real drafting environment.

This is the intended fairness knob for comparison against the original LTD depth
policy:

- `20k` LTD depth-policy timesteps
- `20k` supervised-depth collection timesteps

The supervised pipeline can still derive richer labels from those collected
states, so the workflow logs extra metadata such as draft-model calls and
target-model calls in the dataset manifest for transparency.

## Setup

From `Learning-to-Draft-main`:

```bash
uv venv .venv/modal_supervised_depth
source .venv/modal_supervised_depth/bin/activate
uv pip install modal
modal setup
```

## Download model weights

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action download-models \
  --model-preset qwen3_8b
```

Supported presets:

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

## 1. Collect a supervised dataset

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action collect-depth-dataset \
  --model-preset qwen3_8b \
  --total-timesteps 20000
```

Outputs:

- `/results/<preset>/supervised_depth/dataset_t<timesteps>/dataset.jsonl`
- `/results/<preset>/supervised_depth/dataset_t<timesteps>/dataset_manifest.json`

Important dataset fields:

- `observation`: depth-policy observation vector plus one entropy scalar
- `target_delta`: `throughput(next depth) - throughput(stop now)`
- `continue_label`: `1` if `target_delta > 0`, else `0`

## 2. Train the supervised depth policy

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action train-depth \
  --model-preset qwen3_8b \
  --total-timesteps 20000 \
  --optimizer-steps 20000 \
  --batch-size 256 \
  --lr 0.001 \
  --checkpoint-freq 10000
```

If `--optimizer-steps` is omitted or set to `0`, it defaults to
`total_timesteps`.

Training outputs:

- `/results/<preset>/supervised_depth/train_t<timesteps>/training_metrics.jsonl`
- `/results/<preset>/supervised_depth/train_t<timesteps>/training_summary.json`
- `supervised_depth_model_step_<N>.pt`
- `supervised_depth_model_best.pt`
- `supervised_depth_model.pt`
- validation artifacts:
  - `validation_metrics.jsonl`
  - `validation_selection.json`

The final promoted checkpoint is:

- `/results/<preset>/supervised_depth/train_t<timesteps>/supervised_depth_model.pt`

## 3. Run the benchmark suite

```bash
modal run supervised_depth_modal/modal_supervised_depth_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_8b \
  --depth-model-path qwen3_8b/supervised_depth/train_t20000/supervised_depth_model.pt \
  --model-label "Qwen3 8B"
```

Outputs:

- `/results/<preset>/supervised_depth/benchmark_outputs/baseline/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/eagle3/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/supervised_depth/<dataset>.jsonl`
- `/results/<preset>/supervised_depth/benchmark_outputs/summary.csv`

The benchmark output format is intentionally kept aligned with the original
Qwen3 LTD workflow so comparisons remain clean.

## Hyperparameters

Collection:

- `total_timesteps`: environment interaction budget
- `validation_fraction`: HumanEval validation split fraction, default `0.2`
- `split_seed`: HumanEval split seed, default `42`
- fixed verification size: `60`
- max explored draft depth during collection: `12`

Training:

- `optimizer_steps`: gradient-update budget
- `batch_size`: default `256`
- `lr`: default `1e-3`
- `checkpoint_freq`: default `10000`
- internal example-level validation fraction: `0.1`
- hidden width: `1024`

Inference:

- fixed verification size: `60`
- static draft depth limit passed to Eagle3: `8`
- supervised stop rule is applied inside tree growth through the patched
  depth callback

## Notes

- This workflow uses separate result folders and a separate README.
- It does not edit or depend on changes to the original LTD training code.
- The supervised model adds one entropy feature on top of the original depth
  observation.
