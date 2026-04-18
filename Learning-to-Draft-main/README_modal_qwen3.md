# Qwen3 LTD on Modal

This workflow supports:

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

The main entrypoint is [modal_qwen3.py](./modal_qwen3.py).

## Setup

From `Learning-to-Draft-main`:

```bash
uv venv .venv/modal
source .venv/modal/bin/activate
uv pip install modal
modal setup
```

## Download models

```bash
modal run modal_qwen3.py::download_models --model-preset qwen3_8b
modal run modal_qwen3.py::download_models --model-preset qwen3_14b
```

This caches the base and Eagle3 weights in the `ltd-qwen3-models` Modal volume.

## Iterative training

The paper-faithful workflow runs one explicit stage per Modal execution and
persists progress in `/results/<preset>/iterative/iterative_state.json`.

Check the next required stage:

```bash
modal run modal_qwen3.py \
  --action iterative-status \
  --model-preset qwen3_14b
```

Run that exact stage:

```bash
modal run modal_qwen3.py \
  --action iterative-train \
  --model-preset qwen3_14b \
  --current-stage iter0_size \
  --size-total-timesteps 100000 \
  --depth-total-timesteps 1000000 \
  --batch-size 256 \
  --n-steps 2048 \
  --lr 0.001 \
  --validation-fraction 0.2 \
  --split-seed 42
```

Stage order:

- `iter0_size`
- `iter0_depth`
- `iter1_size`
- `iter2_depth`
- `iter3_size`
- `iter4_depth`

Defaults:

- dataset: `humaneval`
- train/validation split: deterministic 80/20 split of HumanEval
- split seed: `42`
- size-policy initial training: `100000` steps
- depth-policy initial training: `1000000` steps
- PPO: `batch_size=256`, `n_steps=2048`, `n_epochs=20`, `lr=1e-3`
- size policy: `gamma=0.9`, `pi_arch=[1024, 256]`, `vf_arch=[1024, 256]`
- depth policy: `gamma=0.999`, `pi_arch=[1024]`, `vf_arch=[1024, 256]`
- checkpoint frequency: every `10000` timesteps

## Local logging and checkpoints

Each stage directory now stores local files only. No W&B integration is used.

For a stage like `/results/qwen3_14b/iterative/iter0_size`, expect:

- `training_metrics.jsonl`: step, rollout, checkpoint, and best-model events from PPO
- `training_summary.json`: latest saved step, best training-reward step, canonical final checkpoint path, metrics log path
- `validation_metrics.jsonl`: one record per validation candidate checkpoint plus the selected result
- `validation_selection.json`: all validation scores and the promoted checkpoint
- `ppo_speculative_decoder_controller_step_<N>.zip`: periodic saved checkpoints
- `ppo_speculative_decoder_controller_best.zip`: best checkpoint by training reward
- canonical promoted checkpoint:
  - size stage: `ppo_speculative_decoder_controller_rebuttal.zip`
  - depth stage: `ppo_speculative_decoder_controller_v1_single_action.zip`

Validation-based checkpoint selection:

- after each stage, every saved checkpoint is evaluated on the HumanEval validation split
- the promoted checkpoint is the one with the highest validation speedup
- the next stage resumes from that promoted checkpoint

## Final LTD checkpoints

After `iter3_size` and `iter4_depth` complete, use:

- token model: `/results/<preset>/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip`
- depth model: `/results/<preset>/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip`

## Standalone training

Standalone actions still work:

```bash
modal run modal_qwen3.py --action train-size --model-preset qwen3_14b
modal run modal_qwen3.py --action train-depth --model-preset qwen3_14b --token-model-path qwen3_14b/size/ppo_speculative_decoder_controller_rebuttal.zip
```

These write the same local training files and checkpoints, but they do not run
the iterative validation-selection workflow.

## Evaluation

Baseline:

```bash
modal run modal_qwen3.py::evaluate_baseline --model-preset qwen3_14b --bench-name gsm8k
```

Eagle3:

```bash
modal run modal_qwen3.py::evaluate_eagle3 --model-preset qwen3_14b --bench-name gsm8k
```

LTD:

```bash
modal run modal_qwen3.py::evaluate_ltd \
  --model-preset qwen3_14b \
  --bench-name gsm8k \
  --token-model-path qwen3_14b/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip
```

## Four-dataset benchmark and CSV

```bash
modal run modal_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_14b \
  --token-model-path qwen3_14b/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip \
  --model-label "Qwen3 14B"
```

Outputs:

- `/results/<preset>/benchmark_outputs/baseline/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/eagle3/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/ltd/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/summary.csv`

## Inspect Modal volumes

```bash
modal volume ls ltd-qwen3-results /
modal volume ls ltd-qwen3-results qwen3_14b/iterative
modal volume get ltd-qwen3-results qwen3_14b/iterative/iter0_size/training_summary.json ./training_summary.json
```
