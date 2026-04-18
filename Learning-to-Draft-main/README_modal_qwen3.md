# Qwen3 LTD on Modal

This Modal workflow supports:

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

## Paper-faithful training

`iterative-train` now runs exactly one explicit stage per Modal execution and
persists progress in `/results/<preset>/iterative/iterative_state.json`.

First check which stage should run next:

```bash
modal run modal_qwen3.py \
  --action iterative-status \
  --model-preset qwen3_14b
```

Then run that exact stage:

```bash
export WANDB_API_KEY=your_key_here

modal run modal_qwen3.py \
  --action iterative-train \
  --model-preset qwen3_14b \
  --current-stage iter0_size \
  --use-wandb \
  --wandb-project speculative-decoding-rl \
  --size-total-timesteps 100000 \
  --depth-total-timesteps 100000
```

Defaults:

- dataset: `humaneval`
- train/validation split: deterministic 80/20 split of HumanEval
- split seed: `42`
- size-policy initial training: `100000` steps
- depth-policy initial training: `1000000` steps
- iterative schedule: `Iter0 size -> Iter0 depth -> Iter1 size -> Iter2 depth -> Iter3 size -> Iter3 depth`
- PPO: `batch_size=256`, `n_steps=2048`, `n_epochs=20`, `lr=1e-3`
- size policy: `gamma=0.9`, `pi_arch=[1024, 256]`, `vf_arch=[1024, 256]`
- depth policy: `gamma=0.999`, `pi_arch=[1024]`, `vf_arch=[1024, 256]`
- checkpoint save frequency: every `10000` steps

Validation-based checkpoint selection:

- after every stage, all saved checkpoints are evaluated on the HumanEval validation split
- the promoted checkpoint is the one with the highest validation speedup
- the next stage resumes from that promoted checkpoint
- each `modal run ... --action iterative-train` invocation requires an explicit `--current-stage`
- the requested stage must match `next_stage` from `iterative-status`

Training behavior that matches the paper:

- initial size policy trains against heuristic draft depths sampled from `[1, 12]`
- initial depth policy trains against a fixed verification size of `60`
- co-adaptation stages freeze the partner policy and retrain the current policy

One paper ambiguity exists: Section 4 says the first co-adaptation update optimizes depth first, but the experiment section labels `Iter1` as size, `Iter2` as depth, and `Iter3` as size. This implementation follows the explicit `Iter1/Iter2/Iter3` schedule from the experiment section because that is the sequence tied to the reported results.

## Override hyperparameters

The staged Modal command accepts the main paper hyperparameters as flags:

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
  --split-seed 42 \
  --use-wandb \
  --wandb-project speculative-decoding-rl
```

## W&B logging

Training runs log through the RL subprocesses:

- SB3/TensorBoard metrics such as `train/loss`, `train/policy_gradient_loss`, `rollout/ep_rew_mean`, `rollout/ep_len_mean`, and timing statistics
- custom environment metrics such as chosen token budget and rewards

Validation selection logs are written as separate W&B runs:

- per-checkpoint validation speedup
- per-checkpoint validation acceptance length
- for standalone size-policy selection, per-depth sweep metrics across heuristic depths `1..12`
- final selected checkpoint step for each stage

## Outputs

For `model_preset=qwen3_14b`, `iterative-train` writes:

- `/results/qwen3_14b/iterative/split_data/humaneval_train/question.jsonl`
- `/results/qwen3_14b/iterative/split_data/humaneval_val/question.jsonl`
- `/results/qwen3_14b/iterative/iter0_size`
- `/results/qwen3_14b/iterative/iter0_depth`
- `/results/qwen3_14b/iterative/iter1_size`
- `/results/qwen3_14b/iterative/iter2_depth`
- `/results/qwen3_14b/iterative/iter3_size`
- `/results/qwen3_14b/iterative/iter3_depth`
- `/results/qwen3_14b/iterative/iterative_state.json`

Each stage directory contains:

- saved PPO checkpoints every `10000` steps
- `ppo_speculative_decoder_controller_best.zip` from training-reward tracking
- `validation_selection.json` with all validation scores and the promoted checkpoint
- the canonical final checkpoint path overwritten with the best validation checkpoint

Final checkpoints for LTD evaluation:

- token model: `/results/<preset>/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip`
- depth model: `/results/<preset>/iterative/iter3_depth/ppo_speculative_decoder_controller_v1_single_action.zip`

## Standalone training

Standalone actions are still available:

```bash
modal run modal_qwen3.py --action train-size --model-preset qwen3_14b
modal run modal_qwen3.py --action train-depth --model-preset qwen3_14b --token-model-path qwen3_14b/size/ppo_speculative_decoder_controller_rebuttal.zip
```

Their defaults match the paper for optimizer and architecture settings, but they do not run the full iterative validation-selection workflow. For paper reproduction, use `--action iterative-train`.

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
  --depth-model-path qwen3_14b/iterative/iter3_depth/ppo_speculative_decoder_controller_v1_single_action.zip
```

## Four-dataset benchmark and CSV

```bash
modal run modal_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_14b \
  --token-model-path qwen3_14b/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter3_depth/ppo_speculative_decoder_controller_v1_single_action.zip \
  --model-label "Qwen3 14B"
```

Outputs:

- `/results/<preset>/benchmark_outputs/baseline/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/eagle3/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/ltd/<dataset>.jsonl`
- `/results/<preset>/benchmark_outputs/summary.csv`

## Inspect Modal volumes

```bash
modal volume ls
modal volume ls ltd-qwen3-results /
modal volume get ltd-qwen3-results qwen3_14b/benchmark_outputs/summary.csv ./summary.csv
```
