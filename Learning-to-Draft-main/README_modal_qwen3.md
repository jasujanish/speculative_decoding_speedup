# Qwen3 LTD on Modal

Supported model pairs:

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

Main entrypoint: [modal_qwen3.py](./modal_qwen3.py)

## Setup

From `Learning-to-Draft-main`:

```bash
uv venv .venv/modal
source .venv/modal/bin/activate
uv pip install modal
modal setup
```

## Download model weights

```bash
modal run modal_qwen3.py::download_models --model-preset qwen3_8b
modal run modal_qwen3.py::download_models --model-preset qwen3_14b
```

Weights are cached in the `ltd-qwen3-models` Modal volume.

## Iterative training

The paper-style workflow runs one explicit stage per Modal execution and stores
state in `/results/<preset>/iterative/iterative_state.json`.

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
- deterministic HumanEval split: 80/20, seed `42`
- size training: `100000` steps
- depth training: `1000000` steps
- PPO: `batch_size=256`, `n_steps=2048`, `n_epochs=20`, `lr=1e-3`
- size policy: `gamma=0.9`, `pi_arch=[1024, 256]`, `vf_arch=[1024, 256]`
- depth policy: `gamma=0.999`, `pi_arch=[1024]`, `vf_arch=[1024, 256]`
- checkpoint frequency: every `10000` timesteps

## Stage outputs

Each stage directory stores local files only:

- `training_metrics.jsonl`: PPO step, rollout, checkpoint, and best-model events
- `training_summary.json`: latest checkpoint step, best training-reward step, canonical final checkpoint, metrics log path
- `validation_metrics.jsonl`: one record per validation candidate plus the selected checkpoint
- `validation_selection.json`: final validation summary and promoted checkpoint
- `ppo_speculative_decoder_controller_step_<N>.zip`: periodic checkpoints
- `ppo_speculative_decoder_controller_best.zip`: best checkpoint by training reward
- canonical promoted checkpoint:
  - size stage: `ppo_speculative_decoder_controller_rebuttal.zip`
  - depth stage: `ppo_speculative_decoder_controller_v1_single_action.zip`

Validation selection is always based on the HumanEval validation split. The
checkpoint with the highest validation speedup is promoted and used by the next
stage.

## Final LTD checkpoints

After `iter3_size` and `iter4_depth` complete:

- token model: `/results/<preset>/iterative/iter3_size/ppo_speculative_decoder_controller_rebuttal.zip`
- depth model: `/results/<preset>/iterative/iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip`

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

Run the same benchmark after only `iter0_size` and `iter0_depth` have
completed:

```bash
modal run modal_qwen3.py \
  --action benchmark-suite \
  --model-preset qwen3_14b \
  --token-model-path qwen3_14b/iterative/iter0_size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_14b/iterative/iter0_depth/ppo_speculative_decoder_controller_v1_single_action.zip \
  --output-subdir qwen3_14b/benchmark_outputs_iter0 \
  --model-label "Qwen3 14B iter0"
```

This writes to `/results/<preset>/benchmark_outputs_iter0` instead of
overwriting the final benchmark output folder.

## Download result files

`modal volume get` downloads directories recursively when the remote path is a
folder.

Inspect first:

```bash
modal volume ls ltd-qwen3-results /
modal volume ls ltd-qwen3-results qwen3_14b
modal volume ls ltd-qwen3-results qwen3_14b/iterative
```

Download one stage:

```bash
modal volume get ltd-qwen3-results qwen3_14b/iterative/iter0_size ./iter0_size
```

Download the full iterative training tree:

```bash
modal volume get ltd-qwen3-results qwen3_14b/iterative ./qwen3_14b_iterative
```

Download all benchmark outputs:

```bash
modal volume get ltd-qwen3-results qwen3_14b/benchmark_outputs ./qwen3_14b_benchmark_outputs
```

Download every result file for a preset:

```bash
modal volume get ltd-qwen3-results qwen3_14b ./qwen3_14b_results
```

Download a single file:

```bash
modal volume get ltd-qwen3-results qwen3_14b/iterative/iter0_size/validation_selection.json ./validation_selection.json
```

Official Modal CLI reference:

- https://modal.com/docs/reference/cli/volume

## Standalone training

```bash
modal run modal_qwen3.py --action train-size --model-preset qwen3_14b
modal run modal_qwen3.py --action train-depth --model-preset qwen3_14b --token-model-path qwen3_14b/size/ppo_speculative_decoder_controller_rebuttal.zip
```

These write the same local logs and checkpoints, but they do not run the full
iterative validation-selection workflow.

## Standalone Evaluation

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
