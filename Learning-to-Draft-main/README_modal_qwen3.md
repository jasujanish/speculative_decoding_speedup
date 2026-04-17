# Qwen3 + AngelSlim Eagle3 on Modal

This guide explains how to run Learning to Draft (LTD) on Modal using only these two supported model pairs:

- `qwen3_8b` -> `Qwen/Qwen3-8B` + `AngelSlim/Qwen3-8B_eagle3`
- `qwen3_14b` -> `Qwen/Qwen3-14B` + `AngelSlim/Qwen3-14B_eagle3`

The Modal entrypoint for this workflow is [modal_qwen3.py](./modal_qwen3.py).

## 1. Prepare your local environment

From the `Learning-to-Draft-main` directory:

```bash
uv venv .venv/modal
source .venv/modal/bin/activate
uv pip install modal
modal setup
```

The Qwen3 and AngelSlim checkpoints used here are public, so this workflow does
not require Modal secrets just to download or run the models.

Do not export `MODAL_HF_SECRET_NAME` or `MODAL_WANDB_SECRET_NAME` for this
version of the script. The current Modal app is intentionally deterministic and
does not auto-discover secrets from local environment variables.

## 2. Download the model weights to Modal

Choose one preset and download its base model plus Eagle3 draft model into the `ltd-qwen3-models` Volume:

```bash
modal run modal_qwen3.py::download_models --model-preset qwen3_8b
```

or

```bash
modal run modal_qwen3.py::download_models --model-preset qwen3_14b
```

The downloaded paths inside Modal will be:

- `/models/Qwen/Qwen3-8B`
- `/models/AngelSlim/Qwen3-8B_eagle3`

or

- `/models/Qwen/Qwen3-14B`
- `/models/AngelSlim/Qwen3-14B_eagle3`

## 3. Train the LTD size policy

This runs `python -m rl.rl_total` on Modal and stores outputs in the `ltd-qwen3-results` Volume.

Example for Qwen3 8B:

```bash
modal run modal_qwen3.py::train_size_policy \
  --model-preset qwen3_8b \
  --dataset-train humaneval \
  --total-timesteps 100000
```

Example for Qwen3 14B:

```bash
modal run modal_qwen3.py::train_size_policy \
  --model-preset qwen3_14b \
  --dataset-train humaneval \
  --total-timesteps 100000
```

Default output directory:

- `qwen3_8b` -> `/results/qwen3_8b/size`
- `qwen3_14b` -> `/results/qwen3_14b/size`

The main saved size-policy checkpoint will be:

- `/results/qwen3_8b/size/ppo_speculative_decoder_controller_rebuttal.zip`

or

- `/results/qwen3_14b/size/ppo_speculative_decoder_controller_rebuttal.zip`

## 4. Train the LTD depth policy

Depth training expects the size-policy checkpoint path. Pass it relative to `/results`.

Example for Qwen3 8B:

```bash
modal run modal_qwen3.py::train_depth_policy \
  --model-preset qwen3_8b \
  --dataset-train humaneval \
  --total-timesteps 100000 \
  --rl-token-model-path qwen3_8b/size/ppo_speculative_decoder_controller_rebuttal.zip
```

Example for Qwen3 14B:

```bash
modal run modal_qwen3.py::train_depth_policy \
  --model-preset qwen3_14b \
  --dataset-train humaneval \
  --total-timesteps 100000 \
  --rl-token-model-path qwen3_14b/size/ppo_speculative_decoder_controller_rebuttal.zip
```

Default output directory:

- `qwen3_8b` -> `/results/qwen3_8b/depth`
- `qwen3_14b` -> `/results/qwen3_14b/depth`

The main saved depth-policy checkpoint will be:

- `/results/qwen3_8b/depth/ppo_speculative_decoder_controller_v1_single_action.zip`

or

- `/results/qwen3_14b/depth/ppo_speculative_decoder_controller_v1_single_action.zip`

## 5. Run the three evaluation modes

### Baseline Qwen3

```bash
modal run modal_qwen3.py::evaluate_baseline \
  --model-preset qwen3_8b \
  --bench-name gsm8k
```

### Eagle3

```bash
modal run modal_qwen3.py::evaluate_eagle3 \
  --model-preset qwen3_8b \
  --bench-name gsm8k
```

### LTD

```bash
modal run modal_qwen3.py::evaluate_ltd \
  --model-preset qwen3_8b \
  --bench-name gsm8k \
  --token-model-path qwen3_8b/size/ppo_speculative_decoder_controller_rebuttal.zip \
  --depth-model-path qwen3_8b/depth/ppo_speculative_decoder_controller_v1_single_action.zip
```

Replace `qwen3_8b` with `qwen3_14b` to evaluate the 14B preset.

The default JSONL output locations are:

- baseline -> `/results/<preset>/eval/baseline/<bench>.jsonl`
- Eagle3 -> `/results/<preset>/eval/eagle3/<bench>.jsonl`
- LTD -> `/results/<preset>/eval/ltd/<bench>.jsonl`

## 6. Inspect outputs in Modal

Your persistent Modal Volumes are:

- `ltd-qwen3-models`
- `ltd-qwen3-results`

Useful commands:

```bash
modal volume ls
modal volume ls ltd-qwen3-results /
modal volume get ltd-qwen3-results qwen3_8b/eval/ltd/gsm8k.jsonl ./gsm8k_ltd.jsonl
```

## 7. Local shell wrappers

If you want to run the modified LTD scripts outside Modal, the repo entrypoints now accept the Qwen3 presets directly:

```bash
MODEL_PRESET=qwen3_8b sh train_size.sh 0
MODEL_PRESET=qwen3_8b sh train_depth.sh 0
MODEL_PRESET=qwen3_8b sh eval.sh /path/to/depth.zip /path/to/size.zip
```

The Qwen3 evaluation scripts also accept:

- `--model-preset qwen3_8b`
- `--model-preset qwen3_14b`

## 8. What changed in this repo

- Fixed the broken Qwen3 loader in `eagle/model/ea_model.py`
- Added a shared Qwen3 preset registry in `qwen3_model_presets.py`
- Added preset support to Qwen3 evaluation and RL training scripts
- Made `wandb` logging opt-in for RL training
- Added `modal_qwen3.py` for download, training, and evaluation on Modal
