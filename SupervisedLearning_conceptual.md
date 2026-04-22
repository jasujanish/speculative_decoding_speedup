# Supervised Depth Conceptual Guide

This document explains the supervised-learning depth policies in
`Learning-to-Draft-main/supervised_depth_modal/`.

Both policies leave the target Qwen model and Eagle3 drafter frozen. The learned
model is only a small controller that decides whether draft-tree growth should
continue before the next target-model verification.

## 1. Policy Description

There are two intended policy variants: `base` and `titan`.

### Base Policy

The base policy uses the original compact Eagle3 depth observation:

```text
[batch_size, 129]
```

For one depth state, the features are:

- `0:100`: up to 100 cumulative Eagle3 draft scores.
- `100:114`: 14 repeated copies of normalized context length:
  `current_context_length / 1000`.
- `114:128`: 14 repeated copies of normalized draft depth:
  `current_draft_depth / 10`.
- `128`: entropy of the current score frontier.

The base MLP is:

```text
129 -> 1024 -> 1
```

It has 134,145 trainable parameters.

### Titan Policy

Titan uses the same first 129 features plus six extra frontier features:

- max score
- mean score
- score standard deviation
- min score
- current entropy minus previous entropy
- EOS-like log probability from the latest drafter logits

The Titan input shape is:

```text
[batch_size, 135]
```

The Titan MLP is:

```text
135 -> 1664 -> 384 -> 128 -> 1
```

It has 915,073 trainable parameters.

### Output

Both policies output one scalar:

```text
predicted_delta
```

Inference uses the same rule for both:

```text
continue if predicted_delta > 0
stop if predicted_delta <= 0
```

The policies do not directly receive Qwen hidden states or Eagle3 hidden-state
tensors. Those tensors are used internally by Eagle3 to produce draft tokens and
scores. The supervised controller sees only numeric draft-frontier features.

## 2. Dataset Collection

Dataset collection is sweep-based internally for both `base` and `titan`.

`--total-timesteps` means the minimum number of labeled depth-state records to
collect, not the number of sweeps. During each sweep, the collector starts from
the current speculative state, then repeatedly:

1. Builds the policy observation at the current depth.
2. Verifies the current draft tree with `apply_update=False`.
3. Records the throughput for stopping at that depth.
4. Expands one more Eagle3 depth, until `max_draft_depth` is reached.

This produces a depth curve:

```text
T(0), T(1), T(2), ..., T(max_depth)
```

where `T(d)` is the throughput if the system stops and verifies at depth `d`.
After the sweep is measured, the collector emits one dataset record for every
depth state in the sweep. The final sweep is kept intact, so
`collected_examples` can be slightly larger than `total_timesteps`.

This is more data-efficient than collecting one isolated local transition at a
time, because the same measured depth curve is reused to label all depths in the
sweep.

### Base Labels

The base policy keeps the greedy one-step objective:

```text
target_delta(d) = T(d + 1) - T(d)
```

The final depth has no next depth, so its target is `0` and its
`continue_label` is `0`.

### Titan Labels

Titan uses a Q-style best-future objective:

```text
Q_stop(d) = T(d)
Q_continue(d) = max T(k), for k > d
target_delta(d) = Q_continue(d) - Q_stop(d)
```

This teaches the model whether any deeper stopping point beats stopping now,
instead of asking only whether the immediately next depth is better.

### Generated Files

The collector writes:

- `dataset.jsonl`
- `dataset_manifest.json`

Important fields in each record:

- `observation`
- `target_delta`
- `continue_label`
- `target_type`
- `draft_depth`
- `context_length`
- `entropy`
- `stop_throughput`
- `next_throughput`
- `stop_accept_length`
- `next_accept_length`

Titan records also include:

- `score_max`, `score_mean`, `score_std`, `score_min`
- `entropy_delta`
- `eos_logprob_from_last_drafter_logits`
- `q_stop_throughput`
- `q_continue_throughput`
- `best_future_depth`
- `best_future_accept_length`

The manifest records:

- requested minimum labeled records as `total_timesteps`
- `collected_sweeps`
- `collected_examples`
- draft and target model call counts
- fixed verification size, currently `60`
- max explored draft depth, currently `12`
- policy variant
- observation dimension
- estimated collection FLOPs

## 3. Training

Training is ordinary supervised regression:

```text
model(observation) -> target_delta
```

The loss is:

```text
SmoothL1Loss(predicted_delta, target_delta)
```

The optimizer is AdamW.

The collected JSONL records are split internally into:

- 90% training examples
- 10% validation examples

The default split seed is `42`.

Training is epoch-based:

```text
optimizer_steps = epochs * train_batches_per_epoch
```

Checkpoints are written at step 1, at the end of every
`--checkpoint-epochs`, and at the final epoch. The Modal workflow then evaluates
candidate checkpoints on the HumanEval validation split and promotes the
checkpoint with the highest validation speedup.

## 4. Hyperparameters and Default Values

### Shared

- default model preset: `qwen3_8b`
- supported presets: `qwen3_8b`, `qwen3_14b`
- collection budget: `total_timesteps=20000` labeled depth-state records
- HumanEval validation fraction for prompt split: `0.2`
- split seed: `42`
- fixed verification size: `total_token=60`
- Eagle3 `top_k`: `10`
- maximum explored draft depth: `12`
- maximum prompt length: `1748`
- maximum generated tokens per prompt: `256`
- batch size: `256`
- learning rate: `0.001`
- default epochs: `1`
- default checkpoint interval: `1` epoch
- benchmark datasets: `mt_bench`, `gsm8k`

### Base

- policy variant: `base`
- observation dimension: `129`
- target type: `greedy_next_depth`
- hidden dimensions: `[1024]`
- parameters: `134145`

### Titan

- policy variant: `titan`
- observation dimension: `135`
- target type: `q_best_future`
- hidden dimensions: `[1664, 384, 128]`
- parameters: `915073`

## 5. Advantages Over LTD

The supervised-depth approach is simpler than LTD's PPO training:
- It uses supervised regression instead of on-policy PPO updates. This is much more sample efficient as we can reuse training points across epochs.
- We construct expert datasets. The base policy learns the from the optimal greedy policy while the titan policy learns from the optimal policy.
- The base policy is much smaller than the LTD depth policy (so it incurs less of an overhead than the LTD depth policy)
- Training does not require co-optimization, significantly reducing total training time and approximate FLOPs
- Dataset records expose the exact throughput comparisons used as labels.
- The training process is easy to repeat with the same dataset.

Titan adds a stronger label than the greedy base policy. Instead of learning
only whether the next depth is better, Titan learns whether any future depth in
the measured sweep beats stopping now. This is closer to the actual stopping
problem at inference time.
