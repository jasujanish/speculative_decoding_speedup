# Supervised Depth Conceptual Guide

This document explains the supervised-learning replacement for the LTD depth
policy. The code lives in `Learning-to-Draft-main/supervised_depth_modal/`.

The core idea is to avoid reinforcement learning for the depth decision. Instead
of training a PPO policy from rollout rewards, we collect real Eagle3 draft-tree
states and label each state by directly comparing two choices:

```text
stop and verify now
continue exactly one more draft depth, then verify
```

The supervised model learns the throughput difference between those two choices.
At inference time, it is used as a stop rule while Eagle3 grows the draft tree.

## 1. Policy Description

The policy is implemented by `SupervisedDepthModel` in
`supervised_depth_modal/core.py`.

It is not a language model. It does not replace Qwen or Eagle3. It is a small
MLP controller that decides whether Eagle3 should continue expanding the current
draft tree.

### Input Tensor

The model input is a float tensor with shape:

```text
[batch_size, 129]
```

For one depth decision, the 129 features are:

- `0:100`: up to 100 cumulative Eagle3 draft scores from the current frontier.
  With the default `top_k=10`, this corresponds to a flattened `10 x 10`
  candidate score grid from the next possible expansion.
- `100:114`: 14 repeated copies of normalized context length:
  `current_context_length / 1000`.
- `114:128`: 14 repeated copies of normalized draft depth:
  `current_draft_depth / 10`.
- `128`: entropy of the current score frontier.

The MLP does not directly receive Qwen hidden states or Eagle3 hidden states.
Those tensors are used internally by Eagle3 to generate candidate tokens and
frontier scores. The supervised policy only receives the compact 129-dimensional
numeric state derived from the draft frontier.

### Network

The architecture is:

```text
Linear(129, 1024)
ReLU()
Linear(1024, 1)
```

The only hidden layer has width `1024`.

### Output

The model outputs one scalar:

```text
predicted_delta = predicted throughput(next depth) - throughput(stop now)
```

The inference rule is:

- if `predicted_delta > 0`, continue expanding the draft tree;
- if `predicted_delta <= 0`, stop expanding and verify with the target model.

This is trained as regression, not softmax classification. The dataset also
stores `continue_label = int(target_delta > 0)` for interpretability and sign
accuracy, but the training loss is on the real-valued `target_delta`.

### Parameter Count

With the default hidden width of `1024`, the supervised controller has:

```text
Linear(129, 1024): 129 * 1024 + 1024 = 133,120
Linear(1024, 1):   1024 * 1 + 1      =   1,025
Total:                                134,145 parameters
```

This count is only for the supervised depth controller. It does not include the
Qwen target model or the AngelSlim Eagle3 drafter.

## 2. Dataset Collection

Dataset collection is implemented by `DepthStateCollector` and
`collect_supervised_depth_dataset` in `supervised_depth_modal/core.py`.

The collector runs the real target model and the real Eagle3 drafter on
HumanEval training prompts. It records depth-decision states from actual draft
tree construction, not synthetic states.

### Collection Loop

For each supervised example:

1. Start or continue generation from a HumanEval prompt.
2. Build the current 129-dimensional observation.
3. Verify the current draft tree with `apply_update=False`. This clones the KV
   cache state and estimates the throughput if the system stops now.
4. If the current depth is below `max_draft_depth`, expand the draft tree by one
   Eagle3 layer.
5. Verify the expanded tree with `apply_update=False`. This estimates the
   throughput if the system continues one more depth and then stops.
6. Store the supervised regression target:

```text
target_delta = next_throughput - stop_throughput
```

7. Store the helper label:

```text
continue_label = 1 if target_delta > 0 else 0
```

The collection budget `total_timesteps` means number of collected depth-decision
states. For example, `--total-timesteps 10000` collects 10,000 supervised
examples. It is not the number of optimizer steps.

### Throughput Label

Throughput is computed from the actual target verification result:

```text
throughput = accepted_tokens / (100 * elapsed_time_seconds)
```

The factor `100` is a scaling convention inherited from the LTD reward code. It
does not change the sign of `target_delta`, which is what controls the
continue-versus-stop decision.

### Generated Dataset Files

The collector writes:

- `dataset.jsonl`: one JSON record per depth-decision state.
- `dataset_manifest.json`: collection metadata and FLOP accounting.

Each `dataset.jsonl` record contains:

- `observation`: the 129-dimensional policy input.
- `target_delta`: supervised regression target.
- `continue_label`: binary sign of `target_delta`.
- `draft_depth`: current draft depth before the possible expansion.
- `context_length`: current prefix length.
- `entropy`: entropy of the score frontier.
- `stop_throughput`: throughput if stopping at this state.
- `next_throughput`: throughput after one more expansion.
- `stop_accept_length`: accepted length when stopping at this state.
- `next_accept_length`: accepted length after one more expansion.

The manifest records:

- `total_timesteps` and `collected_examples`;
- `episodes_started`;
- `draft_model_calls` and `target_model_calls`;
- fixed verification size, currently `fixed_total_token = 60`;
- maximum explored draft depth, currently `max_draft_depth = 12`;
- `observation_dim = 129`;
- model paths;
- target and drafter parameter counts;
- estimated collection FLOPs using `2 * parameter_count * forward_calls`.

The fixed verification size is important: this supervised method currently
learns only the depth stop rule. It does not learn a separate LTD-style size
policy.

## 3. Training

Training is implemented by `train_supervised_depth_model`.

### Objective

The supervised model is trained to regress `target_delta`:

```text
model(observation) -> predicted target_delta
```

The loss is:

```text
SmoothL1Loss(predicted_delta, target_delta)
```

The optimizer is AdamW.

### Train/Validation Split

The collected `dataset.jsonl` is split internally into:

- 90% training examples;
- 10% validation examples.

The default split seed is `42`. This split is for regression monitoring. It is
separate from the HumanEval train/validation prompt split used by the Modal
workflow for end-to-end checkpoint selection.

### Epochs

Training is epoch-based:

```text
--epochs <N>
```

One epoch means one full pass over the supervised training split. The optimizer
step count is derived from the dataset size and batch size:

```text
optimizer_steps = epochs * train_batches_per_epoch
```

The default batch size is `256`, and the default learning rate is `0.001`.

### Checkpoints and Metrics

Checkpointing is controlled in epoch units:

```text
--checkpoint-epochs <X>
```

The trainer writes:

- an initial checkpoint at optimizer step `1`;
- a checkpoint at the end of every `X` epochs;
- a final checkpoint at the last epoch;
- `supervised_depth_model_best.pt`, selected by internal validation loss;
- `supervised_depth_model.pt`, initially copied from the best internal
  validation checkpoint.

Checkpoint filenames still use optimizer-step numbers:

```text
supervised_depth_model_step_<optimizer_step>.pt
```

Training also writes:

- `training_metrics.jsonl`: train loss, validation loss, sign accuracy, and
  checkpoint events.
- `training_summary.json`: dataset path, epochs, optimizer steps, batch size,
  learning rate, model parameter count, examples seen, and estimated training
  FLOPs.

### End-to-End Promotion

After supervised regression training, the Modal workflow evaluates candidate
checkpoints on the deterministic HumanEval validation split. It measures
end-to-end decoding speedup and average acceptance length `tau`.

The promoted checkpoint is selected by:

1. highest validation speedup;
2. then higher `tau`;
3. then higher checkpoint step.

The selected checkpoint is copied to:

```text
supervised_depth_model.pt
```

This second validation stage matters because the checkpoint with the lowest
regression loss is not necessarily the checkpoint with the best decoding
throughput.

## 4. Hyperparameters and Default Values

These are the current defaults in
`Learning-to-Draft-main/supervised_depth_modal/core.py` and
`Learning-to-Draft-main/supervised_depth_modal/modal_supervised_depth_qwen3.py`.

### Model and Policy

- default model preset: `qwen3_8b`
- supported presets: `qwen3_8b`, `qwen3_14b`
- policy input dimension: `129`
- maximum score features: `100`
- hidden width: `1024`
- output dimension: `1`
- trainable supervised-policy parameters: `134145`

### Data Collection

- collection budget: `total_timesteps=20000`
- HumanEval validation fraction for prompt split: `0.2`
- HumanEval split seed: `42`
- fixed verification size: `total_token=60`
- Eagle3 `top_k`: `10`
- maximum explored draft depth: `max_draft_depth=12`
- maximum prompt length admitted during collection: `1748` tokens
- maximum generated tokens per prompt: `256`
- collection dtype for Qwen/Eagle loading: `torch.float16`
- Eagle3 dynamic length and dynamic token flags during collection:
  `use_dyn_len=False`, `use_dyn_token=False`

### Supervised Training

- epochs: `1`
- checkpoint interval: `checkpoint_epochs=1`
- batch size: `256`
- learning rate: `0.001`
- optimizer: AdamW
- loss: SmoothL1Loss
- internal regression validation fraction: `0.1`
- internal regression split seed: `42`
- device: CUDA when available, otherwise CPU
- training FLOP estimate: `6 * supervised_model_parameters * examples_seen`

The Modal prompt split and the internal regression split are different. The
Modal split decides which HumanEval prompts are used for collection versus
end-to-end validation. The internal regression split divides the collected
`dataset.jsonl` examples into train and validation minibatches.

### Validation and Benchmarking

- validation benchmark: HumanEval validation split
- validation baseline: target-model baseline cached under the workflow root
- validation generation temperature: `0.0`
- validation choices per question: `1`
- validation verification size: `60`
- validation static depth argument: `8`
- benchmark datasets: `mt_bench`, `gsm8k`, `alpaca`, `qa`
- benchmark question range: `question_begin=0`, `question_end=80`
- checkpoint selection metric: highest validation speedup
- checkpoint selection tie-breakers: higher `tau`, then higher checkpoint step

## 5. Advantages Over LTD

The supervised-depth method is intended to make the depth-policy training
problem simpler and more inspectable than LTD's PPO setup.

### Direct Labels Instead of PPO Rewards

LTD trains policies with reinforcement learning signals from draft-and-verify
rollouts. The supervised method labels each state with the direct local
comparison:

```text
throughput(one more depth) - throughput(stop now)
```

That makes the target easier to understand, debug, and plot. A positive label
means continuing was better for that state; a negative label means stopping was
better.

### No On-Policy PPO Instability

The supervised model is trained with ordinary minibatch regression. It avoids
PPO-specific issues such as rollout variance, reward normalization sensitivity,
policy clipping behavior, and coupling between data collection and policy
updates.

### Reusable Dataset

Once `dataset.jsonl` is collected, different supervised models or
hyperparameters can be trained against the same examples. LTD-style PPO training
usually needs fresh on-policy interaction as the policy changes.

### Smaller Policy Surface

This method learns only the depth stop rule. The verification size is fixed at
`total_token = 60`. Removing the size policy makes the system easier to reason
about and makes attribution clearer: speed changes mostly come from when the
draft tree stops growing, not from simultaneous depth and size co-optimization.

### Transparent Accounting

The supervised workflow records data-collection calls, target-model calls,
drafter calls, training examples seen, validation calls, and FLOP estimates in
manifest and summary files. This makes it easier to compare training cost
against LTD runs.

### Easier Debugging

The records expose `target_delta`, `continue_label`, score entropy, draft depth,
context length, and both stop/next throughput estimates. This makes failure
modes concrete. For example, if sign accuracy is poor at deep draft depths, the
dataset can be filtered or inspected directly.

### Tradeoffs

The supervised method is not a complete replacement for every LTD capability.
It uses an expensive label-collection step because each example compares stop
versus one-more-depth verification. It also learns a local one-step depth
decision, while LTD can in principle learn a broader policy over depth and size.
The benefit is that the supervised version is much simpler, smaller, and easier
to measure. While this was not implemented, in theory, if you plan to work with
a fixed max depth, 
```text
predicted_delta = predicted max throughput of all future depths - throughput(stop now)
```