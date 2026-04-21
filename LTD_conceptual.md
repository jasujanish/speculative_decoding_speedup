# Learning to Draft Conceptual Guide

Learning to Draft (LTD) makes Eagle3-style speculative decoding adaptive. The
base model is still the verifier, and the Eagle3 model is still the drafter.
LTD adds two small PPO policies that decide how much draft work to do before
each verifier call:

- the size policy chooses how many draft-tree nodes to keep for target-model
  verification;
- the depth policy chooses whether to keep expanding the draft tree or stop and
  verify.

The objective is throughput per draft-and-verify cycle. In the implementation,
the reward is approximately:

```text
accepted_tokens / (100 * wall_time_seconds)
```

The factor `100` just rescales the reward. It does not change which action is
best.

## 1. Size Policy

The size policy is implemented in `Learning-to-Draft-main/rl/rl_total.py`. It is
called the token policy in some paths because its action controls the number of
draft tokens or tree nodes selected for verification.

### What it controls

Eagle3 grows a draft tree and assigns scores to candidate draft tokens. The size
policy decides the final node budget for the verifier. A larger budget can
increase the accepted-token count, but it can also waste verifier work on low
quality branches. A smaller budget is cheaper, but may leave easy tokens on the
table.

### Input

The size-policy observation has dimension `1268`.

It is built as:

- `1210` cumulative draft-token scores from the current Eagle3 draft tree.
- `29` repeated copies of the normalized current sequence length:
  `current_input_length / 1000`.
- `29` repeated copies of the normalized current draft depth:
  `cnet_step / 10`.

The code computes a hidden state in the observation function, but the current
Qwen3 path does not write that hidden state into the final 1268-dimensional
observation. The effective input is scores plus position/depth features.

### Output

The action space is `Discrete(24)`.

Action `a` maps to:

```text
total_token_budget = (a + 1) * 10
```

So the possible budgets are `10, 20, ..., 240`. The implementation then clips
the requested budget to the number of draft tokens actually available in the
current tree.

### PPO network

The current Qwen3 Modal workflow uses a Stable-Baselines3 `MlpPolicy` with
separate actor and critic MLPs:

- actor hidden sizes: `[1024, 256]`
- critic hidden sizes: `[1024, 256]`
- actor output: `24` logits
- critic output: `1` scalar value

Approximate trainable policy parameters:

```text
actor:  1,568,024
critic: 1,562,113
total:  3,130,137
```

This count is only for the PPO policy network. It does not include the Qwen
target model or Eagle3 drafter weights.

### Training behavior

In a size-policy environment step:

1. The drafter first expands a tree to some depth. In the initial `iter0_size`,
   the depth is random from `1` to `12`. In later size stages, a fixed depth
   policy can decide where to stop.
2. The size policy chooses the verification budget.
3. The top scored draft nodes are converted into a tree attention mask and
   verified by the target model in one forward pass.
4. The reward is based on accepted tokens divided by elapsed draft-plus-verify
   time.

The size policy is saved as:

```text
ppo_speculative_decoder_controller_rebuttal.zip
```

## 2. Depth Policy

The depth policy is implemented in `Learning-to-Draft-main/rl/rl_depth.py`.
It decides whether the current draft tree should keep growing before the
verifier is called.

### What it controls

Depth controls draft investment. More depth can expose long correct branches in
easy contexts. In hard contexts, deeper drafting often creates tokens that the
target model rejects, so stopping earlier is better.

### Input

The depth-policy observation has dimension `128`.

It is built as:

- `100` cumulative scores for the current `top_k x top_k` Eagle3 candidate
  expansion. The default `top_k` is `10`, so this is `10 * 10 = 100` scores.
- `14` repeated copies of the normalized current sequence length:
  `current_input_length / 1000`.
- `14` repeated copies of the normalized current draft depth:
  `cnet_step / 10`.

### Output

The action space is `Discrete(2)`.

- `0`: stop expanding and verify.
- `1`: continue expanding one more draft layer.

There is one forced expansion when `cnet_step == 0`, so the tree has at least
one draft layer before verification. The runtime maximum draft depth is `12`.

### PPO network

The current Qwen3 Modal workflow uses a Stable-Baselines3 `MlpPolicy` with
separate actor and critic MLPs:

- actor hidden sizes: `[1024]`
- critic hidden sizes: `[1024, 256]`
- actor output: `2` logits
- critic output: `1` scalar value

Approximate trainable policy parameters:

```text
actor:    134,146
critic:   394,753
total:    528,899
```

This count is only for the PPO policy network.

### Training behavior

In a depth-policy environment step:

1. If the action is `1`, the Eagle3 drafter expands one layer and the step
   returns reward `0`.
2. If the action is `0`, or if the tree hits the runtime depth limit, the tree
   is finalized and verified.
3. The final size is either fixed at `60` tokens or chosen by the fixed size
   policy when a size-policy checkpoint is provided.
4. The reward after verification is accepted tokens divided by elapsed
   draft-plus-verify time.

The depth policy is saved as:

```text
ppo_speculative_decoder_controller_v1_single_action.zip
```

## 3. Hyperparameters and Default Values

These are the effective defaults used by the Qwen3 Modal workflow in
`Learning-to-Draft-main/modal_qwen3.py` and the PPO scripts in
`Learning-to-Draft-main/rl/`. The standalone RL scripts expose additional CLI
flags, but the values below are the defaults used by the Modal workflow unless a
run overrides them.

### Model and data

- default model preset: `qwen3_8b`
- supported presets:
  - `qwen3_8b`: `Qwen/Qwen3-8B` with `AngelSlim/Qwen3-8B_eagle3`
  - `qwen3_14b`: `Qwen/Qwen3-14B` with `AngelSlim/Qwen3-14B_eagle3`
- training dataset: `humaneval`
- train/validation split: deterministic 80/20 HumanEval split
- HumanEval validation fraction: `0.2`
- split seed: `42`
- maximum prompt length admitted during training: `1748` tokens
- maximum generated tokens per prompt during training: `256`

### Drafting and evaluation

- Eagle3 `top_k`: `10`
- default verification budget: `60`
- default benchmark static depth: `8`
- training maximum draft depth: `12`
- validation benchmark: HumanEval validation split
- validation selection metric: highest speedup over baseline
- validation tie-breakers: higher `tau`, then higher checkpoint step
- benchmark datasets: `mt_bench`, `gsm8k`, `alpaca`, `qa`

### PPO settings

Shared defaults:

- algorithm: Stable-Baselines3 PPO
- policy type: `MlpPolicy`
- rollout length: `n_steps=2048`
- minibatch size: `batch_size=256`
- optimization epochs per rollout: `n_epochs=20`
- initial learning rate: `lr=1e-3`
- learning-rate schedule: 1% warmup, then linear decay
- target KL limit: `None`
- entropy coefficient: `ent_coef=0.01`
- checkpoint/log frequency: every `10000` environment timesteps
- PPO device for the policy network: CPU

Size-policy defaults:

- PPO timesteps per size stage: `100000`
- discount factor: `gamma=0.9`
- actor architecture: `[1024, 256]`
- critic architecture: `[1024, 256]`
- observation dimension: `1268`
- action count: `24`

Depth-policy defaults:

- PPO timesteps per depth stage: `1000000`
- discount factor: `gamma=0.999`
- actor architecture: `[1024]`
- critic architecture: `[1024, 256]`
- observation dimension: `128`
- action count: `2`

## 4. Training Process

LTD trains the two policies with PPO and then co-optimizes them by alternating
which policy is trainable and which policy is fixed.

### PPO environment

Each PPO environment step is one local drafting decision or one full
draft-and-verify cycle:

- The Eagle3 drafter proposes a draft tree.
- LTD chooses either the tree depth or the final node budget.
- The Qwen target model verifies the selected tree with tree attention.
- The accepted prefix length gives the benefit.
- The elapsed draft-plus-verify time gives the cost.
- PPO receives the throughput-like reward.

The target model and Eagle3 drafter are not updated. PPO updates only the small
size/depth controller.

### Checkpointing and promotion

During every stage, the training script writes:

- periodic PPO checkpoints;
- a best checkpoint by training reward;
- `training_metrics.jsonl`;
- `training_summary.json`;
- `dataset_manifest.json`.

The Modal workflow then evaluates candidate checkpoints on the HumanEval
validation split and promotes the one with the highest validation speedup.
The promoted checkpoint becomes the fixed partner or resume checkpoint for
later stages.

### Co-optimization schedule

The current iterative schedule is:

1. `iter0_size`
   - Train size from scratch.
   - No learned depth policy is fixed yet.
   - During training, draft depth is random from `1` to `12`.
   - During validation, each size checkpoint is evaluated across depths `1..12`
     and the average validation speedup is used.
2. `iter0_depth`
   - Train depth from scratch.
   - No learned size policy is fixed yet.
   - Verification size defaults to `60`.
3. `iter1_size`
   - Resume size from `iter0_size`.
   - Fix depth to the promoted `iter0_depth` checkpoint.
4. `iter2_depth`
   - Resume depth from `iter0_depth`.
   - Fix size to the promoted `iter1_size` checkpoint.
5. `iter3_size`
   - Resume size from `iter1_size`.
   - Fix depth to the promoted `iter2_depth` checkpoint.
6. `iter4_depth`
   - Resume depth from `iter2_depth`.
   - Fix size to the promoted `iter3_size` checkpoint.

The final LTD pair is:

- size model: `iter3_size/ppo_speculative_decoder_controller_rebuttal.zip`
- depth model:
  `iter4_depth/ppo_speculative_decoder_controller_v1_single_action.zip`

### Why co-optimization is needed

The two policies affect each other. A depth policy that grows deeper changes the
set of candidate nodes that the size policy can select. A size policy that keeps
more or fewer nodes changes whether extra depth is useful. Training one policy
against a stale or random partner can produce a controller that is locally good
but mismatched to the final inference system.

The alternating schedule lets each policy adapt to the current behavior of the
other policy while preserving stability: only one controller is updated in a
stage, and the other controller is treated as fixed.

### Inference after training

At inference time, LTD loads both promoted policies:

- the depth policy dynamically decides when to stop expanding the Eagle3 draft
  tree;
- the size policy dynamically chooses how many high-scoring draft nodes to send
  to the verifier;
- the target model still verifies the selected tree and determines which tokens
  are actually accepted.

The generated output remains governed by the target-model verification step.
LTD is an efficiency controller over draft-tree construction, not a replacement
for target-model verification.
