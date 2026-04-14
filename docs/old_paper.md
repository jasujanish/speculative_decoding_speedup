# **Link:** https://arxiv.org/pdf/2603.01639

# **AI GENERATED SUMMARY:**
Breakdown: Learning to Draft (LTD) for Adaptive Speculative DecodingHere is a step-by-step breakdown of the methodology presented in Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning (arXiv:2603.01639). While I am working from the core architectural summaries available online, the fundamental mechanics map cleanly onto standard reinforcement learning and transformer architectures.1. The Core Problem: The Speculative Trade-offIn standard speculative decoding, inference is accelerated by pairing a small, fast "draft" model with a large "target" model. The draft model quickly generates a sequence of candidate tokens, and the target model verifies them in parallel.The bottleneck in current state-of-the-art systems (like Eagle3) is the static nature of this process. They often rely on fixed heuristics or proxy metrics—like maximizing the raw number of accepted tokens (acceptance length)—without accounting for the actual wall-clock time spent generating and verifying those tokens. If the draft model spends too much time generating a deep tree of candidates that ultimately get rejected by the target model, the efficiency gains collapse.2. The Solution: Reinforcement Learning for ThroughputLTD reframes this static heuristic process as a dynamic decision-making problem. Rather than treating drafting and verification as isolated steps, LTD formulates the interaction between the draft and target models as a Reinforcement Learning (RL) environment.Step-by-Step MechanismStep 1: Defining the Co-Adaptive PoliciesInstead of a single rigid rule, the system introduces two jointly trained, co-adaptive policies:The Drafting Policy: Decides how many tokens to generate and how to structure the candidate tree dynamically, conditioned on the current context.The Verification Policy: Adapts to the drafting policy's output to optimize the parallel verification step.Step 2: The State RepresentationThe state space encodes the current context, the structural history of the generation, and crucially, the draft model's confidence. High prediction entropy in the draft model signals a high probability of rejection, prompting the policy to truncate the draft early rather than wasting compute.Step 3: The Reward FunctionInstead of using proxy metrics, the reward signal is the actual throughput of each draft-and-verify cycle. When optimizing this via policy gradient methods or value-based approaches in PyTorch, the network directly learns the complex trade-off between the time cost of generating candidates and the expected acceptance length.Step 4: Synergistic TrainingBecause the two policies are co-adaptive, they evolve together. The drafting policy learns to only produce long sequences when the target model is highly likely to accept them, while backing off during highly uncertain, high-temperature generation steps.3. Results and ImpactBy building on the Eagle3 framework and replacing its static heuristics with this learned approach, LTD achieves significant speedup ratios (ranging from 2.24× to 4.32×). Most notably, it maintains its robustness in high-temperature scenarios—a notoriously difficult setting where the output distribution flattens, and standard dynamic speculative methods usually break down.

# **AI GENERATED IN DEPTH EXPLANATION**
Paper: **Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning**  
arXiv: 2603.01639  
This note explains the method in plain language, then reconstructs the full algorithm step by step.

---

## 1. The problem the paper is solving

Speculative decoding speeds up a **large target LLM** by letting a **smaller draft model** guess several future tokens, then having the target model verify those guesses in parallel.

The catch is that there are **two competing costs** in every speculative decoding cycle:

1. **Draft cost**: how much time the small model spends building candidate continuations.
2. **Verification cost**: how much time the large model spends checking those candidates.

A larger draft tree can increase the number of accepted tokens, but it also increases draft time and verification time. So the real goal is **not** “maximize accepted tokens,” but rather:

> **maximize throughput = accepted tokens / total cycle time**

That is the key idea behind LTD.

---

## 2. The core intuition

Prior dynamic methods usually optimize a proxy like **acceptance length**. LTD argues that this is incomplete.

Why?

- If you make the draft tree too large, you may get more accepted tokens,
  but verification becomes expensive.
- If you keep verification too small, the draft model may generate useful candidates
  that never get checked.
- So **drafting and verification should be chosen together**, not independently.

LTD therefore learns **two policies** that co-adapt:

- a **depth policy**: decides how deep to grow the draft tree
- a **size policy**: decides how many candidate tokens to send to the target model for verification

These are trained using **throughput as the reward**.

---

## 3. What a single draft-and-verify cycle looks like

A single decoding cycle starts from the most recently accepted token.

### Step 1: Start a new draft tree
The last accepted token becomes the root of a new speculative tree.

### Step 2: Expand the tree with the draft model
The draft tree is built using a beam-search-like process with:

- **beam width** `W`
- **draft depth** `D`

At the first step:
- the draft model predicts a distribution over next tokens
- the top-`W` tokens are selected as the first tree level

At later steps:
- each of the current `W` leaves produces `W` children
- so you temporarily get `W^2` candidates
- then you keep the best `W` candidates by cumulative path probability

This repeats until the system decides to stop.

### Step 3: Choose how many candidate tokens to verify
After the tree has been built, LTD does **not** necessarily send the whole tree to the target model.

Instead, it chooses a **verification size** `V`:
- select the top `V` candidate tokens in the full draft tree
- flatten them into a sequence
- build the corresponding tree-attention mask
- send that to the target model

### Step 4: Verify with the target model
The target model checks the candidates in **one forward pass**.

It accepts tokens until the first mismatch:
- all matching candidate tokens are accepted
- at the first wrong token, the rest of the candidate sequence is discarded
- the target model emits the correct next token
- that newly confirmed token becomes the root of the next cycle

### Step 5: Compute the reward
Let:

- `L_A` = number of accepted tokens in this cycle
- `T_draft` = draft time
- `T_verify` = verification time

Then LTD defines the cycle reward as:

```text
R = L_A / (T_draft + T_verify)
```

This is the throughput of the cycle.

---

## 4. Why this is better than optimizing acceptance length alone

Acceptance length alone is misleading.

Two examples:

### Case A: High acceptance, bad speed
You verify a huge number of candidates.
- Accepted tokens go up
- Verification time goes up even more
- Net throughput gets worse

### Case B: Low latency, bad speed
You verify too few candidates.
- Verification is cheap
- But you fail to exploit parallelism
- Net throughput also gets worse

LTD explicitly optimizes the balance between these extremes.

---

## 5. The two learned policies

## 5.1 Depth policy

The depth policy decides, after each draft-model forward pass:

- **CONTINUE** expanding the draft tree
- **STOP** and move to verification

So the depth policy is a sequential controller.

### Inputs to the depth policy
The paper gives it a lightweight observation:

- the log-probabilities of the candidate tokens at the **current frontier**
- current draft depth `D`
- current input length `L`

This is intentionally minimal because the depth policy is called multiple times inside one generation step, so its latency matters a lot.

### Output
A binary action:
- `0` / STOP
- `1` / CONTINUE

### Architecture
A lightweight FFN / MLP with one hidden layer of size 1024.

### Reward structure
The depth policy only gets a nonzero reward when it finally chooses **STOP**.
All intermediate CONTINUE actions get zero reward.

This creates a credit-assignment problem, so the paper uses a **high discount factor** (`gamma = 0.999`) to avoid making the policy overly biased toward stopping too early.

---

## 5.2 Size policy

The size policy decides how many tokens from the completed draft tree should be verified.

### Inputs to the size policy
It sees a richer summary of the whole drafted tree:

- log-probability scores of **all candidate tokens** in the tree
- final draft depth `D`
- total input length `L`

Padding is used to keep the observation size fixed.

### Output
A discrete action among 12 choices, mapped linearly to verification sizes:

```text
V in {20, 40, 60, ..., 240}
```

### Architecture
A 2-layer FFN / MLP with hidden sizes `[1024, 256]`.

### Why the size policy sees more than the depth policy
The size policy acts only once per cycle, after the full tree exists, so it can afford to inspect a richer state.

---

## 6. The full LTD algorithm, step by step

Here is the algorithm in operational form.

### Step 0: Initialize
You have:

- a target model
- a draft model
- beam width `W`
- a depth policy `pi_D`
- a size policy `pi_V`

You also have the current accepted prefix.

### Step 1: Set the last accepted token as the root
Begin a new speculative cycle from the token immediately after the current accepted prefix.

### Step 2: First draft expansion
Run one draft-model pass from the root and take the top `W` candidate tokens.

These become the first frontier of the tree.

### Step 3: Ask the depth policy whether to continue
Using:
- current frontier token scores
- current depth
- current sequence length

the depth policy outputs either:

- **CONTINUE** → keep growing the tree
- **STOP** → terminate drafting

### Step 4: If CONTINUE, expand one more layer
For each of the `W` current leaves:

- generate `W` children
- collect `W^2` temporary candidates
- rank by cumulative path probability
- keep the top `W`

Now you have a new frontier.

### Step 5: Repeat depth decisions
Repeat Steps 3–4 until the depth policy outputs **STOP**.

At this point, the draft tree is complete.

### Step 6: Summarize the whole tree for the size policy
Build the size-policy state from:

- log-probabilities of all tree candidates
- final depth
- input length

### Step 7: Choose the verification budget
The size policy picks a discrete action corresponding to `V`, the number of candidates to verify.

### Step 8: Select the top `V` tokens from the tree
Take the best `V` candidate tokens from the tree by predicted probability.

### Step 9: Flatten + build tree mask
Flatten the selected tree nodes into the form expected by the target model and construct the tree-attention mask.

### Step 10: Run one target-model verification pass
The target model verifies these `V` candidates in parallel.

### Step 11: Accept tokens until first mismatch
The target model:

- accepts the matching prefix of the candidates
- discards the rest after the first mismatch
- emits the correct next token at the mismatch point

### Step 12: Measure reward
Record:

- accepted length `L_A`
- draft time `T_draft`
- verification time `T_verify`

Compute:

```text
R = L_A / (T_draft + T_verify)
```

### Step 13: Append accepted tokens to the output
Update the generated sequence.

### Step 14: Start the next cycle
Use the newly confirmed token as the root of the next draft tree.

Repeat until generation is complete.

---

## 7. How RL is used

The paper formulates this as a reinforcement learning problem and uses **PPO**.

The general PPO objective is standard clipped policy optimization. The important point is not the exact formula, but what LTD feeds into it:

- **state** = draft-tree statistics + length/depth features
- **action** = continue/stop or verification size
- **reward** = throughput of the completed cycle

So LTD is directly trained to improve actual decoding efficiency rather than a proxy.

---

## 8. Why there are two policies instead of one

At first glance, you might ask:

> Why not make one policy output both depth and verification size?

The paper’s design makes sense for two reasons:

### Reason 1: The decisions happen at different times
- Depth decisions happen **during** tree growth
- Size decisions happen **after** the tree is complete

So the information available is different.

### Reason 2: Their latency constraints differ
- The depth policy is called many times, so it must be extremely cheap
- The size policy is called once, so it can inspect a richer state

This split lets the system be both adaptive and efficient.

---

## 9. The co-adaptation idea

One of the most important ideas in the paper is that the two policies are **not** best trained fully independently.

Why not?

Because the best depth depends on what verification size will later be chosen, and the best verification size depends on the kind of trees the depth policy tends to produce.

So LTD uses an iterative co-adaptation process.

### Phase 1: independent initialization
They first train each policy in a stable setting:

- **size policy** is trained while depth is randomized in `[1, 12]`
- **depth policy** is trained with a fixed verification size `V = 60`

This gives two reasonable starting policies.

### Phase 2: alternating optimization
Then they alternate:

1. freeze size policy, update depth policy against the current dynamic size behavior
2. freeze depth policy, update size policy against the current dynamic depth behavior

This allows the two policies to adapt to each other.

The paper reports that even **two rounds** of this alternating process are enough to produce strong synergy.

---

## 10. What LTD keeps from Eagle3, and what it changes

LTD is built on top of **Eagle3**.

It keeps the basic tree-based speculative decoding pipeline:

- draft model builds a tree
- target model verifies candidates with tree attention

But it replaces Eagle3’s **static heuristics** with learned policies:

- Eagle3 uses fixed or manually tuned drafting / verification settings
- LTD chooses them dynamically based on the current context and current tree

So LTD is best understood as:

> **Eagle3 + RL-based dynamic control of depth and verification size**

---

## 11. What the observation state is really capturing

The observation features are intentionally simple.

The paper’s reasoning is:

- token probabilities are strong predictors of whether candidates will be accepted
- draft depth and current sequence length are strong predictors of latency
- richer features like entropy or full hidden states can improve prediction,
  but they may cost too much at inference time

So LTD tries to get the key signals for:
- **quality** (will the candidates be accepted?)
- **cost** (how expensive will the cycle be?)

without adding much overhead.

---

## 12. Why the reward is throughput instead of acceptance length

This is the conceptual heart of the paper.

Acceptance length only tracks the numerator:

```text
accepted tokens
```

Throughput tracks the full tradeoff:

```text
accepted tokens / time
```

That makes the learned behavior different.

A policy trained on acceptance length may:
- overbuild the draft tree
- verify too many candidates
- get impressive acceptance numbers
- still be slower overall

A policy trained on throughput is pressured to find the **best speed-quality tradeoff** for each cycle.

---

## 13. Training details that matter

A few implementation choices are especially important:

### PPO
Both policies are trained with PPO.

### Discount factors
- **size policy**: `gamma = 0.9`
- **depth policy**: `gamma = 0.999`

The larger gamma for the depth policy is important because its reward arrives only when it stops.

### Policy architectures
- size policy actor/critic: 2 layers, `[1024, 256]`
- depth policy actor: 1 layer, `1024`
- depth policy critic: same as the size-side value net according to the appendix

### Initial training setup
- size policy: 100k steps, random draft depths from 1 to 12
- depth policy: 1M steps, fixed verification size 60

---

## 14. Why the overhead stays small

A natural concern is:

> If we add neural policies, do we lose the speedup?

The paper argues the answer is mostly no, because:

- the policies are lightweight MLPs
- the depth policy uses only minimal features
- the size policy runs only once per cycle

Reported overhead is small:
- under 1.5% on Llama
- around 1.2% on Vicuna

So the policy cost is much smaller than the speed gains they report.

---

## 15. What the empirical results say

The paper reports:

- overall speedup ratios between **2.24× and 4.32×**
- improvement over Eagle3 up to **36.4%**
- robustness under higher-temperature sampling, with about **5%** gain where other dynamic methods often become ineffective

The main message is not just “RL works,” but:

> directly optimizing throughput and co-adapting depth + verification gives better real inference speed than optimizing a proxy metric in isolation.

---

## 16. A compact mental model

Here is the simplest way to think about LTD:

### Old approach
“Draft a tree with a mostly fixed rule, verify with a mostly fixed rule.”

### LTD
“At each decoding cycle:
- decide how much drafting is worth doing
- decide how much verification is worth spending
- learn both decisions from real wall-clock payoff”

So LTD turns speculative decoding from a heuristic scheduling problem into a learned control problem.

---

## 17. Pseudocode view

```text
while not finished:

    root = last accepted token
    tree = initialize(root)

    # Sequentially grow the tree
    while True:
        expand tree frontier with draft model
        observe frontier scores + depth + length
        action_D = depth_policy(state_D)

        if action_D == STOP:
            break

    # Choose verification budget
    observe all tree scores + final depth + length
    V = size_policy(state_V)

    selected_tokens = top_V_tokens(tree, V)
    flat_tokens, tree_mask = flatten_with_tree_mask(selected_tokens)

    verified_result = target_model_verify(flat_tokens, tree_mask)

    accepted_tokens = matching_prefix_until_first_mismatch(verified_result)
    output.extend(accepted_tokens)

    reward = len(accepted_tokens) / (draft_time + verify_time)

    update / record RL experience
```

---

## 18. What is genuinely novel here

The novelty is not just “use RL somewhere in speculative decoding.”

The real contributions are:

1. **Optimize throughput directly**
   - not acceptance length alone

2. **Use two coordinated policies**
   - one for draft depth
   - one for verification size

3. **Train them to co-adapt**
   - so drafting and verification become jointly optimized

4. **Keep inference overhead tiny**
   - by using simple observations and lightweight MLPs

---

## 19. Limitations / things to keep in mind

Even though the method is strong, a few caveats are worth noting:

- It is built on the tree-speculative decoding setup, so it inherits the complexity of that infrastructure.
- It needs RL training, which is extra engineering compared with fixed heuristics.
- Its behavior is shaped by the chosen training distribution and model family.
- The paper uses lightweight MLP policies; that helps latency, but it may also limit representational power.

These do not invalidate the method, but they matter if you are thinking about extensions.

---

## 20. One-sentence summary

**LTD accelerates tree-based speculative decoding by learning, with PPO, how deep to grow the draft tree and how many candidates to verify, using cycle throughput as the reward so drafting and verification are optimized together rather than separately.**

---

## 21. Citation guide for this note

This explanation is based on:

- the paper abstract and introduction
- Section 3.1–3.2 (throughput motivation and draft-and-verify cycle)
- Section 4 (method)
- Appendix A.2 / A.3 / A.8 (policy architecture, observations, actions, discount factors, overhead-related details)
