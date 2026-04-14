# How LTD Works — A Beginner-Friendly Recipe

This doc explains, from scratch, how **Learning to Draft (LTD)** speeds up tree-based
speculative decoding with reinforcement learning.

**Assumed background.** You've seen speculative decoding once before: you know there's
a small *draft* model that guesses ahead and a big *target* model that checks the
guesses in one forward pass. You do **not** need to know anything about PPO or RL —
we'll build that up with a toy example before applying it.

**Style.** This is written like a cooking recipe: first the dish (what LTD produces),
then the ingredients (the pieces), then the steps, then the training procedure, then
the fine-tuning details. Read top to bottom.

---

## Part 1 — The Dish: What LTD Actually Does

### 1.1 The setup LTD inherits from Eagle3

LTD is a drop-in *controller* on top of **Eagle3**, a tree-based speculative
decoding system. The Eagle3 pipeline looks like this, every generation cycle:

1. Start from the last token the target model has confirmed.
2. Use the draft model to grow a small **tree** of possible continuations
   (beam-search-like: keep the best `W` paths at each level, grow `D` levels deep).
3. Pick some number `V` of candidate tokens from the tree.
4. Run the target model **once** on all `V` candidates (via tree attention).
5. The target accepts every candidate that matches its own prediction, stops
   at the first mismatch, and emits the correct replacement token.

The two knobs you can tune are the **draft depth `D`** and the **verification
size `V`**. Eagle3's baseline just picks fixed values (e.g., `D=8`, `V=60`) — a
one-size-fits-all setting that you tune once and leave alone.

### 1.2 Why fixed D and V is leaving speed on the table

Think about what `D` and `V` do to cycle time:

- **Bigger `D`**: draft model runs more times → draft time goes up, but you
  might collect more accepted tokens per cycle.
- **Bigger `V`**: target model processes more tokens in one forward pass →
  verification time goes up (in discrete jumps, because GPUs are batch-happy),
  but you get more chances of accepting a long prefix.

The **optimal** `D` and `V` depend on the current context. An "easy" context
(the draft model is confident and usually right) rewards a deep tree; a "hard"
context (the draft model is flailing) rewards stopping early. A fixed setting
can't adapt.

### 1.3 The real goal: throughput, not acceptance length

Prior adaptive methods tried to maximize **acceptance length** — the number of
tokens accepted per cycle. LTD argues this is the wrong target.

Picture two strategies:

- **Strategy A (greedy):** Build a huge tree, verify 240 tokens. Acceptance
  length is high (e.g., 8 tokens). But you spent so much time drafting and
  verifying that each cycle takes ages. Net tokens/second: low.
- **Strategy B (timid):** Draft one token, verify it. Acceptance length is
  1. Each cycle is fast, but you barely parallelize. Net tokens/second: low.

The right target is **throughput**:

```
throughput = accepted_tokens / (draft_time + verify_time)
```

LTD trains its policies to directly optimize this quantity, measured
per-cycle in real wall-clock terms.

---

## Part 2 — The Ingredients

LTD adds **two small neural networks** ("policies") to the Eagle3 pipeline.
Nothing else in the pipeline changes.

| Policy | Decides | Called | Architecture |
|---|---|---|---|
| **Depth policy** | CONTINUE or STOP growing the tree | many times per cycle, once per draft layer | 1-layer MLP, 1024 hidden, binary output |
| **Size policy** | How many tokens `V` to verify | once per cycle, after tree is built | 2-layer MLP, [1024, 256] hidden, 12-way softmax |

Note that the size policy has a bigger brain: it runs only once, so it can
afford to look at more data. The depth policy runs every layer, so it has to
stay cheap.

Both policies are tiny compared to the LLMs they sit next to. The paper
reports that together they add <1.5% overhead on Llama.

---

## Part 3 — One Decoding Cycle, With a Baby Example

Let's walk through one complete LTD cycle with made-up numbers. Set:

- Beam width `W = 2` (each leaf expands into 2 children; we keep the best 2).
- Vocabulary has 3 tokens: `A`, `B`, `C`.
- The last confirmed token (our root) is `<the>`.

### Step 1 — First draft expansion

The draft model runs on `<the>` and gives probabilities:

```
P(A | <the>) = 0.6
P(B | <the>) = 0.3
P(C | <the>) = 0.1
```

Top `W=2` tokens become the first frontier:

```
Tree layer 1:  A (0.6)   B (0.3)
```

### Step 2 — Ask the depth policy: keep going?

Build the observation vector for the depth policy. It needs:

- **Log-probs of candidates at the current frontier.** For `W=2`, we show the
  last layer's `W² = 4` candidate scores. (Right now we only have 2 scores;
  the paper pads or uses the top-`W²` from the last expansion as relevant;
  implementation detail.)
- **Current depth** `D = 1`.
- **Current input length** `L = however many tokens are in context`.

The depth policy's MLP reads this vector and outputs a binary action.
Suppose it says **CONTINUE**.

### Step 3 — Expand one more layer

Each of the 2 leaves runs through the draft model. We get `W × W = 4` new
candidates with **cumulative** path probabilities (parent prob × child prob):

```
Path A→A:  0.6 × 0.5 = 0.30
Path A→B:  0.6 × 0.4 = 0.24
Path B→A:  0.3 × 0.7 = 0.21
Path B→C:  0.3 × 0.2 = 0.06
```

We keep the top `W = 2` paths:

```
Tree layer 2 frontier:  A→A (0.30)   A→B (0.24)
```

The other two paths (`B→A`, `B→C`) are **dead** at the frontier, but the
intermediate nodes `A` and `B` still exist in the tree.

### Step 4 — Ask the depth policy again

Build a fresh observation from the new frontier's 4 scores (the `W²`
candidates before we pruned down to 2), plus `D = 2`, `L = ...`.

Suppose it says **STOP** this time. Drafting is done.

### Step 5 — Ask the size policy how much to verify

Now we flatten the entire tree's candidate list. Suppose the tree has
6 nodes in total. Their log-probabilities are padded into a fixed-length
vector, concatenated with `D = 2` and `L`, and fed to the size policy.

The size policy has 12 possible outputs. They map linearly to:

```
V ∈ {20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240}
```

Suppose it picks `V = 40`. (In our baby example we only have 6 candidates,
so `V=40` really means "all of them" — in practice trees are bigger.)

### Step 6 — Verify

Select the top `V = 40` candidate tokens by predicted probability, flatten
into a sequence, build the tree-attention mask, and run the target model
**once**. The target accepts the longest matching prefix; at the first
mismatch, it emits the correct token.

Say it accepts 3 tokens from the tree plus the correction → `L_A = 4`.

### Step 7 — Measure the reward

We timed the whole cycle:

- `T_draft = 8 ms` (draft model forward passes)
- `T_verify = 12 ms` (target model forward pass)

Throughput:

```
R = 4 / (8 + 12) = 4 / 20 = 0.2 tokens/ms
```

This scalar is the learning signal. The policies will be nudged towards
behaviors that make this number go up.

### Step 8 — Next cycle

The newly confirmed last accepted token becomes the root of a fresh tree.
Back to Step 1.

---

## Part 4 — Who Is The Policy? A Detour Into PPO

Now: how do we actually train those two MLPs so they make good CONTINUE/STOP
and `V`-picking decisions?

LTD uses **Proximal Policy Optimization (PPO)**. If you've never seen PPO,
here's the full concept from scratch in one page.

### 4.1 The setup PPO solves

You have an agent that sits in a **state** `s` and picks an **action** `a`.
After acting, the world gives you a **reward** `r` and a new state `s'`.

The agent's brain is the **policy** `π_θ(a | s)`: a function (with trainable
parameters `θ`) that takes a state and outputs a *probability distribution*
over actions.

For LTD's depth policy:

- state `s` = current frontier's log-probs + depth + length
- action `a` ∈ {CONTINUE, STOP}
- `π_θ(a | s)` = a 2-number softmax output from the MLP

You want to find `θ` that makes `π_θ` collect lots of reward, on average,
over many episodes.

### 4.2 The naive idea (REINFORCE)

If I take an action `a` in state `s` and later collect total reward `G`,
I want to **increase** `π_θ(a | s)` if `G` was high, and **decrease** it
if `G` was low.

That's literally the rule. Gradient:

```
∇θ log π_θ(a | s) · G
```

Run this on every action you took, sum, step. Done. This is REINFORCE.

**Problem:** huge variance. Your reward `G` bounces around wildly depending
on random things that had nothing to do with this particular action.

### 4.3 The fix: use advantage, not raw reward

Instead of asking "did this action lead to high reward?" we ask "did this
action lead to *higher reward than my baseline expectation from this state*?"

Define:

- `V(s)` = "how much reward do I *usually* collect starting from state `s`?"
  This is called the **value function**, and we train a separate MLP (the
  **critic**) to estimate it.
- `A(s, a)` = `G - V(s)` = **advantage** of action `a`: how much better
  than average this particular action turned out.

Now the gradient is:

```
∇θ log π_θ(a | s) · A(s, a)
```

If `A > 0`, this action was surprisingly good → reinforce it. If `A < 0`,
it was surprisingly bad → dampen it. Much less variance.

### 4.4 The PPO-specific trick: don't change the policy too fast

REINFORCE-with-advantage has one more problem: it can take **huge** gradient
steps and overshoot, landing the policy somewhere terrible. PPO's fix: when
you do a gradient step, **clip the change** so the new policy doesn't deviate
too far from the old one.

Concretely, PPO computes the **ratio** of new to old policy probabilities:

```
ρ(θ) = π_θ(a | s) / π_θ_old(a | s)
```

When `θ = θ_old`, `ρ = 1`. As you update, `ρ` moves away from 1. PPO's
objective is:

```
J(θ) = E[ min( ρ · A , clip(ρ, 1-ε, 1+ε) · A ) ]
```

Where `ε` is a small number like `0.2`. The `clip` says: if the ratio tries
to go above `1 + ε` (policy probability growing fast) or below `1 - ε`
(shrinking fast), cap it.

The `min(...)` makes the clipping conservative: you only clip when doing
so would make the objective *smaller*, i.e., you never let the optimizer
exploit a large ratio to push an already-overconfident action further.

### 4.5 A baby PPO example

Let's train a stupid policy on a 1-state, 2-action problem. State is always
`s₀`. Actions are `LEFT` and `RIGHT`. Rewards: `LEFT → 1`, `RIGHT → 0`.

Initialize the policy so `π_θ(LEFT | s₀) = π_θ(RIGHT | s₀) = 0.5`.

**Rollout 1:** sample 4 actions from the policy.
```
LEFT  → reward 1
RIGHT → reward 0
LEFT  → reward 1
RIGHT → reward 0
```

Baseline (estimated value) for `s₀`: average reward so far ≈ 0.5. So:

- For the two `LEFT` transitions: `A = 1 - 0.5 = +0.5`
- For the two `RIGHT` transitions: `A = 0 - 0.5 = -0.5`

**Gradient step:** compute `ρ = π_θ(a) / π_θ_old(a)` for each. Right after
a rollout, `θ = θ_old`, so `ρ = 1` initially. Optimize the PPO objective for
several epochs over this batch of 4 transitions.

As `θ` updates, `π_θ(LEFT | s₀)` starts rising above `0.5`. Suppose after
some inner epochs it's at `0.7`. Then:

- For `LEFT` samples: `ρ = 0.7 / 0.5 = 1.4` (above `1 + ε = 1.2`) →
  clipped. Gradient capped. No more pushing.
- For `RIGHT` samples: `ρ = 0.3 / 0.5 = 0.6` (below `1 - ε = 0.8`) →
  clipped. Gradient capped. No more pushing.

PPO stops the policy from lurching any further in one update. We gather a
new rollout, repeat.

After a few rollouts, the policy converges to `π_θ(LEFT | s₀) ≈ 1`,
exactly as you'd want.

### 4.6 Putting it together

So every PPO iteration does:

1. **Rollout.** Run the current policy for some number of episodes, collecting
   `(s, a, r, s')` tuples.
2. **Advantage estimation.** Compute `A(s, a)` for every transition using the
   critic's current value estimate.
3. **Policy update.** Take several gradient steps on the PPO clipped
   objective, using the rollout data. LTD does 20 epochs per update.
4. **Critic update.** Also train the value MLP to better predict `V(s)`.
5. **Repeat.**

That's the whole algorithm. Everything below in LTD is just a specific
choice of state, action, reward, and network architecture plugged into
this loop.

---

## Part 5 — PPO Applied to LTD

### 5.1 The depth policy's PPO setup

- **State.** `W²` candidate log-probs from the current frontier expansion +
  scalar current depth `D` + scalar input length `L`.
- **Action.** Discrete ∈ {CONTINUE, STOP}.
- **Reward.** `0` for every CONTINUE. On the STOP action, you get the whole
  cycle's throughput `R = L_A / (T_draft + T_verify)` as a single terminal
  reward.
- **Discount.** `γ = 0.999`. (Very high — see 5.3.)
- **Architecture.** 1-layer MLP, 1024 hidden. Value net is shared with the
  size policy's value architecture (2 layers).

### 5.2 The size policy's PPO setup

- **State.** All candidate log-probs in the full draft tree (padded to fixed
  length) + final `D` + `L`.
- **Action.** Discrete ∈ {0, 1, ..., 11}, mapped linearly to
  `V ∈ {20, 40, ..., 240}`.
- **Reward.** This policy acts **once** per cycle, and immediately gets
  the cycle's throughput `R` as its reward. No bootstrapping needed.
- **Discount.** `γ = 0.9`. (Lower — only one decision.)
- **Architecture.** 2-layer MLP, [1024, 256] hidden. Value net same shape.

### 5.3 Why `γ = 0.999` for the depth policy

This is the most subtle detail in LTD. Imagine an untrained depth policy
that outputs STOP at every step. It grows a depth-1 tree, and collects
some small reward.

With low discount (`γ = 0.5`), the reward for an early CONTINUE action
would be heavily discounted by the time STOP pays out, making every
CONTINUE look worthless. The policy would collapse to "always STOP", which
gets you shallow trees and poor throughput.

With `γ = 0.999`, CONTINUE actions still see most of the future reward
even if STOP is several steps away. The policy can actually learn that
deep trees sometimes pay off a lot, and it's worth a few CONTINUE steps to
get there.

### 5.4 What the observation states look like, at a glance

**Depth policy** sees a skinny state (fast to compute):

```
[ log_prob_1, log_prob_2, ..., log_prob_(W²), D, L ]
```

**Size policy** sees a wide state (it runs once, so it can afford this):

```
[ log_prob_1, ..., log_prob_N, (padding zeros), D, L ]
```

where `N` is the number of candidate nodes across the whole tree.

---

## Part 6 — The Training Recipe

You can't just start both policies at random and run joint PPO end-to-end
— they'd destabilize each other. Here's LTD's actual recipe.

### Step 1 — Train the size policy with a random depth generator

Set up an environment where the draft depth `D` is sampled **uniformly** at
random from `{1, 2, ..., 12}` each cycle. Run PPO to train only the size
policy against this.

- Duration: 100,000 training steps.
- The depth is not a learned thing yet — it's noise.
- Why: we get a size policy that's robust to *any* tree depth it might
  encounter, before we start training a real depth policy.

### Step 2 — Train the depth policy with fixed `V = 60`

Now freeze the size policy. Actually, just fix `V = 60` directly (matching
Eagle3's default). Run PPO to train only the depth policy.

- Duration: 1,000,000 training steps. (Longer, because this one is harder:
  sparse reward + long action chain.)
- Why: we want a depth policy that's good against a sensible fixed `V`,
  before we let them chase each other.

### Step 3 — Iterate (the co-adaptation loop)

Now alternate:

- Freeze the size policy. Update the depth policy for some steps.
- Freeze the depth policy. Update the size policy for some steps.

Why this order matters: the best depth *depends* on what `V` will be, and
vice versa. Doing it jointly from scratch is unstable. Doing it alternately
lets each policy catch up to the other's current behavior.

The paper reports that even **two rounds** of alternation give most of the
benefit.

### Step 4 — Ship it

At inference time, both policies are frozen MLPs that get called during
decoding. No more learning, no gradients, nothing. Just forward passes.

---

## Part 7 — Hyperparameters Worth Memorizing

From Appendix A of the paper (you'll want these if you reproduce LTD):

| Thing | Value |
|---|---|
| PPO optimizer | Adam |
| PPO epochs per update | 20 |
| Entropy coefficient | 0.01 |
| Learning rate | 1e-3, linear decay to 0 |
| LR warmup | first 1% of training steps |
| Size policy `γ` | 0.9 |
| Depth policy `γ` | 0.999 |
| Size policy hidden dims | [1024, 256] |
| Depth policy hidden dims | [1024] |
| Activations | ReLU |
| PPO clip `ε` | default (0.2) from Stable-Baselines3 |
| Library | Stable-Baselines3 |
| Size policy training steps | 100,000 |
| Depth policy training steps | 1,000,000 |
| Size policy action space | 12 discrete `V`s, linearly mapped to {20,...,240} |
| Depth policy action space | binary (CONTINUE/STOP) |
| Beam width `W` | usually 10 (Eagle3 default) |

Training data (both policies): HumanEval prompts, run through the paired
draft+target models.

---

## Part 8 — What's Actually New (vs. Eagle3)

It's worth spelling out what LTD keeps vs. changes:

**Keeps (unchanged from Eagle3):**
- Tree-based draft generation
- Tree-attention verification in the target model
- The draft and target model weights themselves

**Changes:**
- The draft depth `D` is now **dynamic**, decided online by the depth policy
  instead of being a fixed hyperparameter.
- The verification size `V` is now **dynamic**, decided online by the size
  policy instead of being a fixed hyperparameter.
- The optimization target becomes **throughput** rather than acceptance length.

That's it. The rest is plumbing.

---

## Part 9 — Known Limitations (Which Are Our Opening)

LTD is strong but has four seams worth poking at for follow-up work:

1. **Width is pruned after construction, not during.** The draft model still
   expands every leaf into `W` children and builds the full `W`-wide tree.
   Only *after* that does the size policy trim the verification set. That's
   wasted drafter compute on branches we end up throwing away.
2. **Size policy ablates to a small gain.** LTD's own ablation shows the
   depth policy does most of the work; the size policy adds comparatively
   little. So the two-policy split may be over-engineered.
3. **MLPs are thin.** The policies are 1–2 layer MLPs. You could plausibly
   capture the draft frontier better with SSMs or attention over the
   frontier, if the extra overhead is worth it.
4. **PPO is sample-inefficient.** 1M steps for the depth policy alone.
   Offline methods (imitation learning from an oracle, DAgger, SAC) might
   get there faster.

---

## Part 10 — One-Paragraph Summary

**LTD** takes Eagle3 speculative decoding and replaces its two static
hyperparameters — draft depth `D` and verification size `V` — with two
small MLPs that decide `D` and `V` per cycle. The MLPs are trained with
**PPO**, using the cycle's actual throughput (accepted tokens divided by
total wall-clock time) as the reward. A two-stage curriculum (train the
size policy against random depths, train the depth policy against a fixed
`V=60`, then alternate) stabilizes the joint optimization. The result is
a **2.24–4.32× speedup** over vanilla autoregressive decoding and up to
**+36.4%** over Eagle3, with policy overhead under 1.5%.

---

## Glossary

- **Acceptance length (`L_A` or `τ`)**: Number of tokens the target model
  accepts per draft-and-verify cycle.
- **Action**: In RL, the thing the agent does. Here: CONTINUE/STOP, or
  which `V` to pick.
- **Advantage `A(s, a)`**: How much better an action turned out than the
  baseline `V(s)` expected.
- **Beam width `W`**: Number of paths kept at each tree layer during
  drafting.
- **Critic / value network**: A second MLP that estimates `V(s)`, used to
  reduce gradient variance in PPO.
- **Draft depth `D`**: Number of tree layers the drafter builds before
  stopping.
- **Discount factor `γ`**: How much future rewards are worth relative to
  immediate ones. Closer to 1 means future rewards count almost as much as
  immediate ones.
- **Episode**: One decoding cycle — from accepting a token to accepting
  the next batch.
- **PPO clip `ε`**: Ceiling on how far the new policy's action probabilities
  can move from the old policy's in a single update.
- **Policy `π_θ`**: The actor MLP. Maps state to action probabilities.
- **Reward `R`**: Throughput of the cycle, `L_A / (T_draft + T_verify)`.
- **Rollout**: A batch of `(state, action, reward)` transitions collected
  by running the current policy.
- **State `s`**: Everything the policy looks at before acting. For LTD,
  a vector of candidate log-probs plus scalars for depth and length.
- **Target model**: The big LLM you're trying to run faster.
- **Throughput**: Tokens produced per unit wall-clock time.
- **Verification size `V`**: Number of candidate tokens from the tree that
  get sent to the target model for one-shot parallel verification.
