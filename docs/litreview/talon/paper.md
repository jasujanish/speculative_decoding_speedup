Talon: Confidence-Aware Speculative Decoding with Adaptive Token Trees

- 

- 1 Introduction

- 
2 Related Work

- Chain-based speculative decoding.

- Tree-based speculative decoding.

- 
3 Background

- 3.1 Chain-based Drafting

- 3.2 Tree-based Drafting

- 
4 Motivated Experiments

- 4.1 The "Acceptance Funnel" Phenomenon

- 4.2 Variance in Real-World Accepted Length

- 
5 Method

- 5.1 From Static Grids to Dynamic Budgets

- 
5.2 Hybrid Tree Expansion Strategy

- 
5.2.1 Confidence-Gated Expansion

- Intuition.

- 5.2.2 Robust Tree Initialization

- 5.3 Evaluating Talon with Draft Efficiency

- 
6 Experiments

- 
6.1 Experimental Setup

- Tasks and Datasets.

- Implementation Details.

- 6.2 Main Results

- 6.3 Evaluating Talon with Draft Efficiency

- 
6.4 Ablation Study

- 
6.4.1 Ablating Robust Tree Initialization

- Necessity of Robust Initialization (k=1​v​s.k=0k=1\ vs.\ k=0).

- Overhead of additional Robust Initialization (k≥2k\geq 2).

- 6.4.2 Ablating Confidence-gated Expansion

- 6.5 Case Study

- 7 Conclusion

- 
8 Limitations

- Scalability to Large Batch Sizes.

- Hyperparameter Generalization.

- 
9 Ethic Statements

- Inheritance of Model Behaviors.

- A Formalized Algorithms

- B Token-Tree Verification

- 
C Additional Motivated Experiments

- C.1 Heat Map Visualization of LlaMA-3.1-Instruct-8B

- C.2 Dynamic Accepted Length of Various Models

- D Derivation of Draft Efficiency

- 
E Details of Experiments

- 
E.1 Implementation Details

- E.1.1 Comparison Baselines

- 
E.1.2 Hardware and Software Configurations

- Hardware Environment.

- Software Environment.

- Generation Configuration.

- E.1.3 Draft Model Selection

- E.2 More Visualization Results of Draft Efficiency

- E.3 More Evaluation Results under Temperature Settings

- 
E.4 More Ablation Studies of Talon

- Impact of Global Token Budget (NN).

- Sensitivity to Confidence Threshold (μ\mu).

- E.5 More Case Studies of Talon

- E.6 Tree Construction Overhead

- 
F More Discussions to Related Work

- Chain-based speculative decoding.

- Tree-based speculative decoding.

- Summary and positioning of Talon.

- G LLM Usage

## 

Talon: Confidence-Aware Speculative Decoding with Adaptive Token Trees

Tianyu Liu1,2  Qitan Lv1,2∗  Yuhao Shen3  Xiao Sun2  Xiaoyan Sun1  
1University of Science and Technology of China 
2Shanghai AI Laboratory  3Zhejiang University
{tianyu_liu, qitanlv}@mail.ustc.edu.cn  riven@zju.edu.cn
sunxiao@pjlab.org.cn
 sunxiaoyan@ustc.edu.cn
Equal Contribution.The Corresponding Author.

## Abstract

Speculative decoding (SD) has become a standard technique for accelerating LLM inference without sacrificing output quality.
Recent advances in speculative decoding have shifted from sequential chain-based drafting to tree-structured generation, where the draft model constructs a tree of candidate tokens to explore multiple possible drafts in parallel.
However, existing tree-based SD methods typically build a fixed-width, fixed-depth draft tree, which fails to adapt to the varying difficulty of tokens and contexts.
As a result, the draft model cannot dynamically adjust the tree structure to early stop on difficult tokens and extend generation for simple ones.
To address these challenges, we introduce Talon, a training-free, budget-driven adaptive tree expansion framework that can be plugged into existing tree-based methods.
Unlike static methods, Talon constructs the draft tree iteratively until a fixed token budget is met, using a hybrid expansion strategy that adaptively allocates the node budget to each layer of the draft tree.
This framework naturally shapes the draft tree into a “deep-and-narrow” form for deterministic contexts and a “shallow-and-wide” form for uncertain branches, effectively optimizing the trade-off between exploration width and generation depth under a given budget.
Extensive experiments across 5 models and 6 datasets demonstrate that Talon consistently outperforms state-of-the-art Eagle-3, achieving up to 5.16×\times end-to-end speedup over auto-regressive decoding.

Talon: Confidence-Aware Speculative Decoding with Adaptive Token Trees

Tianyu Liu1,2††thanks: Equal Contribution.   Qitan Lv1,2∗   Yuhao Shen3   Xiao Sun2††thanks: The Corresponding Author.   Xiaoyan Sun1

1University of Science and Technology of China

2Shanghai AI Laboratory  3Zhejiang University

{tianyu_liu, qitanlv}@mail.ustc.edu.cn  riven@zju.edu.cn

sunxiao@pjlab.org.cn
 sunxiaoyan@ustc.edu.cn

## 
1 Introduction

Figure 1: Illustration of chain-based drafting and tree-based drafting (Eagle-3 (Li et al., 2025a)) with K=2K=2. At each step, Eagle calls draft model forward on the selected KK parent nodes of last step, selects top-KK child nodes for each parent, and filters K×KK\times K child nodes. Then Eagle employs an additional top-KK operation to choose KK nodes as parents for next step.

While Large Language Models (LLMs) have achieved remarkable success in various benchmarks (OpenAI Team, 2025; Google DeepMind Team, 2025; Anthropic Team, 2025; QwenTeam et al., 2025; DeepSeek-AI et al., 2025; Team et al., 2025), their deployment is severely constrained by their auto-regressive token-by-token generation.
This sequential dependency prevents models from predicting multiple tokens (Gloeckle et al., 2024) in a single step, causing inference latency to scale linearly with output length (Pope et al., 2022) and making real-time interaction computationally expensive.

To alleviate this limitation, Speculative Decoding (SD) (Leviathan et al., 2023; Chen et al., 2023) has emerged as a promising paradigm to break the strict sequential dependency (Liu et al., 2025a).
SD decouples each decoding step into two sub-procedures: efficient drafting and parallel verification. A lightweight draft model first proposes a short candidate sequence, and the target LLM then validates all proposed tokens in parallel, accepting multiple tokens with a single target model forward.

However, early speculative decoding methods typically employ a chain-based drafting strategy, which is inherently vulnerable.
A single rejection at the beginning of the draft sequence invalidates all subsequent tokens, leading to a substantial waste of computational resources (Miao et al., 2024).
To further improve the overall acceptance rate, recent works have shifted from chain-based to tree-based drafting (Miao et al., 2024; Cai et al., 2024; Li et al., 2025b; Du et al., 2024; Li et al., 2024).
Instead of predicting a single sequence, these approaches construct a token tree covering multiple plausible continuation paths.
By leveraging Tree Attention (Yao et al., 2025), the target LLM can verify multiple draft token sequences in parallel within a single forward pass, significantly increasing the draft acceptance rate and overall acceleration.

Figure 2: Limitations of tree-based drafting methods. (a) when the model is already confident to its prediction, the draft tree still grows kk child nodes. (b) when the model is very confused and highly uncertain, the top-KK draft tokens are still not sufficient.

State-of-the-art tree-based methods, such as Eagle (Li et al., 2025b), construct the draft tree via a rigid layer-wise expansion mechanism.
As illustrated in Figure 1, these approaches iteratively expand a fixed number of child nodes for every parent and select the top-KK candidates to proceed to the next layer.
However, this strategy enforces a static tree structure with pre-determined dimensions.
Such rigid allocation fails to adapt to the model’s dynamic confidence (output probabilities of its tokens) 111“Dynamic” in Eagle means that the draft model dynamically select top-KK nodes as next-layer parents, but it cannot adjust kk to adapt for contexts. We provide an example in Figure 8 in Appendix to further illustrate the difference.: it still expands kk child nodes even if the draft model is already confident (shown in Figure 2(a)), while it cannot allocate more node budget when the model is confused and highly uncertain (shown in Figure 2(b)). Moreover, it prematurely truncates high-confidence paths that could extend deeper, while squandering the computational budget on expanding low-probability branches in uncertain contexts.

To tackle these inefficiencies, we introduce Talon, a novel training-free, budget-driven adaptive tree expansion framework to opTimize drAft tree for faster specuLative decOdiNg.
Departing from the rigid fixed width and depth generation, Talon employs a dynamic growth algorithm that incrementally expands the draft tree until the number of nodes reaches a given budget NN.
Specifically, we design a hybrid expansion strategy to optimize the tree topology:
at the first layer, we propose a fixed width initialization that utilizes a top-KK operation to alleviate the early rejection phenomenon. At subsequent layers, we propose a confidence-gated expansion strategy to adaptively allocate the node budget.
This allows the draft structure to evolve to deep-and-narrow for deterministic contexts to maximize draft length, or shallow-and-wide for uncertain ones to enhance hit rate.

To summarize, our contributions are:

- 
(i)

We identify a key limitation in existing tree-based speculative decoding methods: the strict, layer-wise static tree expansion fails to adapt to the model’s varying confidence, leading to inefficient utilization of the computational budget.

- 
(ii)

We propose Talon, a budget-driven speculative decoding framework. By employing a dynamic tree growth algorithm with robust tree initialization and confidence-gated expansion, Talon constructs adaptive draft trees that dynamically adapts (deep-and-narrow or shallow-and-wide) based on context uncertainty.

- 
(iii)

Extensive experiments verify the effectiveness of Talon. It significantly outperforms state-of-the-art method Eagle-3 in terms of draft efficiency and wall-clock speedup, particularly in scenarios with fluctuating generation difficulty.

## 
2 Related Work

In this section, we review speculative decoding from the perspective of drafting structures: chain-based speculative decoding and tree-based speculative decoding.
We position Talon in the context of extensive related works in Appendix F.

## Chain-based speculative decoding.

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) introduces a lossless draft-and-verify (Zhang et al., 2024a) principle: a cheap drafter proposes a short continuation and a target model verifies it in parallel, accepting the longest matched prefix.
This paradigm suffers from a clear efficiency bottleneck—early rejection (Sun et al., 2025)—since a mismatch at an early position invalidates all downstream drafted tokens.
Follow-up work improves chain-based SD by (i) producing better proposals with minimal overhead (e.g., lightweight head-based drafting that reuses target representations(Cai et al., 2024; Du et al., 2024; Li et al., 2025b)), and (ii) reducing wasted draft computation via adaptive lookahead (Huang et al., 2025) or pipelined scheduling (Liu et al., 2025a).
Nevertheless, the speedup is fundamentally sensitive to the hardest tokens, because one rejection discards the entire suffix.

## Tree-based speculative decoding.

Tree-based SD generalizes the draft from a single chain to a token tree, allowing the verifier to choose among multiple candidate branches within one forward and thus mitigating early rejection.
SpecInfer (Miao et al., 2024) is an early representative that organizes candidates as a token tree and verifies them in parallel with tree attention.
More recent work strengthens the tree-based pipeline from two angles: (i) improving tree proposals via well-calibrated drafters (Zhang et al., 2024b; Huo et al., 2025; Gao et al., 2025) and context-aware dynamic draft trees (Xiong et al., 2024), and (ii) optimizing the draft-tree structure under a node budget, including dynamic-programming-based and search-based construction (e.g., Sequoia (Chen et al., 2025) and OPT-Tree (Wang et al., 2024)).
However, existing methods typically rely on rigid or heuristic expansion patterns, failing to explicitly allocate a token budget based on real-time confidence.
This gap motivates Talon, a training-free, budget-driven framework that constructs adaptive token trees (deep-and-narrow vs. shallow-and-wide).

## 
3 Background

We first formulate chain-based drafting and tree-based drafting for better understanding.

## 
3.1 Chain-based Drafting

In standard speculative decoding, a smaller draft model ℳd\mathcal{M}_{d} generates a single sequence of γ\gamma tokens (a chain) as a speculation for the target model ℳt\mathcal{M}_{t}. Given the prefix x≤tx_{\leq t}, the drafting process generates a sequence xt+1,…,xt+γx_{t+1},\dots,x_{t+\gamma} auto-regressively:

xt+i∼Pℳd(⋅|x<t+i),1≤i≤γx_{t+i}\sim P_{\mathcal{M}_{d}}(\cdot|x_{<t+i}),\quad 1\leq i\leq\gamma

(1)

The target model ℳt\mathcal{M}_{t} then verifies this sequence in parallel. The key limitation of this approach is the sequential dependency during verification: if the token at position ii is rejected, all subsequent tokens x>ix_{>i} are discarded regardless of their correctness, resulting in wasted computation.

## 
3.2 Tree-based Drafting

Figure 3: Visualization of token acceptance frequency within a static draft tree (K=10,D=8K=10,D=8). The heatmap reveals an "Acceptance Funnel" effect:
while the acceptance frequency of the first layer is relatively uniform, the acceptance in subsequent layers (d≥1d\geq 1) shows a funnel trend that the accepted tokens concentrate more on high confidence regions (e.g., top-1 and top-2), rendering the wide static expansion computationally wasteful. Note that the first layer only has KK nodes, while its subsequent layers have K×KK\times K nodes.

To mitigate the limitations of chain-based drafting, recent works (e.g., Eagle) verify a token tree 𝒯\mathcal{T} to cover diverse paths.
As formalized in Algorithm 1 and Figure 1, these methods employ a static construction strategy: at each depth dd, the model takes a parallel forward on all parent nodes and outputs their next-token distributions. Then, it selects top-KK entries from the output distribution of each parent node, acquiring K×KK\times K leaf nodes. After that, the model measures the path score of each layer node vv:

p​(v)=∏j∈Path​(xt,v)Pℳd​(j|x<j)p(v)=\prod_{j\in\text{Path}(x_{t},v)}P_{\mathcal{M}_{d}}(j\ |\ x_{<j})

(2)

where Path​(xt,v)\text{Path}(x_{t},v) represents the path from the root node xtx_{t} to the leaf node vv, x<jx_{<j} denotes all prefix tokens of jj. Then it uses another top-KK operation based on the path score to select KK leaf nodes as parent nodes of next layer. Finally, the generation ends at depth DD, and the tree 𝒯\mathcal{T} will be pruned to meet a global budget NN.
While structured, this rigid “expand-then-shrink” mechanism often generates redundant nodes that are discarded during intermediate shrinking or final pruning, leading to suboptimal resource allocation.

## 
4 Motivated Experiments

Figure 4: Real-World Mean Accepted Tokens (MAT) distribution across different queries in Eagle. The results exhibit high volatility: even within the same task category (e.g., Math or Coding), the optimal generation length fluctuates significantly.

To empirically investigate the limitations of static tree-based speculative decoding, we conduct a pilot analysis using Eagle Li et al. (2025b) as a representative baseline. We employ a fixed tree topology with width K=10K=10 and depth D=8D=8 (official settings in Eagle paper) with Qwen3-8B on MT-Bench (Zheng et al., 2023). We also conduct additional motivated experiments in Appendix C with Llama-series models and various datasets to demonstrate the generality of our motivation.

## 
4.1 The "Acceptance Funnel" Phenomenon

We first investigate the acceptance frequency of each position in the draft tree. Detailed model configurations are provided in Table 2. Figure 3 visualizes the acceptance frequency of tokens at each position within the static tree structure. Two key observations emerge regarding the distribution of effective speculation:

Diminishing Returns of Width in Deep Layers.
As illustrated in Figure 3, the acceptance distribution exhibits a funnel-like pattern. In the first initial layer (d=0d=0), the acceptance probability is relatively uniform across the top-KK candidates. This suggests that due to the initial stochasticity and minor distributional divergence between the draft model ℳd\mathcal{M}_{d} and the target model ℳt\mathcal{M}_{t}, a wider search breadth is necessary to capture the valid continuation.
However, as the tree deepens (d≥1d\geq 1), the acceptance mass concentrates sharply on the high confidence regions (e.g. top-1 and top-2 candidates).
We attribute this to the cumulative error amplification in auto-regressive drafting: for deep nodes, the draft model is either (1) confidently aligned with the target model (correct prediction), or (2) hallucinating a divergent path where even the top-KK candidates fail to recover the target distribution. Consequently, maintaining a static width K=10K=10 at deeper layers yields negligible marginal utility. When the draft model is aligned, a much smaller KK can find the correct draft tokens. When the draft model is highly uncertain, there needs a larger KK to ensure the coverage and stops the generation to avoid wasteful draft model forward.

## 
4.2 Variance in Real-World Accepted Length

While Section 4.1 reveals the structural redundancy within fixed width, we further investigate the weakness of fixed depth. We plot the distribution of Mean Accepted Tokens (MAT) for various queries in Figure 4. (Same settings with Figure 3)

The Static Depth Dilemma.
As shown in Figure 4, the accepted tokens fluctuate drastically even within the same task category (e.g., Math or Coding), exposing the limitation of a fixed-depth drafting policy. In low-entropy contexts where the draft model is well-aligned with the target, the potential acceptance length often exceeds the pre-defined depth DD. However, a static fixed depth DD prevents ℳd\mathcal{M}_{d} from generating more draft tokens for maximum speedup. Conversely, in high-entropy scenarios where prediction becomes ambiguous, the draft model tends to hallucinate deep branches that are destined for rejection. A fixed depth DD forces the allocation of computational resources to these invalid tokens, incurring latency overhead with zero marginal utility.

## 
5 Method

We introduce Talon, a budget-driven framework that optimizes draft tree construction by dynamically allocating resources based on real-time model confidence.

Figure 5: 
(a) Talon ’s Budget-Driven Tree. Talon dynamically allocates the token budget based on confidence. It uses Top-KK at the root for robustness and confidence gating at deeper layers, resulting in adaptive topologies—deep chains for high-confidence tokens (e.g., the folded “algo-rithm” sequence) and wide branches for uncertain ones. The expansion stops when the global budget is met (indicated by the icon).
(b) Tree Attention Mask. The structural mask used by the target model to verify the adaptive tree in parallel. The verification process follows the standard token-tree verification protocol (see Appendix B for details).

## 
5.1 From Static Grids to Dynamic Budgets

To overcome the inefficiencies of the static tree structures analyzed in Section 4–specifically the misalignment between fixed topology and varying token difficulty–Talon shifts the constraint from shape to capacity. We define a global Token Budget (NN), representing the maximum number of nodes allowed in the draft tree 𝒯\mathcal{T}. The objective is to iteratively “invest” this budget to grow a draft tree that dynamically adapts–deep-and-narrow for deterministic contexts and shallow-and-wide for uncertain ones–thereby maximizing effective speculation length under a fixed computational cost.

## 
5.2 Hybrid Tree Expansion Strategy

Talon constructs the tree layer-by-layer using a hybrid strategy: a robust initialization at the root (Layer 0) followed by confidence-gated expansion for subsequent layers (Layer d≥1d\geq 1).

## 
5.2.1 Confidence-Gated Expansion

For depths d≥1d\geq 1, we employ a global filtering mechanism to gate candidates based on their relative confidence. Let 𝒫d\mathcal{P}_{d} be the set of parent nodes. We define the Candidate Pool 𝒮d\mathcal{S}_{d} as the union of all child extensions:

𝒮d=⋃v∈𝒫d{(v,w)∣w∈𝒱}.\mathcal{S}_{d}=\bigcup_{v\in\mathcal{P}_{d}}\{(v,w)\mid w\in\mathcal{V}\}.

(3)

Each candidate u=(v,w)∈𝒮du=(v,w)\in\mathcal{S}_{d} is assigned a cumulative path probability p​(u)=p​(v)⋅Pℳd​(w∣x≤v)p(u)=p(v)\cdot P_{\mathcal{M}_{d}}(w\mid x_{\leq v}).

Motivated by (Minh et al., 2025), we first identify the anchor confidence md=maxu∈𝒮d⁡p​(u)m_{d}=\max_{u\in\mathcal{S}_{d}}p(u). We then retain candidates whose confidence falls within a dynamic margin μ\mu of the anchor:

𝒫d+1={u∈𝒮d∣p​(u)≥μ⋅md},\mathcal{P}_{d+1}=\{u\in\mathcal{S}_{d}\mid p(u)\geq\mu\cdot m_{d}\},

(4)

where μ∈(0,1]\mu\in(0,1] is a hyperparameter. To strictly respect the budget, if |𝒫d+1||\mathcal{P}_{d+1}| exceeds the remaining budget N−|𝒯|N-|\mathcal{T}|, we retain only the top candidates with the highest path scores.

## Intuition.

This mechanism naturally aligns topology with entropy. In deterministic contexts (e.g., “The capital of France is Paris”), the anchor md≈1.0m_{d}\approx 1.0 imposes a strict threshold, automatically pruning branches to form a deep-and-narrow chain. In high-entropy contexts (e.g., “Quantum computing can…”), a lower mdm_{d} relaxes the threshold, admitting diverse candidates to form a shallow-and-wide layer that prioritizes coverage.

## 
5.2.2 Robust Tree Initialization

While confidence gating effectively allocates budget at depth, applying it at the root (d=0d=0) compromises robustness due to draft model over-confidence.
Small draft models often exhibit imperfect calibration, assigning near-certain probability (e.g., >0.99>0.99) to incorrect tokens in simple contexts. Relative gating would prematurely collapse the tree into a single wrong branch–an over-confidence trap. Unlike deeper errors, a root failure causes catastrophic rejection of the entire draft sequence.
To mitigate calibration errors, we employ a Standard Top-KK Initialization for the first layer. Regardless of the confidence distribution, we explicitly expand the top-KK tokens:

𝒫1={(xt,v)∣v∈Top-K(Pℳd(⋅|xt))}.\mathcal{P}_{1}=\{(x_{t},v)\mid v\in\text{Top-}K\left(P_{\mathcal{M}_{d}}(\cdot|x_{t})\right)\}.

(5)

This ensures the speculative tree covers diverse plausible directions at the most critical juncture before switching to budget-efficient gating.

## 
5.3 Evaluating Talon with Draft Efficiency

To rigorously quantify the cost-effectiveness of Talon, we analyze the wall-time speedup by decomposing it into quality and cost factors.

Let NpN_{p} denote the total number of forward passes executed by the target model (verification steps), NqN_{q} be the total forward passes executed by the draft model to generate a sequence and LL be the output length. We introduce a new metric, Draft Efficiency (δ\delta), defined as the ratio of these two quantities:

δ=NqNp.\delta=\frac{N_{q}}{N_{p}}.

(6)

Intuitively, δ\delta represents the speculation cost–the average number of draft steps the system “invests” to secure a single verification opportunity. By deriving the wall-time speedup RR as a function of the Mean Accepted Tokens (τ=L/Np\tau=L/N_{p}) and the relative model cost (cc), we obtain the following speedup formulation ( see derivation in Appendix D):

R=τ1+c⋅δ.R=\frac{\tau}{1+c\cdot\delta}.

(7)

With Equation 7, we can theoretically explain why Talon achieves superior performance. Approaches like Eagle enforce a fixed depth DD, effectively locking the draft cost to a constant δ=D+1\delta=D+1 regardless of the context complexity. This rigidity leads to dual inefficiencies.

In high-entropy contexts (hard cases), the acceptance reward τ\tau naturally drops. Eagle continues to pay the high fixed cost δ\delta, causing the speedup RR to degrade significantly. Talon addresses this by halting expansion early; the resulting reduction in “investment” (δ↓\delta\downarrow) compensates for the lower reward, thereby mitigating the performance drop. Conversely, in low-entropy contexts (easy cases), static methods artificially cap the reward τ\tau at depth DD. Talon allows the draft tree to extend far beyond this limit, significantly increasing τ\tau. While growing deeper also increases the draft cost δ\delta, this investment yields a positive net gain. Since the relative cost ratio is typically small (c≪1c\ll 1), (1+c​δ)(1+c\delta) grows much slower than τ\tau, ensuring that the overall wall-time speedup RR continues to improve as the tree deepens.

## 
6 Experiments

Table 1: Main Results on Six Benchmarks (Temperature=0). Comparison between Eagle-3 and our proposed Talon across various models. We report Mean Acceptance Tokens (MAT) and Wall-time Speedup relative to standard decoding. Bold numbers denote the best speedup performance.

Model
Method
Alpaca
GSM8K
HumanEval
MT-Bench
QA
CNN/DM

Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.

Vicuna-13B

Eagle-3
6.61
3.78×\times

6.74
3.87×\times

8.31
4.77×\times

7.12
4.05×\times

5.24
3.03×\times

6.93
3.43×\times

SD
2.19
1.30×\times

2.20
1.19×\times

2.86
1.47×\times

2.51
1.29×\times

1.97
1.20×\times

2.63
1.41×\times

Medusa
2.44
1.90×\times

2.63
2.05×\times

2.78
2.18×\times

2.58
2.01×\times

2.10
1.62×\times

2.09
1.56×\times

Hydra
3.51
2.40×\times

3.66
2.53×\times

3.87
2.67×\times

3.64
2.46×\times

2.88
1.95×\times

2.82
1.86×\times

OPT-Tree
6.56
3.71×\times

6.77
3.65×\times

8.21
4.35×\times

6.95
3.75×\times

5.22
3.10×\times

6.91
3.22×\times

Talon
6.36
3.94×\times
6.80
3.99×\times
9.48
5.16×\times
7.29
4.27×\times
5.03
3.26×\times
7.20
3.58×\times

DSL-8B

Eagle-3
5.64
3.23×\times

7.40
4.24×\times

6.70
3.85×\times

5.82
3.33×\times

5.02
2.86×\times

5.03
2.89×\times

Talon
5.18
3.46×\times
7.46
4.43×\times
6.28
3.96×\times
5.45
3.56×\times
4.66
3.12×\times
4.75
3.14×\times

Llama3-8B

Eagle-3
6.93
3.92×\times

6.42
3.63×\times

7.08
4.05×\times

6.38
3.67×\times

5.36
3.01×\times

5.46
3.06×\times

Talon
6.51
4.04×\times
6.24
3.81×\times
7.28
4.20×\times
6.22
3.85×\times
5.09
3.27×\times
5.28
3.30×\times

Qwen3-8B

Eagle-3
3.45
2.08×\times

3.92
2.37×\times

3.89
2.35×\times

3.64
2.20×\times

3.46
2.09×\times

3.27
1.95×\times

Talon
3.39
2.44×\times
3.85
2.67×\times
3.78
2.69×\times
3.60
2.57×\times
3.41
2.46×\times
3.20
2.30×\times

Qwen3-32B

Eagle-3
2.75
1.83×\times

3.36
2.22×\times

2.98
1.96×\times

2.98
1.90×\times

2.66
1.78×\times

2.57
1.57×\times

Talon
2.72
2.05×\times
3.30
2.38×\times
2.96
2.15×\times
2.94
2.10×\times
2.64
1.99×\times
2.54
1.72×\times

## 
6.1 Experimental Setup

## Tasks and Datasets.

We evaluate Talon on a comprehensive suite of LLM backbones, including Llama-3.1-8B-Instruct (Team, 2024), Qwen3 8B and 32B (QwenTeam et al., 2025), DeepSeek-R1-Distill-LLaMA-8B (DSL) (DeepSeek-AI et al., 2025), and Vicuna-13B (Zheng et al., 2023).
Following standard protocols, we conduct evaluations on six diverse benchmarks: MT-Bench (Zheng et al., 2023), Alpaca (Ding et al., 2023), GSM8K (Cobbe et al., 2021), HumanEval (Chen et al., 2021), CNN/DM (Nallapati et al., 2016), and QA (Kwiatkowski et al., 2019), which are widely used benchmarks for instruction, math reasoning, code generation, chat, QA and summarization.

## Implementation Details.

We evaluate Talon against the state-of-the-art tree-based method Eagle-3 (Li et al., 2025b). All experiments are conducted on a single NVIDIA H200 (140GB) GPU with batch size 1 by default.
The global token budget NN is set to 60 for both Talon and Eagle-3.
The threshold μ\mu in Talon is set to 0.03 for all experiments.
Detailed configurations (hyperparameters, hardware, baselines, models) are provided in Appendix E.1.
We use two metrics: mean accepted tokens (Mat) and wall-time speedup (Spd.).

## 
6.2 Main Results

We present the main comparison with Eagle-3 in Table 1.
We further evaluate Talon under stochastic sampling (T=1T=1) in Table 3, and provide detailed ablation studies on tree initialization and budget settings in Appendix E.4.

We observe the following:
(a) Universal Speedup. Talon consistently outperforms Eagle-3 across all 8 models and 6 datasets. Notably, it achieves up to 5.16×5.16\times speedup on HumanEval with Vicuna-13B and 2.30×2.30\times speedup on CNN/DM with Qwen3-8B, significantly surpassing the baseline’s 4.77×4.77\times and 1.95×1.95\times.
(b) Reasoning Adaptability. The performance gap is particularly pronounced in reasoning-intensive tasks. On GSM8K (Math) and HumanEval (Code), Talon achieves substantial gains (e.g., 2.67×2.67\times vs 2.37×2.37\times on Qwen3-8B GSM8K). This verifies that our budget-driven adaptive expansion effectively captures correct paths in low-entropy reasoning steps where static trees often under-explore.
(c) Robustness. As shown in Table 3, Talon maintains strong performance with T=1T=1. While static trees suffer from flattened distributions, Talon’s confidence-gated mechanism successfully adapts the tree topology, achieving 2.51×2.51\times speedup on Qwen3-8B (HumanEval) compared to Eagle-3’s 2.20×2.20\times.

Figure 6: Draft Efficiency (δ\delta) vs. Mean Accepted Tokens (τ\tau). The gray dashed line represents the Oracle baseline (τ=δ\tau=\delta). The orange shaded region highlights the computation waste of static methods (Eagle-3). Talon (blue line and shaded region) closely tracks the Oracle, minimizing waste by dynamically aligning draft cost with generation difficulty.

## 
6.3 Evaluating Talon with Draft Efficiency

We evaluate the draft efficiency (defined in Section 5.3) of Talon and Eagle and visualize the relationship between the overhead (δ\delta) and the benefit (τ\tau) in Figure 6. Static methods (orange line) pay a fixed high computational cost without considering context difficulty, resulting in sub-optimal speedup. In contrast, Talon (blue line) closely approaches the zero-waste Oracle line (τ=δ\tau=\delta). By dynamically shrinking the token trees in uncertain regions and expanding them in deterministic ones, Talon effectively decouples draft cost from tree depth. This confirms that our framework maximizes speedup by ensuring computational resources are only invested where they yield high acceptance utility.
We also provide more visualization results of different models in Appendix E.2.

## 
6.4 Ablation Study

Figure 7: Ablation study on the number of initial Top-KK layers (kk). k=0k=0 represents the pure adaptive strategy without robust initialization, while k≥1k\geq 1 indicates using static Top-KK expansion for the first kk layers. (a)-(b) Wall-time speedup on HumanEval and Alpaca shows that k=1k=1 (Talon’s default) achieves the highest throughput. (c)-(d) Radar charts across six benchmarks further confirm that k=1k=1 (orange line) yields the most robust performance, outperforming both the uninitialized (k=0k=0) and over-extended (k≥2k\geq 2) configurations.

## 
6.4.1 Ablating Robust Tree Initialization

Talon employs a hybrid strategy that begins with robust tree initialization for the first layer to ensure robustness, followed by confidence-gated expansion layers. However, a natural question arises, “Why is robust tree initialization necessary only for the first layer”? To answer this question, we conduct an ablation study that using robust tree initialization for the first kk layers. k=0k=0 corresponds to a pure confidence-gated strategy without robust tree initialization.

## Necessity of Robust Initialization (k=1​v​s.k=0k=1\ vs.\ k=0).

As shown in Fig. 7 (a) and (b), removing the robust tree initialization leads to a noticeable drop in generation speed (e.g., declining from 263.8 to 261.5 tok/s on Llama3-8B).This degradation validates our hypothesis that draft models often suffer from root layer over-confidence; without a forced Top-KK expansion at the first layer, the model risks falling into incorrect branches due to over-confidence, leading to early draft rejections.

## Overhead of additional Robust Initialization (k≥2k\geq 2).

Conversely, extending the robust initialization phase to deeper layers (k=2,3k=2,3) yields suboptimal returns. It delays the transition to the more efficient confidence-gating expansion, thereby increasing the computational overhead (δ\delta) without a proportional gain in acceptance.The radar charts in Fig. 7 (c) and (d) clearly illustrate that k=1k=1 (orange line) consistently outperforms other configurations, confirming that a single layer of robust initialization followed immediately by adaptive gating achieves the optimal speedup.

## 
6.4.2 Ablating Confidence-gated Expansion

To further investigate the mechanism behind Talon, we conduct two additional ablation studies on the influence of different threshold μ\mu and different budget NN in Appendix E.4. The results show that the threshold μ\mu in Talon serves as a trade-off hyperparameter between exploration and exploitation. A larger μ\mu leads to more deep-and-narrow draft trees, especially effective in reasoning tasks such as coding and math, while a smaller μ\mu is more appropriate for some creative tasks. Moreover, the results demonstrate that Talon is more flexible to different computation budget. For some resource-intensive scenarios, static methods waste more computation with fixed KK and DD. However, Talon seamlessly adapts to different computation budget by simply adjusting NN.

## 
6.5 Case Study

We also provide an real-world case of Talon generated draft trees in Appendix E.5. Additionally, we provide an in-depth runtime breakdown of the tree construction phase in Appendix E.6, verifying that the algorithmic overhead of Talon remains negligible even with large vocabularies.

## 
7 Conclusion

In this work, we presented Talon, a training-free framework that shifts speculative decoding from rigid geometric constraints to a flexible, budget-driven paradigm.
By employing a hybrid expansion strategy that combines robust initialization with confidence gating, Talon dynamically shapes the draft tree—evolving into deep-and-narrow chains for deterministic contexts or shallow-and-wide branches for uncertain ones.
Extensive evaluations across 5 LLMs and 6 benchmarks demonstrate that Talon consistently outperforms SOTA methods like Eagle-3, achieving up to 5.16×5.16\times speedup.

## 
8 Limitations

While Talon demonstrates significant speedups and adaptability across various benchmarks, we identify several limitations that present avenues for future research:

## Scalability to Large Batch Sizes.

Our current evaluation focuses on latency-critical scenarios with a batch size of 1, which is the primary use case for real-time interaction. In high-throughput scenarios with large batch sizes, the compute-bound nature of the GPU may saturate, and the overhead of maintaining diverse dynamic tree structures for each request in a batch could become non-trivial. The memory management for varying tree topologies across a batch also presents implementation challenges. Extending budget-driven adaptive speculation to large-batch serving systems remains an open engineering challenge.

## Hyperparameter Generalization.

While we found a fixed threshold (μ=0.03\mu=0.03) and budget (N=60N=60) to be robust across most tested models and datasets, optimal performance in highly specialized domains might require task-specific tuning. Developing an auto-tuning mechanism that adjusts μ\mu and NN on-the-fly based on acceptance history would be a valuable extension.

## 
9 Ethic Statements

## Inheritance of Model Behaviors.

Talon is an inference acceleration framework designed to speed up existing Large Language Models without modifying their weights. As a speculative decoding method, it aims to losslessly recover the distribution of the target model. Consequently, Talon inherits the ethical properties, biases, and potential safety risks of the underlying target LLM and draft model. It does not introduce new capabilities for generating harmful content, nor does it mitigate existing biases in the base models. Users should continue to apply standard safety guardrails and alignment techniques to the target models deployed with Talon.

## References

- 
Z. Ankner, R. Parthasarathy, A. Nrusimha, C. Rinard, J. Ragan-Kelley, and W. Brandon (2024)
Hydra: sequentially-dependent draft heads for medusa decoding.

External Links: 2402.05109,
Link

Cited by: §E.1.1,
Appendix F.

- 
Anthropic Team (2025)
System card: claude opus 4.5.

Note: https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdfSystem card, Anthropic

Cited by: §1.

- 
G. Bachmann, S. Anagnostidis, A. Pumarola, M. Georgopoulos, A. Sanakoyeu, Y. Du, E. Schönfeld, A. Thabet, and J. Kohler (2025)
Judge decoding: faster speculative sampling requires going beyond model alignment.

External Links: 2501.19309,
Link

Cited by: Appendix F.

- 
T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao (2024)
MEDUSA: simple llm inference acceleration framework with multiple decoding heads.

In Proceedings of the 41st International Conference on Machine Learning,

ICML’24.

Cited by: §E.1.1,
Appendix F,
Appendix F,
§1,
§2.

- 
C. Chen, S. Borgeaud, G. Irving, J. Lespiau, L. Sifre, and J. Jumper (2023)
Accelerating large language model decoding with speculative sampling.

External Links: 2302.01318,
Link

Cited by: §E.1.1,
Appendix F,
§1,
§2.

- 
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
Evaluating large language models trained on code.

External Links: 2107.03374,
Link

Cited by: §6.1.

- 
Z. Chen, A. May, R. Svirschevski, Y. Huang, M. Ryabinin, Z. Jia, and B. Chen (2025)
Sequoia: scalable, robust, and hardware-aware speculative decoding.

External Links: 2402.12374,
Link

Cited by: Appendix F,
§2.

- 
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman (2021)
Training verifiers to solve math word problems.

External Links: 2110.14168,
Link

Cited by: §6.1.

- 
DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang (2025)
DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning.

External Links: 2501.12948,
Link

Cited by: §1,
§6.1.

- 
N. Ding, Y. Chen, B. Xu, Y. Qin, Z. Zheng, S. Hu, Z. Liu, M. Sun, and B. Zhou (2023)
Enhancing chat language models by scaling high-quality instructional conversations.

External Links: 2305.14233,
Link

Cited by: §6.1.

- 
C. Du, J. Jiang, X. Yuanchen, J. Wu, S. Yu, Y. Li, S. Li, K. Xu, L. Nie, Z. Tu, and Y. You (2024)
GLIDE with a cape: a low-hassle method to accelerate speculative decoding.

In Proceedings of the 41st International Conference on Machine Learning,

ICML’24.

Cited by: Appendix F,
§1,
§2.

- 
Y. Fu, P. Bailis, I. Stoica, and H. Zhang (2024)
Break the sequential dependency of llm inference using lookahead decoding.

In Proceedings of the 41st International Conference on Machine Learning,

ICML’24.

Cited by: Appendix F.

- 
X. Gao, W. Xie, Y. Xiang, and F. Ji (2025)
Falcon: faster and parallel inference of large language models through enhanced semi-autoregressive drafting and custom-designed decoding tree.

In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence,

AAAI’25/IAAI’25/EAAI’25.

External Links: ISBN 978-1-57735-897-8,
Link,
Document

Cited by: §2.

- 
F. Gloeckle, B. Y. Idrissi, B. Rozière, D. Lopez-Paz, and G. Synnaeve (2024)
Better & faster large language models via multi-token prediction.

External Links: 2404.19737,
Link

Cited by: §1.

- 
Google DeepMind Team (2025)
Gemini 3 pro model card.

Note: https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdfModel card, Google DeepMind

Cited by: §1.

- 
Z. He, Z. Zhong, T. Cai, J. Lee, and D. He (2024)
REST: retrieval-based speculative decoding.

In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard (Eds.),

Mexico City, Mexico, pp. 1582–1595.

External Links: Link,
Document

Cited by: Appendix F.

- 
A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi (2020)
The curious case of neural text degeneration.

In International Conference on Learning Representations,

External Links: Link

Cited by: Figure 15,
§E.6.

- 
Y. Hu, K. Wang, X. Zhang, F. Zhang, C. Li, H. Chen, and J. Zhang (2024)
SAM decoding: speculative decoding via suffix automaton.

External Links: 2411.10666,
Link

Cited by: Appendix F.

- 
K. Huang, X. Guo, and M. Wang (2025)
SpecDec++: boosting speculative decoding via adaptive candidate lengths.

External Links: 2405.19715,
Link

Cited by: Appendix F,
§2.

- 
F. Huo, J. Tan, K. Zhang, X. Cai, and S. Sun (2025)
C2T: a classifier-based tree construction method in speculative decoding.

External Links: 2502.13652,
Link

Cited by: Appendix F,
§2.

- 
T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov (2019)
Natural questions: a benchmark for question answering research.

Transactions of the Association for Computational Linguistics 7, pp. 453–466.

External Links: ISSN 2307-387X,
Document,
Link,
https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00276/1923288/tacl_a_00276.pdf

Cited by: §6.1.

- 
Y. Leviathan, M. Kalman, and Y. Matias (2023)
Fast inference from transformers via speculative decoding.

In Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.),

Proceedings of Machine Learning Research, Vol. 202, pp. 19274–19286.

External Links: Link

Cited by: §E.1.1,
Appendix F,
§1,
§2.

- 
Y. Li, F. Wei, C. Zhang, and H. Zhang (2024)
EAGLE-2: faster inference of language models with dynamic draft trees.

In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y. Chen (Eds.),

Miami, Florida, USA, pp. 7421–7432.

External Links: Link,
Document

Cited by: Appendix A,
Appendix F,
§1.

- 
Y. Li, F. Wei, C. Zhang, and H. Zhang (2025a)
EAGLE-3: scaling up inference acceleration of large language models via training-time test.

External Links: 2503.01840,
Link

Cited by: §E.1.1,
Appendix F,
Figure 1.

- 
Y. Li, F. Wei, C. Zhang, and H. Zhang (2025b)
EAGLE: speculative sampling requires rethinking feature uncertainty.

External Links: 2401.15077,
Link

Cited by: Appendix B,
Appendix B,
Appendix F,
§1,
§1,
§2,
§4,
§6.1.

- 
J. Liu, Q. Wang, J. Wang, and X. Cai (2024)
Speculative decoding via early-exiting for faster llm inference with thompson sampling control mechanism.

External Links: 2406.03853,
Link

Cited by: Appendix F.

- 
T. Liu, Y. Li, Q. Lv, K. Liu, J. Zhu, W. Hu, and X. Sun (2025a)
PEARL: parallel speculative decoding with adaptive draft length.

In The Thirteenth International Conference on Learning Representations,

External Links: Link

Cited by: Appendix F,
§1,
§2.

- 
T. Liu, Q. Lv, H. Li, X. Gao, and X. Sun (2025b)
LogitSpec: accelerating retrieval-based speculative decoding via next next token speculation.

External Links: 2507.01449,
Link

Cited by: Appendix F.

- 
X. Luo, Y. Wang, Q. Zhu, Z. Zhang, X. Zhang, Q. Yang, D. Xu, and W. Che (2024)
Turning trash into treasure: accelerating inference of large language models with token recycling.

External Links: 2408.08696,
Link

Cited by: Appendix F.

- 
J. Mamou, O. Pereg, D. Korat, M. Berchansky, N. Timor, M. Wasserblat, and R. Schwartz (2024)
Dynamic speculation lookahead accelerates speculative decoding of large language models.

External Links: 2405.04304,
Link

Cited by: Appendix F.

- 
X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, C. Shi, Z. Chen, D. Arfeen, R. Abhyankar, and Z. Jia (2024)
SpecInfer: accelerating large language model serving with tree-based speculative inference and verification.

In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3,

ASPLOS ’24, New York, NY, USA, pp. 932–949.

External Links: ISBN 9798400703867,
Link,
Document

Cited by: Appendix B,
Appendix F,
Appendix F,
§1,
§2.

- 
N. N. Minh, A. Baker, C. Neo, A. G. Roush, A. Kirsch, and R. Shwartz-Ziv (2025)
Turning up the heat: min-p sampling for creative and coherent LLM outputs.

In The Thirteenth International Conference on Learning Representations,

External Links: Link

Cited by: §5.2.1.

- 
R. Nallapati, B. Zhou, C. dos Santos, Ç. Gu˙\dot{}lçehre, and B. Xiang (2016)
Abstractive text summarization using sequence-to-sequence RNNs and beyond.

In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, S. Riezler and Y. Goldberg (Eds.),

Berlin, Germany, pp. 280–290.

External Links: Link,
Document

Cited by: §6.1.

- 
NVIDIA, P. Vingelmann, and F. H.P. Fitzek (2020)
CUDA, release: 10.2.89.

External Links: Link

Cited by: §E.1.2.

- 
OpenAI Team (2025)
GPT-5 system card.

Note: https://cdn.openai.com/gpt-5-system-card.pdfSystem card, OpenAI

Cited by: §1.

- 
A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019)
PyTorch: an imperative style, high-performance deep learning library.

External Links: 1912.01703,
Link

Cited by: §E.1.2.

- 
R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, and J. Dean (2022)
Efficiently scaling transformer inference.

External Links: 2211.05102,
Link

Cited by: §1.

- 
QwenTeam, A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu (2025)
Qwen3 technical report.

External Links: 2505.09388,
Link

Cited by: §1,
§6.1.

- 
A. Saxena (2023)
Prompt lookup decoding.

External Links: Link

Cited by: Appendix F.

- 
Z. Sun, U. Mendlovic, Y. Leviathan, A. Aharoni, J. H. Ro, A. Beirami, and A. T. Suresh (2025)
Block verification accelerates speculative decoding.

In The Thirteenth International Conference on Learning Representations,

External Links: Link

Cited by: Appendix F,
§2.

- 
G. Team, A. Zeng, X. Lv, Q. Zheng, Z. Hou, B. Chen, C. Xie, C. Wang, D. Yin, H. Zeng, J. Zhang, K. Wang, L. Zhong, M. Liu, R. Lu, S. Cao, X. Zhang, X. Huang, Y. Wei, Y. Cheng, Y. An, Y. Niu, Y. Wen, Y. Bai, Z. Du, Z. Wang, Z. Zhu, B. Zhang, B. Wen, B. Wu, B. Xu, C. Huang, C. Zhao, C. Cai, C. Yu, C. Li, C. Ge, C. Huang, C. Zhang, C. Xu, C. Zhu, C. Li, C. Yin, D. Lin, D. Yang, D. Jiang, D. Ai, E. Zhu, F. Wang, G. Pan, G. Wang, H. Sun, H. Li, H. Li, H. Hu, H. Zhang, H. Peng, H. Tai, H. Zhang, H. Wang, H. Yang, H. Liu, H. Zhao, H. Liu, H. Yan, H. Liu, H. Chen, J. Li, J. Zhao, J. Ren, J. Jiao, J. Zhao, J. Yan, J. Wang, J. Gui, J. Zhao, J. Liu, J. Li, J. Li, J. Lu, J. Wang, J. Yuan, J. Li, J. Du, J. Du, J. Liu, J. Zhi, J. Gao, K. Wang, L. Yang, L. Xu, L. Fan, L. Wu, L. Ding, L. Wang, M. Zhang, M. Li, M. Xu, M. Zhao, M. Zhai, P. Du, Q. Dong, S. Lei, S. Tu, S. Yang, S. Lu, S. Li, S. Li, Shuang-Li, S. Yang, S. Yi, T. Yu, W. Tian, W. Wang, W. Yu, W. L. Tam, W. Liang, W. Liu, X. Wang, X. Jia, X. Gu, X. Ling, X. Wang, X. Fan, X. Pan, X. Zhang, X. Zhang, X. Fu, X. Zhang, Y. Xu, Y. Wu, Y. Lu, Y. Wang, Y. Zhou, Y. Pan, Y. Zhang, Y. Wang, Y. Li, Y. Su, Y. Geng, Y. Zhu, Y. Yang, Y. Li, Y. Wu, Y. Li, Y. Liu, Y. Wang, Y. Li, Y. Zhang, Z. Liu, Z. Yang, Z. Zhou, Z. Qiao, Z. Feng, Z. Liu, Z. Zhang, Z. Wang, Z. Yao, Z. Wang, Z. Liu, Z. Chai, Z. Li, Z. Zhao, W. Chen, J. Zhai, B. Xu, M. Huang, H. Wang, J. Li, Y. Dong, and J. Tang (2025)
GLM-4.5: agentic, reasoning, and coding (arc) foundation models.

External Links: 2508.06471,
Link

Cited by: §1.

- 
L. Team (2024)
The llama 3 herd of models.

External Links: 2407.21783,
Link

Cited by: §6.1.

- 
J. Wang, Y. Su, J. Li, Q. Xia, Z. Ye, X. Duan, Z. Wang, and M. Zhang (2024)
OPT-tree: speculative decoding with adaptive draft tree structure.

External Links: 2406.17276,
Link

Cited by: §E.1.1,
Appendix F,
§2.

- 
Y. Weng, D. Mei, H. Qiu, X. Chen, L. Liu, J. Tian, and Z. Shi (2025)
CORAL: learning consistent representations across multi-step training with lighter speculative drafter.

External Links: 2502.16880,
Link

Cited by: Appendix F.

- 
T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush (2020)
Transformers: state-of-the-art natural language processing.

In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations,

Online, pp. 38–45.

External Links: Link

Cited by: §E.1.2.

- 
B. Xiao, C. Shi, X. Nie, F. Yang, X. Deng, L. Su, W. Chen, and B. Cui (2024)
Clover: regressive lightweight speculative decoding with sequential knowledge.

External Links: 2405.00263,
Link

Cited by: Appendix F.

- 
Y. Xiong, R. Zhang, Y. Li, T. Wu, and L. Zou (2024)
DySpec: faster speculative decoding with dynamic token tree structure.

External Links: 2410.11744,
Link

Cited by: §2.

- 
J. Yao, K. Chen, K. Zhang, J. You, B. Yuan, Z. Wang, and T. Lin (2025)
DeFT: decoding with flash tree-attention for efficient tree-structured llm inference.

External Links: 2404.00242,
Link

Cited by: Appendix F,
§1.

- 
Z. Zeng, J. Yu, Q. Pang, Z. Wang, H. Zhuang, H. Shao, and X. Zou (2024)
Chimera: a lossless decoding method for accelerating large language models inference by fusing all tokens.

External Links: 2402.15758,
Link

Cited by: Appendix F.

- 
J. Zhang, J. Wang, H. Li, L. Shou, K. Chen, G. Chen, and S. Mehrotra (2024a)
Draft & verify: lossless large language model acceleration via self-speculative decoding.

In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.),

Bangkok, Thailand, pp. 11263–11282.

External Links: Link,
Document

Cited by: Appendix F,
§2.

- 
L. Zhang, X. Wang, Y. Huang, and R. Xu (2025)
Learning harmonized representations for speculative sampling.

In The Thirteenth International Conference on Learning Representations,

External Links: Link

Cited by: Appendix F.

- 
S. Zhang, H. Wang, D. Ma, Z. Zhu, L. Chen, K. Lan, and K. Yu (2024b)
AdaEAGLE: optimizing speculative decoding via explicit modeling of adaptive draft structures.

External Links: 2412.18910,
Link

Cited by: Appendix F,
§2.

- 
L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica (2023)
Judging LLM-as-a-judge with MT-bench and chatbot arena.

In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track,

External Links: Link

Cited by: §4,
§6.1.

## 
Appendix A Formalized Algorithms

Algorithm 1 Static Tree Construction (Eagle)

0: Draft Model ℳd\mathcal{M}_{d}, Prefix x≤tx_{\leq t}, Depth DD, Width KK, Budget NN

1: 𝒯←{xt}\mathcal{T}\leftarrow\{x_{t}\}, 𝒫0←{xt}\mathcal{P}_{0}\leftarrow\{x_{t}\}, p​(xt)←1.0p(x_{t})\leftarrow 1.0

2: Fixed depth and width for-loop

3: for d=0d=0 to DD do

4:  𝒮d←∅\mathcal{S}_{d}\leftarrow\emptyset

5:  ⊳\triangleright Expand to K×KK\times K child nodes

6:  for v∈𝒫dv\in\mathcal{P}_{d} do

7:   Cv←Top-K(Pℳd(⋅∣x≤v))C_{v}\leftarrow\text{Top-}K(P_{\mathcal{M}_{d}}(\cdot\mid x_{\leq v}))

8:   p​(u)←p​(v)⋅Pℳd​(u∣x≤v),∀u∈Cvp(u)\leftarrow p(v)\cdot P_{\mathcal{M}_{d}}(u\mid x_{\leq v}),\forall u\in C_{v}

9:   𝒮d←𝒮d∪Cv\mathcal{S}_{d}\leftarrow\mathcal{S}_{d}\cup C_{v}

10:  end for

11:  ⊳\triangleright Shrink to KK next-layer parent nodes

12:  𝒫d+1←Top-​K​(𝒮d​ by ​p​(u))\mathcal{P}_{d+1}\leftarrow\text{Top-}K(\mathcal{S}_{d}\text{ by }p(u))

13:  𝒯←𝒯∪𝒫d+1\mathcal{T}\leftarrow\mathcal{T}\cup\mathcal{P}_{d+1}

14: end for

15: ⊳\triangleright Final pruning to budget NN

16: 𝒯←Top-​N​(𝒯​ by ​p​(u))\mathcal{T}\leftarrow\text{Top-}N(\mathcal{T}\text{ by }p(u))

17: return 𝒯\mathcal{T}

Algorithm 2 Adaptive Tree Construction (Talon)

0: Draft Model ℳd\mathcal{M}_{d}, Prefix x≤tx_{\leq t}, Budget NN, Width KK, Threshold μ\mu

1: 𝒯←{xt}\mathcal{T}\leftarrow\{x_{t}\}, 𝒫0←{xt}\mathcal{P}_{0}\leftarrow\{x_{t}\}, p​(xt)←1.0p(x_{t})\leftarrow 1.0

2: d←0d\leftarrow 0

3: ⊳\triangleright Budget-driven adaptive tree expansion

4: while |𝒯|<N|\mathcal{T}|<N do

5:  𝒮d←∅\mathcal{S}_{d}\leftarrow\emptyset

6:  ⊳\triangleright Gather next-layer candidate set 𝒮d\mathcal{S}_{d}

7:  for v∈𝒫dv\in\mathcal{P}_{d} do

8:   Cv←{(v,w)∣w∈𝒱}C_{v}\leftarrow\{(v,w)\mid w\in\mathcal{V}\}

9:   p​(u)←p​(v)⋅Pℳd​(w∣x≤v),∀u=(v,w)∈Cvp(u)\leftarrow p(v)\cdot P_{\mathcal{M}_{d}}(w\mid x_{\leq v}),\forall u=(v,w)\in C_{v}

10:   𝒮d←𝒮d∪Cv\mathcal{S}_{d}\leftarrow\mathcal{S}_{d}\cup C_{v}

11:  end for

12:  ⊳\triangleright Hybrid Expansion Strategy

13:  if d=0d=0 then

14:   ⊳\triangleright Robust Init

15:   𝒫d+1←Top-​K​(𝒮d​ by ​p​(u))\mathcal{P}_{d+1}\leftarrow\text{Top-}K(\mathcal{S}_{d}\text{ by }p(u))

16:  else

17:   ⊳\triangleright Confidence Gating

18:   md←maxu∈𝒮d⁡p​(u)m_{d}\leftarrow\max_{u\in\mathcal{S}_{d}}p(u)

19:   𝒫d+1←{u∈𝒮d∣p​(u)≥μ⋅md}\mathcal{P}_{d+1}\leftarrow\{u\in\mathcal{S}_{d}\mid p(u)\geq\mu\cdot m_{d}\}

20:  end if

21:  ⊳\triangleright Budget Check

22:  if |𝒯|+|𝒫d+1|>N|\mathcal{T}|+|\mathcal{P}_{d+1}|>N then

23:   𝒫d+1←Top-​(N−|𝒯|)​(𝒫d+1​ by ​p​(u))\mathcal{P}_{d+1}\leftarrow\text{Top-}(N-|\mathcal{T}|)(\mathcal{P}_{d+1}\text{ by }p(u))

24:  end if

25:  𝒯←𝒯∪𝒫d+1\mathcal{T}\leftarrow\mathcal{T}\cup\mathcal{P}_{d+1}

26:  d←d+1d\leftarrow d+1

27: end while

28: return 𝒯\mathcal{T}

Figure 8: Examples of Eagle-style draft token trees. At each layer, Eagle uses a top-KK operation to select KK nodes as next layer parents. In this way, Eagle can dynamically adjust the edges between last-layer parents and current-layer children. However, they cannot adjust KK to adapt for contexts, which leads to significant resource wastes.

In this section, we provide the formalized algorithms for both the static baseline and our proposed framework to illustrate the structural differences in their expansion strategies.

Algorithm 1 outlines the static tree construction strategy employed by state-of-the-art methods such as Eagle. This approach relies on rigid geometric constraints defined by a fixed depth DD and width KK. The construction proceeds layer-by-layer up to depth DD. At each step dd, the draft model expands K×KK\times K child nodes from the parent set 𝒫d\mathcal{P}_{d} and subsequently applies a Top-K operation to shrink the candidates back to size KK for the next layer. Finally, the tree is pruned globally to meet the token budget NN. This "expand-then-shrink" mechanism enforces a static topology regardless of generation difficulty, often generating redundant nodes that are discarded during intermediate steps. Note that “static” here means that each layer of the draft tree has a fixed number of nodes regardless of the context difficulty. The “dynamic” claimed in (Li et al., 2024) means that the draft model dynamically select top-KK nodes as next-layer parents, but it cannot adjust KK to adapt for context. Figure 8 shows 4 different Eagle-style draft trees with static tree topology.

Algorithm 2 presents the details of Talon, our proposed budget-driven adaptive framework. Unlike static methods, Talon is constrained only by a global node budget NN and constructs the tree iteratively until this capacity is filled. The core of Algorithm 2 is the Hybrid Expansion Strategy, which dynamically selects the expansion logic based on the current depth. For the root layer (d=0d=0), Talon employs a Robust Tree Initialization via a standard Top-K expansion to mitigate potential mis-calibration in the draft model’s initial prediction. For all subsequent layers (d≥1d\geq 1), the algorithm switches to a Confidence-Gated Expansion mechanism. It calculates an anchor confidence mdm_{d} within the candidate pool and filters nodes using a relative threshold μ\mu (i.e., p​(u)≥μ⋅mdp(u)\geq\mu\cdot m_{d}). This "prune-while-expanding" approach allows the tree topology to adapt between "deep-and-narrow" for deterministic contexts and "shallow-and-wide" for uncertain ones.

## 
Appendix B Token-Tree Verification

The verification phase of our framework adheres to the standard paradigm of tree-based speculative decoding, widely adopted in the field (Li et al., 2025b; Miao et al., 2024). By utilizing the Tree Attention mechanism, the target LLM verifies the entire draft tree in a single forward pass. A structured attention mask ensures that each token within the tree attends only to its predecessors along the root-to-leaf path, effectively simulating independent parallel verifications for all candidate branches while maintaining causal consistency.

Following the methodology established in Li et al. (2025b), the verification proceeds in three key steps: (1) calculating the posterior probability of each token in the tree given the target model’s output; (2) selecting the optimal valid prefix that satisfies the acceptance criteria; and (3) resampling a correction token from the residual distribution at the point of divergence. This rigorous process guarantees that the final output distribution is mathematically identical to that of the target model. We refer readers to the original Eagle paper (Li et al., 2025b) for detailed algorithms and proofs.

## 
Appendix C Additional Motivated Experiments

(a) Llama-3.1-Instruct-8B

(b) Qwen3-32B

(c) Vicuna-13B

(d) DeepSeek-R1-Distill-LLaMA-8B

Figure 9: Real-World Mean Accepted Tokens (MAT) distribution across different queries with EAGLE baseline. We visualize the MAT fluctuations on (a) Llama-3.1-8B, (b) Qwen3-32B, (c) Vicuna-13B, and (d) DeepSeek-R1-Distill-8B. Across varying parameter scales and architectures, we consistently observe significant volatility in generation difficulty: low-entropy tasks (e.g., Math/Coding) often allow for high acceptance, while high-entropy tasks (e.g., Roleplay) necessitate shallow trees. This universal variance highlights the limitation of static fixed-depth policies and underscores the necessity of Talon’s adaptive strategy.

To demonstrate the generality of the limitations observed in Section 3, specifically the "Acceptance Funnel" phenomenon and the "Static Depth Dilemma," we conduct additional empirical analyses on a broader range of LLMs. These experiments confirm that the inefficiencies of static tree structures are not model-specific artifacts but inherent challenges in speculative decoding.

## 
C.1 Heat Map Visualization of LlaMA-3.1-Instruct-8B

In Section 4, we utilized Qwen3-8B to illustrate the funnel-like distribution of accepted tokens. To verify whether this pattern holds across different architectures, we perform the same visualization on Llama-3.1-8B-Instruct. As shown in Figure 10, the results exhibit a striking similarity to our previous findings. The acceptance probability in the initial layer is relatively dispersed, necessitating a robust search width to capture correct continuations. However, as the tree deepens, the acceptance mass concentrates sharply on the high-confidence regions. This consistent "Acceptance Funnel" across models further justifies Talon’s hybrid expansion strategy: enforcing robustness at the root while employing confidence-gated pruning at deeper layers to eliminate the redundancy of static width.

Figure 10: Visualization of token acceptance frequency with Llama-3-8B using a static draft tree (K=10,D=8K=10,D=8). Consistent with Figure 3, the heatmap demonstrates the “Acceptance Funnel” phenomenon: acceptance is dispersed in the first layer but rapidly concentrates on top-ranked candidates in deeper layers, rendering wide static expansion inefficient.

## 
C.2 Dynamic Accepted Length of Various Models

We further extend the analysis of the "Static Depth Dilemma" to evaluate how generation difficulty fluctuates across different models and tasks. Figure 9 presents the Mean Accepted Tokens (MAT) distribution for Llama-3.1-8B-Instruct, Qwen3-32B-Instruct, Vicuna-13B, and DeepSeek-R1-Distill-LLaMA-8B, respectively.

Across all evaluated models and diverse task categories, we observe significant volatility in the effective speculation length. In deterministic, logic-constrained tasks such as Math and Coding, the MAT frequently reaches high peaks. In these low-entropy scenarios, the draft model is often confident and accurate, yet a pre-defined static depth limit artificially caps the potential speculation length, preventing the system from fully exploiting the easy context for maximum speedup. Conversely, in open-ended or high-entropy tasks like Roleplay or Writing, the acceptance length often drops significantly due to higher uncertainty. Here, a rigid static tree forces the draft model to hallucinate deep branches that are destined for rejection, resulting in wasted computation. This universal variance underscores that a "one-size-fits-all" static tree structure is fundamentally suboptimal, highlighting the necessity of Talon’s budget-driven adaptive mechanism that dynamically adjusts tree depth based on real-time confidence.

## 
Appendix D Derivation of Draft Efficiency

In this section, we provide the detailed derivation of the wall-time speedup formula presented in Section 5.3. We define the wall-clock time for a single forward pass of the draft model and the target model as TqT_{q} and TpT_{p}, respectively. Let LL denote the total number of tokens in the final generated output sequence. Throughout the generation process, the draft model executes a total of NqN_{q} forward passes, while the target model executes NpN_{p} verification steps.

In standard Auto-Regressive (AR) decoding, the target model generates tokens sequentially, resulting in a total inference latency:

TA​R=L⋅Tp,T_{AR}=L\cdot T_{p},

(8)

In contrast, Speculative Decoding (SD) decouples the process into drafting and verification phases. The total latency is the sum of time spent on both phases, expressed as:

TS​D=Np⋅Tp+Nq⋅Tq.T_{SD}=N_{p}\cdot T_{p}+N_{q}\cdot T_{q}.

(9)

The wall-time speedup RR is strictly defined as the ratio of the baseline latency to the speculative latency:

R=TA​RTS​D=L⋅TpNp⋅Tp+Nq⋅Tq.R=\frac{T_{AR}}{T_{SD}}=\frac{L\cdot T_{p}}{N_{p}\cdot T_{p}+N_{q}\cdot T_{q}}.

(10)

To relate this speedup to algorithmic efficiency, we simplify the fraction by dividing both the numerator and the denominator by Np⋅TpN_{p}\cdot T_{p}. We introduce three key metrics: (1) the relative model cost cc:

c=TqTp,c=\frac{T_{q}}{T_{p}},

(11)

(2) the Mean Accepted Tokens τ\tau:

τ=LNp,\tau=\frac{L}{N_{p}},

(12)

and (3) the Draft Efficiency δ\delta

δ=NqNp,\delta=\frac{N_{q}}{N_{p}},

(13)

which quantifies the computation invested in drafting per verification step. Substituting these variables into Equation 10 yields:

R=LNp1+NqNp⋅TqTp=τ1+c⋅δ.R=\frac{\frac{L}{N_{p}}}{1+\frac{N_{q}}{N_{p}}\cdot\frac{T_{q}}{T_{p}}}=\frac{\tau}{1+c\cdot\delta}.

(14)

This relationship highlights the advantage of Talon. Static methods enforce a fixed depth DD, locking draft efficiency to a constant δ=D+1\delta=D+1. Consequently, when the acceptance length τ\tau drops in difficult contexts, the speedup RR degrades significantly. In contrast, Talon dynamically adjusts the tree size, reducing δ\delta in uncertain scenarios. By ensuring the draft cost δ\delta scales down alongside τ\tau, Talon preserves a robust speedup RR across varying generation difficulties.

## 
Appendix E Details of Experiments

## 
E.1 Implementation Details

## 
E.1.1 Comparison Baselines

To strictly evaluate the effectiveness of Talon, we compare it against a comprehensive set of competitive baselines, categorizing them into chain-based, tree-based and other MLP-based speculative decoding approaches. We first include standard Speculative Decoding (SD) (Leviathan et al., 2023; Chen et al., 2023) as the fundamental chain-based baseline to measure the raw speedup gain over auto-regressive decoding. For MLP-based methods, which aim to reuse target hidden states with MLP to predict multiple draft tokens, we compare against Medusa (Cai et al., 2024) and Hydra (Ankner et al., 2024). These methods employ lightweight MLP decoding heads to predict subsequent tokens in parallel, representing a distinct direction in efficiently drafting.

Target Model
Method
Draft Model Checkpoint (Hugging Face)

Vicuna-1.3-13B
Medusa
FasterDecoding/medusa-vicuna-13b-v1.3

Hydra
ankner/hydra-vicuna-13b-v1.3

SD (Standard)
double7/vicuna-68m

OPT-Tree / Eagle-3 / Talon

yuhuili/EAGLE3-Vicuna1.3-13B

DSL-8B

Eagle-3 / Talon

yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B

Llama-3.1-8B-Instruct

Eagle-3 / Talon

yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

Qwen3-8B

Eagle-3 / Talon

AngelSlim/Qwen3-8B_eagle3

Qwen3-32B

Eagle-3 / Talon

AngelSlim/Qwen3-32B_eagle3

Table 2: 
List of draft model checkpoints used in our experiments.
Talon shares the identical draft weights with Eagle-3 across all tested benchmarks to ensure a fair evaluation of the tree topology efficiency.

In the realm of tree-based speculative decoding, we select Eagle-3 (Li et al., 2025a) as the primary state-of-the-art baseline. Eagle-3 utilizes feature-level auto-regression with a multi-layer fusion mechanism and typically constructs a static draft tree with fixed width and depth. Comparing against Eagle-3 allows us to directly demonstrate the advantages of our adaptive topology over rigid geometric constraints.

Most importantly, we include OPT-Tree (Wang et al., 2024) as a key baseline, as it represents the related work most closely aligned with Talon. Similar to our approach, OPT-Tree aims to optimize the draft tree topology. However, a critical distinction lies in the construction paradigm: OPT-Tree typically relies on search-based heuristics or a “generate-then-prune” strategy, which can introduce non-trivial computational overhead during inference. In contrast, Talon adopts a training-free, budget-driven expansion strategy that dynamically shapes the tree on the fly based on real-time confidence, thereby achieving adaptivity without the latency costs associated with complex search algorithms.

(a) Llama-3.1-Instruct-8B

(b) Qwen3-32B

(c) Vicuna-13B

(d) Qwen3-8B

Figure 11: Detailed visualization of Draft Efficiency (δ\delta) versus Mean Accepted Tokens (τ\tau) across four different LLMs. The x-axis represents the mean accepted length, while the y-axis represents the computational cost (draft steps per verification). The gray dashed line denotes the optimal Oracle baseline (τ=δ\tau=\delta) where no computation is wasted. While the static Eagle-3 baseline (orange) maintains a fixed high cost regardless of difficulty, TALON (blue) dynamically adjusts its draft budget, closely tracking the Oracle curve and significantly reducing computational waste in high-entropy (low τ\tau) scenarios and increasing acceptance reward in low-entropy (high τ\tau) scenarios.

## 
E.1.2 Hardware and Software Configurations

## Hardware Environment.

All experiments are conducted on a server equipped with a single NVIDIA H200 (141GB) GPU (NVIDIA et al., 2020). We perform all evaluations with a batch size of 1 to follow the standard settings and simulate real-world latency-critical inference scenarios.

## Software Environment.

Our implementation is based on PyTorch (Paszke et al., 2019) version 2.6.0+cu124 and Transformers (Wolf et al., 2020) version 4.57.1. The code is compiled with CUDA 12.8. For SD (Vanilla Speculative Decoding), we use Transformers’ official assisted decoding with their default settings. For Medusa and Hydra, we use their official codebases and adhere to their recommended environment settings to ensure fair comparison. For OPT-Tree, as its official codebase only supports Eagle-2, we re-implement OPT-Tree to use official Eagle-3 draft model checkpoint.

## Generation Configuration.

Unless otherwise specified, we employ greedy decoding (Temperature T=0T=0) for the main speedup benchmarks reported in Table 1. For the robustness experiments under stochastic sampling (Table 3), we set the temperature to T=1.0T=1.0 (No additional top-k or top-p operation). The maximum generation length is set to 1024 tokens for standard benchmarks. Regarding Talon’s hyper-parameters, we set the global token budget N=60N=60 and the confidence threshold μ=0.03\mu=0.03 by default across all models, K=10K=10 for robust tree initialization. For Eagle-3, we set the tree width K=10K=10 and D=8D=8, which is reported as optimal values in their manuscripts.

Figure 12: Ablation study on the global token budget (NN). We compare the wall-time generation speed of Talon against the Eagle-3 baseline across varying budget constraints ranging from N=32N=32 to N=96N=96. The results demonstrate that Talon consistently outperforms the static baseline across all budget levels. The performance gap is particularly pronounced in resource-constrained settings (e.g., N=32N=32), confirming that Talon’s confidence-gated expansion strategy is significantly more efficient at prioritizing high-value draft tokens than the rigid "expand-then-shrink" mechanism of static methods.

## 
E.1.3 Draft Model Selection

To ensure full reproducibility and a strictly fair comparison, we list the exact Hugging Face model checkpoints used for all draft models in our experiments.
Crucially, for Talon, we utilize the exact same draft model checkpoints as the state-of-the-art baseline EAGLE-3 (and OPT-Tree where applicable). This ensures that any observed performance gains are attributed solely to our adaptive tree expansion algorithm rather than superior draft weights.
The detailed configurations are provided in Table 2.

## 
E.2 More Visualization Results of Draft Efficiency

To comprehensively validate the theoretical efficiency analysis presented in Section 5.3, we provide extended visualizations of the relationship between Draft Efficiency (δ\delta) and Mean Accepted Tokens (τ\tau) across varying model architectures. Figure 11 illustrates this correlation for Llama-3.1-8B-Instruct, Qwen3-32B, Vicuna-13B, and Qwen3-8B. In these plots, the x-axis represents the acceptance length (how many tokens are actually accepted within each decoding step), while the y-axis represents the computational overhead draft efficiency (the ratio of draft steps to verification steps). The gray dashed line serves as the Oracle baseline (τ=δ\tau=\delta), representing an ideal zero-waste scenario where every drafted token is accepted.

As clearly demonstrated across all four sub-figures, the static baseline (Eagle-3) exhibits a rigid, horizontal trajectory. This pattern confirms that static tree-based methods incur a constant computational overhead determined solely by their fixed geometric hyperparameters (width KK and depth DD), regardless of the actual generation difficulty. Consequently, in high-entropy scenarios where the acceptance length τ\tau is low, static methods suffer from a substantial "efficiency gap"—highlighted by the extensive orange shaded regions—indicating that the system is squandering computational resources on generating draft branches that are destined to be rejected.

In sharp contrast, TALON demonstrates a highly adaptive behavior, with its efficiency curve strictly tracking the Oracle baseline across the entire spectrum of generation difficulties. The upward-sloping blue trajectory indicates that TALON dynamically modulates its resource investment: it autonomously reduces the draft budget in uncertain contexts to minimize waste, while scaling up the tree size in deterministic contexts to maximize the acceptance length. This tight alignment between the draft cost (δ\delta) and the acceptance reward (τ\tau) empirically verifies that our confidence-gated expansion strategy effectively decouples the draft structure from rigid constraints, ensuring that computational resources are allocated only when they yield a positive marginal utility. (Notably, the curve of Qwen3-8B and Qwen3-32B is relatively far from oracle line, suggesting that there is still a room to increase Talon’s end-to-end speedup. We will show in following sections that decreasing Talon’s threshold from default 0.03 to smaller threshold 0.01 yields better performance.)

Figure 13: Sensitivity analysis of the confidence threshold μ\mu. This parameter controls the trade-off between exploration width and generation depth. The results indicate that the optimal μ\mu is positively correlated with the model’s draft-target alignment (MAT). For models with lower alignment like Qwen3-8B (a), a lower threshold (μ=0.01\mu=0.01) yields the best performance by encouraging a "shallow-and-wide" search to ensure coverage. In contrast, for highly aligned models like DSL-8B (c), a higher threshold (μ=0.04\mu=0.04) is preferred to form "deep-and-narrow" chains that maximize the speculation length.

Table 3: Main Results on Six Benchmarks (Temperature=1). Comparison between Eagle-3 and our proposed Talon across various models. We report Mean Acceptance Tokens (MAT) and Wall-time Speedup relative to standard decoding. Bold numbers denote the best speedup performance.

Model
Method
Alpaca
GSM8K
HumanEval
MT-Bench
QA
CNN/DM

Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.
Mat
Spd.

Vicuna-13B

Eagle-3
5.61
3.15×\times

5.95
3.29×\times

6.77
3.78×\times

5.79
3.21×\times

4.72
2.64×\times

6.01
2.96×\times

SD
1.84
1.11×\times

1.84
1.08×\times

2.27
1.29×\times

2.04
1.13×\times

1.71
1.05×\times

2.21
1.23×\times

Medusa
2.83
2.19×\times

2.82
2.20×\times

2.97
2.35×\times

2.84
2.22×\times

2.49
1.92×\times

2.26
1.69×\times

Hydra
4.23
2.86×\times

3.98
2.73×\times

4.18
2.91×\times

4.06
2.77×\times

3.31
2.25×\times

3.15
2.07×\times

OPT-Tree
5.56
3.21×\times

5.97
3.25×\times

6.47
3.53×\times

5.69
3.16×\times

4.86
2.77×\times

5.84
2.79×\times

Talon
5.43
3.28×\times
5.82
3.35×\times
7.25
3.97×\times
5.88
3.38×\times
4.63
2.82×\times
5.95
2.96×\times

DSL-8B

Eagle-3
4.48
2.49×\times

6.43
3.58×\times

5.47
3.04×\times

4.56
2.55×\times

4.04
2.25×\times

4.32
2.39×\times

Talon
4.29
2.75×\times
6.44
3.70×\times
5.30
3.25×\times
4.44
2.81×\times
3.91
2.54×\times
4.17
2.63×\times

Llama3-8B

Eagle-3
5.19
2.85×\times

4.59
2.51×\times

6.24
3.46×\times

3.96
2.20×\times

3.12
1.72×\times

4.44
2.42×\times

Talon
5.21
3.15×\times
4.74
2.84×\times
6.40
3.56×\times
4.13
2.58×\times
3.20
2.08×\times
4.33
2.62×\times

Qwen3-8B

Eagle-3
3.32
1.95×\times

3.81
2.24×\times

3.77
2.20×\times

3.46
2.02×\times

3.17
1.87×\times

3.17
1.85×\times

Talon
3.28
2.25×\times
3.74
2.50×\times
3.70
2.51×\times
3.41
2.32×\times
3.15
2.17×\times
3.11
2.11×\times

Qwen3-32B

Eagle-3
2.55
1.67×\times

3.22
2.10×\times

2.95
1.89×\times

2.74
1.75×\times

2.44
1.61×\times

2.46
1.51×\times

Talon
2.53
1.84×\times
3.17
2.27×\times
2.91
2.07×\times
2.71
1.92×\times
2.41
1.77×\times
2.43
1.64×\times

## 
E.3 More Evaluation Results under Temperature Settings

While the primary evaluations in the main text focus on greedy decoding (Temperature T=0T=0), real-world LLM applications—particularly creative writing and open-ended chat—frequently rely on stochastic sampling to induce diversity in the generated content. To rigorously assess the robustness of our framework in these non-deterministic scenarios, we conduct a comprehensive evaluation using standard sampling with Temperature T=1T=1. The detailed comparative results between Talon and the state-of-the-art baseline Eagle-3 are reported in Table 3.

As evidenced by the quantitative results, Talon consistently outperforms Eagle-3 across all evaluated models and benchmarks, demonstrating superior adaptability to stochastic environments. A fundamental challenge in high-temperature settings is that the target model’s probability distribution becomes flatter, reducing the dominance of the top-1 token and increasing the likelihood of selecting lower-ranked candidates. Static approaches like Eagle-3, which enforce a fixed width and depth, often struggle in this regime because they rigidly expand the top-KK candidates regardless of the entropy. This leads to a scenario where the draft tree either misses the sampled token due to insufficient coverage in uncertain branches or wastes computation on high-confidence paths that are not selected.

In contrast, Talon’s confidence-gated expansion strategy naturally excels under these conditions. By determining the tree topology based on relative probability thresholds rather than a fixed node count, TALON dynamically widens the search space in high-entropy layers to capture the dispersed probability mass, while keeping the tree narrow in clearer contexts. This flexibility is reflected in the substantial speedup gains; for instance, on the Vicuna-13B model for HumanEval, Talon achieves a wall-time speedup of 3.97×\times, significantly surpassing the 3.53×\times speedup of Eagle-3. Similarly, on Qwen3-8B, Talon maintains a distinct advantage in both reasoning-intensive tasks like GSM8K and open-ended tasks like Alpaca. These findings confirm that Talon’s budget-driven mechanism is not limited to deterministic acceleration but is a robust solution for diverse generation settings, effectively mitigating the efficiency degradation often observed in stochastic speculative decoding.

## 
E.4 More Ablation Studies of Talon

In this section, we conduct in-depth ablation studies to evaluate the sensitivity and robustness of Talon with respect to its two key hyper-parameters: the global token budget (NN) and the confidence threshold (μ\mu).

x Draft token (accepted)

x Target token (rejected & resampled)

Eagle-3 (Tokens: 197 | Acc: 88.8%)

Talon (Tokens: 197 | Acc: 92.4%)

<s>
Here
’
s
the
completed
code
:

‘‘‘
python

def
next
_
small
est
(
lst
):

"""

You
are
given
a
list
of
integers
.

Write
a
function
next
_
small
est
()
that
...
 Return None if there is no such element.

 next_smallest([1, 2, 3, 4, 5]) == 2

<s>
Here
’
s
the
completed
code
:

‘‘‘
python

def
next
_
small
est
(
lst
):

"""

You
are
given
a
list
of
integers
.

Write
a
function
next
_
small
est
()
that
...
 Return None if there is no such element.

 next_smallest([1, 2, 3, 4, 5]) == 2

Figure 14: Comparison of token generation traces. Left: Eagle-3 shows frequent interruptions (pink) due to verification failures. Right: Talon maintains longer accepted chains (cyan) by utilizing confidence-aware adaptive trees.

## Impact of Global Token Budget (NN).

We first investigate how the inference performance scales with the available computational budget. Figure 12 compares the wall-time speedup of Talon against the Eagle-3 baseline across varying node budgets ranging from 32 to 96. For a fair comparison, we apply the same global budget constraint to Eagle-3 using its standard pruning mechanism. The results demonstrate that Talon consistently outperforms the static baseline across all budget levels. Notably, in resource-constrained scenarios (e.g., N=32N=32), TALON exhibits a significant advantage. This indicates that our confidence-gated expansion is highly efficient at prioritizing the most promising candidate tokens, ensuring that a limited budget is invested in high-probability paths rather than being diluted by a fixed-width expansion. As the budget increases to 96, Talon continues to scale effectively, utilizing the additional capacity to extend generation depth in deterministic regions, whereas static methods often hit a performance plateau due to their rigid structural constraints.

## Sensitivity to Confidence Threshold (μ\mu).

Next, we analyze the influence of the confidence threshold μ\mu on generation speed, as visualized in Figure 13. The threshold μ\mu acts as a gatekeeper that balances the trade-off between exploration width and generation depth. Interestingly, we observe that the optimal μ\mu is positively correlated with the degree of alignment between the draft and target models, which can be approximated by the Mean Accepted Tokens (MAT).

For models exhibiting high alignment and high MAT, such as DeepSeek-R1-Distill-LLaMA-8B (DSL-8B), the optimal threshold tends to be relatively high (peaking around μ=0.04\mu=0.04). In these well-aligned scenarios, the draft model’s confidence is a reliable proxy for correctness; thus, a stricter threshold effectively filters out unlikely branches, shaping the tree into a "deep-and-narrow" structure that maximizes draft length. Conversely, when the draft-target alignment is weaker (resulting in lower MAT), a lower threshold becomes more advantageous. This phenomenon explains the behavior of Qwen3-8B in Figure 13, where the performance peaks at a lower threshold of μ=0.01\mu=0.01. Here, the draft model is less certain, necessitating a more lenient threshold to encourage "shallow-and-wide" exploration, thereby ensuring sufficient coverage of the target distribution to prevent early rejection.

## 
E.5 More Case Studies of Talon

To qualitatively analyze the behavior of Talon, we visualize the decoding traces of a code generation task in Figure 14. Code generation typically represents a low-entropy regime, where the next tokens (e.g., syntax keywords, standard indentations) can be predicted with high confidence.

As shown in the left panel, Eagle employs a relatively static expansion strategy. Even when the model is confident, Eagle is limited by its fixed tree structure, resulting in frequent interruptions (pink tokens) and forcing the model to resample from the target model. This limits the Mean Accepted Tokens (MAT) per step.

In contrast, Talon (right panel) dynamically leverages its confidence-gated expansion mechanism. Upon detecting the low-entropy nature of the current context, Talon adaptively allocates the token budget to extend the tree depth rather than width. This allows the draft model to speculate deep-and-narrow draft tokens. Consequently, Talon significantly reduces the frequency of verification calls and achieves a higher MAT, demonstrating the efficiency of adaptive token trees in deterministic generation tasks.

## 
E.6 Tree Construction Overhead

To rigorously quantify the algorithmic efficiency of our proposed method, we conduct a micro-benchmark focusing specifically on the tree construction overhead. We simulate the next-token probability distributions using a Zipfian distribution P​(r)∝1/rαP(r)\propto 1/r^{\alpha} (Holtzman et al., 2020), covering diverse generation scenarios ranging from high-entropy creative writing tasks (α=0.7\alpha=0.7) and standard natural language (α=1.35\alpha=1.35) to highly deterministic code generation (α=5.0\alpha=5.0). The warmup step is 20, and we run each tree expansion strategy 100 times and report their mean latency. In this controlled environment, we compare the single-layer expansion latency of Eagle against Talon. Eagle relies on a rigid dual Top-K mechanism that performs sorting operations twice per layer—once for child selection and again for parent ranking—which scales poorly with large vocabulary sizes. In contrast, Talon utilizes a single confidence-gated sampling operation, implemented via efficient element-wise masking and non-zero index retrieval, which does not involving ranking operation.

As illustrated in Figure 15, Talon consistently reduces the tree construction latency across varying vocabulary sizes (32​K32K to 152​K152K), achieving speedups ranging from 1.18×\times to 1.44×\times. The advantage is particularly pronounced in deterministic settings (α=5.0\alpha=5.0), where our adaptive mechanism naturally sparsifies the candidate set, thereby minimizing memory access overhead. However, it is important to acknowledge that this sampling and tree construction latency constitutes a negligible fraction (<5%<5\%) of the total end-to-end inference time, which remains dominated by the model’s forward passes. Consequently, Talon’s primary wall-time speedup derives from its superior Draft Efficiency (δ\delta) rather than this micro-optimization in sampling.

Figure 15: 
Runtime breakdown of the tree expansion overhead.
We benchmark the latency of a single-layer expansion step across varying vocabulary sizes (32​K32K, 128​K128K, 152​K152K) using Zipfian distributions (Holtzman et al., 2020) to simulate High Entropy (α=0.7\alpha=0.7), Natural Language (α=1.35\alpha=1.35), and Deterministic (α=5.0\alpha=5.0) contexts.
Talon consistently outperforms the static dual Top-K approach of Eagle, achieving up to 1.44×\times speedup in the construction phase by avoiding expensive sorting operations. Note that this step accounts for less than 5% of the total inference time.

## 
Appendix F More Discussions to Related Work

This appendix provides an extended discussion on the evolution of speculative decoding, tracing the field’s progression from chain-based verification to tree-structured verification. We highlight how varying approaches trade off draft quality, verification cost, and implementation complexity.

## Chain-based speculative decoding.

The classical formulation of speculative decoding, often referred to as speculative sampling, is a draft-and-verify procedure (Leviathan et al., 2023; Chen et al., 2023; Zhang et al., 2024a). Given a prefix, a fast drafter autoregressively proposes a length-KK continuation. The target model then verifies these KK positions in a single forward pass, accepting the longest matching prefix and resampling at the first mismatch. While attractive for its lossless nature, the speedup of this framework hinges strictly on the acceptance length and the computational cost ratio between the drafter and the target.

Improving the drafter.
A major line of work focuses on producing higher-quality drafts with minimal overhead to boost acceptance rates.
Medusa (Cai et al., 2024) introduces multiple lightweight decoding heads atop the target model to predict several next tokens in parallel by reusing target hidden states.
Because independent heads may ignore intra-draft dependencies, follow-up approaches like Hydra (Ankner et al., 2024) and Clover (Xiao et al., 2024) introduce sequential dependencies among heads to better approximate autoregressive drafting.
Chimera (Zeng et al., 2024) proposes a lightweight draft architecture combining short-range and full-context signals to enhance quality while maintaining speed.
Complementarily, GliDe with CaPE (Du et al., 2024) reuses the target KV cache to lower drafting overhead and employs confidence-guided proposal expansion to provide stronger candidates.

Controlling draft length and reducing wasted work.
Even with a strong drafter, a fixed speculation length KK can be suboptimal: large KK wastes computation when acceptance is low, while small KK caps potential speedup.
SpecDec++ (Huang et al., 2025) formulates the candidate length selection as an MDP, adaptively stopping drafting using an acceptance-prediction signal.
DISCO (Mamou et al., 2024) similarly predicts when to stop speculation by dynamically selecting the speculation length.
PEARL (Liu et al., 2025a) addresses the system-level bottleneck of mutual waiting between drafting and verification by overlapping phases and enabling segmented, adaptive draft lengths.
Additionally, Block Verification (Sun et al., 2025) has been proposed to accelerate the verification phase itself.

Addressing training–inference misalignment.
When a specialized drafter is trained, mismatches between training distributions and inference-time contexts can hurt acceptance.
HASS (Zhang et al., 2025) proposes harmonized objectives and context alignment to reduce these inconsistencies.
CORAL (Weng et al., 2025) further improves consistency via cross-step representation alignment and reduces the effective cost of the LM head.
Orthogonally, Judge Decoding (Bachmann et al., 2025) observes that many rejected tokens are plausible and proposes relaxing verification via a compact judging module.

## Tree-based speculative decoding.

Tree-based Speculative Decoding (SD) generalizes the draft chain into a draft tree, enabling the verifier to choose among multiple candidate branches. This significantly reduces the penalty of early mismatches. The key enabler is tree attention, which allows the target model to verify multiple paths in parallel (Miao et al., 2024; Cai et al., 2024).

Token-tree verification and structured drafting.
SpecInfer (Miao et al., 2024) is a representative early system that organizes drafter outputs into a token tree verified via tree-based attention.
EAGLE (Li et al., 2025b) advances this by performing autoregression at the feature level.
EAGLE-2 (Li et al., 2024) introduces a context-aware dynamic draft tree guided by drafter confidence, while EAGLE-3 (Li et al., 2025a) further improves drafting via multi-layer fusion and training-time techniques.

Training-free tree construction via retrieval.
To avoid training specialized drafters, some methods combine SD with retrieval.
REST (He et al., 2024) retrieves draft tokens from a datastore, while Prompt Lookup Decoding (Saxena, 2023) leverages n-gram matching within the current context.
Token Recycling (Luo et al., 2024) stores previously observed candidate-token transitions in a compact adjacency matrix to retrieve a draft tree.
SAM-Decoding (Hu et al., 2024) utilizes a suffix automaton to efficiently find the longest suffix match for drafting. LogitSpec (Liu et al., 2025b) proposes to use the last logit as a guidance to retrieve more matched and accurate draft tokens.
While plug-and-play, these methods are bounded by the availability of high-quality matches in the context or datastore.
Alternative draft-model-free approaches like Lookahead Decoding (Fu et al., 2024) generate parallel n-grams using Jacobi iteration, and EESD (Liu et al., 2024) uses early exiting from intermediate layers to generate drafts.

Tree-attention efficiency.
Since tree-based SD relies on non-trivial KV updates, system efficiency is critical. DeFT (Yao et al., 2025) proposes an IO-aware flash tree-attention algorithm tailored for tree-structured inference to improve verification throughput.

## Summary and positioning of Talon.

While tree-based SD has become a standard paradigm, existing approaches face distinct limitations in adaptability and efficiency.
Learning-based methods, such as C2T (Huo et al., 2025) and AdaEAGLE (Zhang et al., 2024b), require training specialized modules and remain constrained by rigid parameter spaces (e.g., predicting a fixed KK), failing to achieve fully elastic topology changes.
Optimization-based methods also exhibit drawbacks: Sequoia (Chen et al., 2025) relies on offline algorithms to find a globally optimal tree, resulting in a static template that cannot adapt to instance-wise difficulty.
Similarly, search-based strategies like OPT-Tree (Wang et al., 2024) often follow a “generate-then-prune” paradigm or employ complex search heuristics at inference time, which introduces non-trivial computational overhead.
Talon distinguishes itself through a training-free, budget-driven framework that operates via on-the-fly construction.
Analogous to “pre-pruning” in decision trees, Talon adopts a “prune-while-expanding” strategy: it iteratively allocates the node budget based on real-time confidence, naturally halting expansion in uncertain branches without generating wasteful nodes first.
This allows the draft tree to fluidly morph between “deep-and-narrow” and “shallow-and-wide” shapes, maximizing speculation utility with minimal construction cost.

## 
Appendix G LLM Usage

We used a large language model (LLM)–based writing assistant solely for grammar and wording improvements on draft text. The LLM did not generate research ideas, claims, proofs, algorithms, code, figures, or analyses, and it did not have access to any non-public data. All edits suggested by the LLM were manually reviewed and either accepted or rewritten by the authors, who take full responsibility for the final content. The LLM is not an author of this paper.

Generated on Mon Jan 12 09:25:03 2026 by LaTeXML