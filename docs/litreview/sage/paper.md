SAGE: Accelerating Vision-Language Models via Entropy-Guided Adaptive Speculative Decoding

1 Introduction

2 Preliminaries

2.1 Vision-Language Models

2.2 Speculative Decoding

3 Methodology

3.1 Entropy-Based Confidence Estimation

3.2 Adaptive Tree Structure Generation

Adaptive Depth.

Adaptive Width.

Hierarchical Width Decay.

Dynamic Tree Construction.

Temporal Correlation Exploitation.

3.3 Overall Inference Pipeline

Phase 1: Tree-based Draft Generation.

Phase 2: Parallel Verification.

Phase 3: Dynamic Tree Update.

Adaptive History Tracking.

Complexity Analysis.

4 Theoretical Analysis

4.1 Entropy and Acceptance Probability

4.2 Optimal Tree Configuration

Implications for Adaptive Strategy.

5 Experiments

5.1 Experimental Setup

5.2 Performance Evaluation on Dense Models

5.3 Performance Evaluation on MoE Models

5.4 Performance Evaluation on LLMs

5.5 Ablation Studies

6 Related Work

6.1 Efficient Inference for Vision-Language Models

6.2 Speculative Decoding

6.3 Adaptive Decoding Strategies

7 Conclusions

A Appendix

A.1 Initialized draft tree structure

A.2 Analysis of Computational Time Overhead

A.3 Detailed Theoretical Proofs

SAGE: Accelerating Vision-Language Models via Entropy-Guided Adaptive Speculative Decoding

Yujia Tong

Tian Zhang

Yunyang Wan

Kaiwei Lin

Jingling Yuan

Chuang Hu

Abstract

Speculative decoding has emerged as a promising approach to accelerate inference in vision-language models (VLMs) by enabling parallel verification of multiple draft tokens. However, existing methods rely on static tree structures that remain fixed throughout the decoding process, failing to adapt to the varying prediction difficulty across generation steps. This leads to suboptimal acceptance lengths and limited speedup. In this paper, we propose SAGE, a novel framework that dynamically adjusts the speculation tree structure based on real-time prediction uncertainty. Our key insight is that output entropy serves as a natural confidence indicator with strong temporal correlation across decoding steps. SAGE constructs deeper-narrower trees for high-confidence predictions to maximize speculation depth, and shallower-wider trees for uncertain predictions to diversify exploration. SAGE improves acceptance lengths and achieves faster acceleration compared to static tree baselines. Experiments on multiple benchmarks demonstrate the effectiveness of SAGE: without any loss in output quality, it delivers up to 3.36×3.36\times decoding speedup for LLaVA-OneVision-72B and 3.18×3.18\times for Qwen2.5-VL-72B.

Machine Learning, ICML

1 Introduction

Although vision-language models (VLMs) (Li et al., 2024a; Zhang et al., 2024) exhibit outstanding performance in multimodal tasks, their enormous parameter scale often leads to a sharp increase in computational overhead during the inference phase, including higher memory demands and longer response times. Due to the need to process both visual and textual information simultaneously, VLMs typically require more computational resources than single-modality models (Vasu et al., 2025), making the latency bottleneck in auto-regressive generation particularly prominent.

A promising approach to alleviating this bottleneck is Speculative Decoding  (Leviathan et al., 2023; Sun et al., 2023). The core idea of this approach lies in breaking the limitation of traditional auto-regressive models, which can only generate tokens one by one. Its working mechanism typically relies on an efficient small draft model and a powerful original target model. During the generation process, the small draft model rapidly and continuously predicts multiple candidate tokens, forming a draft sequence. Subsequently, the original target model verifies the entire draft sequence in parallel in a single step, retaining only the correct prefix. By leveraging the parallel computing capability of the target model, speculative decoding achieves a significant acceleration in inference while strictly ensuring that the output remains completely consistent with that of the original model (Xia et al., 2023; Kim et al., 2023).

However, most existing research on speculative decoding  (Leviathan et al., 2023; Sun et al., 2023) has primarily focused on large language models (LLMs), with relatively limited exploration in the domain of vision-language models. Recently, studies such as SpecVLM (Ji et al., 2025) have begun to address this gap by proposing speculative decoding frameworks specifically tailored for VLMs. SpecVLM introduces a training-free speculative decoding approach designed for Video LLMs, which combines token pruning with tree-based draft generation to accelerate inference.

Despite these advancements, a critical limitation persists in current speculative decoding approaches: the tree structure governing draft generation is typically defined statically prior to inference and remains fixed throughout the decoding process. This static configuration fails to account for the inherent variability in prediction difficulty across different generation steps. The model’s prediction confidence varies substantially across tokens. Deterministic elements such as domain-specific terminology produce concentrated probability distributions with low entropy and high confidence. Open-ended content and creative expressions, by contrast, yield dispersed probability distributions with high entropy and low confidence (Holtzman et al., 2020; Li et al., 2024c). A fixed tree structure cannot adapt to these varying conditions: when entropy is low, a narrow and shallow tree wastes the opportunity to speculate further ahead; conversely, when entropy is high, an overly deep tree leads to wasted computation on branches that are unlikely to be accepted. This mismatch between static tree configurations and dynamic prediction entropy results in suboptimal acceptance lengths and consequently limits the achievable speedup ratio.

To address this challenge, we propose SAGE, a novel framework that dynamically adjusts the tree structure based on the model’s real-time prediction uncertainty. Our key insight is that the entropy of the output probability distribution serves as a natural and computationally inexpensive indicator of prediction confidence. Specifically, at each decoding step, we compute the normalized entropy of the draft model’s output distribution and derive a confidence score inversely proportional to this entropy. This confidence score then guides the adaptive construction of the speculation tree: when confidence is high (low entropy), we construct deeper but narrower trees to capitalize on the high acceptance probability and speculate further into the future; when confidence is low (high entropy), we construct shallower but wider trees to explore more candidate branches while avoiding wasted computation on deep paths prone to rejection. Furthermore, we exploit the temporal correlation of entropy across consecutive decoding steps—empirically, adjacent tokens tend to exhibit similar uncertainty levels—allowing us to use the current step’s entropy to effectively optimize the next step’s tree structure with negligible computational overhead. Our approach requires no additional training and can be seamlessly integrated into existing speculative decoding frameworks for VLMs, achieving improved acceptance lengths and superior inference acceleration while maintaining output equivalence with the original model.

Our key contributions are as follows:

•

We reveal that the entropy of draft model outputs serves as a natural confidence indicator, which motivates our entropy-guided adaptive tree construction strategy.

•

We propose SAGE, which dynamically constructs deeper-narrower trees for high-confidence predictions and shallower-wider trees for uncertain ones based on real-time entropy estimation.

•

We provide theoretical analysis establishing the relationship between prediction entropy and token acceptance probability, and derive optimal tree depth/width configurations that justify our adaptive strategy.

•

Extensive experiments on multiple VLM benchmarks demonstrate that SAGE achieves higher acceptance lengths and superior speedup compared to static tree baselines while maintaining output equivalence.

2 Preliminaries

2.1 Vision-Language Models

Vision-language models (VLMs) adopt an encoder-decoder architecture. Given an input image II, the visual encoder extracts visual tokens 𝐙={z1,z2,…,zm}\mathbf{Z}=\{z_{1},z_{2},\ldots,z_{m}\}, which are projected into the language model’s embedding space. The language model generates output tokens auto-regressively:

P​(yt∣𝐙,𝐲<t)=softmax​(𝐖o​𝐡t),P(y_{t}\mid\mathbf{Z},\mathbf{y}_{<t})=\mathrm{softmax}(\mathbf{W}_{o}\mathbf{h}_{t}),

(1)

where 𝐡t\mathbf{h}_{t} denotes the hidden state at step tt and 𝐖o\mathbf{W}_{o} is the output projection matrix. The complete response 𝐘={y1,…,yn}\mathbf{Y}=\{y_{1},\ldots,y_{n}\} is generated as:

P​(𝐘∣𝐙)=∏t=1nP​(yt∣𝐙,𝐲<t).P(\mathbf{Y}\mid\mathbf{Z})=\prod_{t=1}^{n}P(y_{t}\mid\mathbf{Z},\mathbf{y}_{<t}).

(2)

Due to this auto-regressive nature, the decoding process suffers from significant latency, as each token must be generated sequentially with nn forward passes required to produce a response of length nn. This sequential dependency prevents parallelization during inference, making the decoding phase the primary computational bottleneck in VLMs, especially when generating long-form responses.

2.2 Speculative Decoding

Speculative decoding accelerates auto-regressive generation by using a smaller draft model ℳd\mathcal{M}_{d} to speculate γ\gamma candidate tokens, which are verified in parallel by the target model ℳt\mathcal{M}_{t}. Given candidates y^t:t+γ\hat{y}_{t:t+\gamma}, the target model computes:

Pℳt​(yi∣𝐙,𝐲<t,y^t:i−1),∀i∈{t,…,t+γ}.P_{\mathcal{M}_{t}}(y_{i}\mid\mathbf{Z},\mathbf{y}_{<t},\hat{y}_{t:i-1}),\quad\forall i\in\{t,\ldots,t+\gamma\}.

(3)

The acceptance length τ\tau is the longest prefix matching the target’s predictions, advancing τ+1\tau+1 tokens per iteration.

Tree-based Drafting. Instead of a single sequence, the draft model constructs a candidate tree 𝒯\mathcal{T}. At depth ll, top-kk tokens are selected:

Topk​(Pℳd​(y∣𝐙,𝐲<t,𝐩<l)),\mathrm{Top}_{k}\big(P_{\mathcal{M}_{d}}(y\mid\mathbf{Z},\mathbf{y}_{<t},\mathbf{p}_{<l})\big),

(4)

where 𝐩=[p1,…,pd]\mathbf{p}=[p_{1},\ldots,p_{d}] denotes the path indices. A tree attention mask 𝐌∈ℝ|𝒯|×|𝒯|\mathbf{M}\in\mathbb{R}^{|\mathcal{T}|\times|\mathcal{T}|} ensures each token only attends to its ancestors. The optimal path is:

𝐩∗=arg⁡max𝐩∈𝒯​∑l=1|𝐩|𝟙​[y^𝐩≤l=arg⁡max⁡Pℳt​(y∣⋅)].\mathbf{p}^{*}=\arg\max_{\mathbf{p}\in\mathcal{T}}\sum_{l=1}^{|\mathbf{p}|}\mathbbm{1}\big[\hat{y}_{\mathbf{p}_{\leq l}}=\arg\max P_{\mathcal{M}_{t}}(y\mid\cdot)\big].

(5)

The expected speedup ratio is:

S≈𝔼​[τ]+1cd⋅|𝒯|+ct,S\approx\frac{\mathbb{E}[\tau]+1}{c_{d}\cdot|\mathcal{T}|+c_{t}},

(6)

where cdc_{d} and ctc_{t} denote the computational costs of the draft and target models, respectively.

Figure 1: Overview of SAGE. The framework consists of three phases:
(1) tree-based draft generation, (2) parallel verification by the target
model, and (3) entropy-guided dynamic tree update that computes a
confidence score from output entropy and adapts tree depth/width accordingly.

3 Methodology

In this section, we present our SAGE framework. We first introduce how to estimate prediction confidence using output entropy (§ 3.1), then describe how to adaptively construct tree structures based on this confidence (§ 3.2), and finally present the overall inference pipeline (§ 3.3).

3.1 Entropy-Based Confidence Estimation

The key insight of our approach is that the entropy of the output probability distribution serves as a natural indicator of prediction uncertainty. When the model is confident about its prediction, the probability mass concentrates on a few tokens, resulting in low entropy; conversely, high uncertainty leads to a more uniform distribution with high entropy (Holtzman et al., 2020; Li et al., 2024c).

Given the output logits 𝐨t∈ℝ|V|\mathbf{o}_{t}\in\mathbb{R}^{|V|} from the draft model at decoding step tt, we extract the top-kk probabilities and renormalize them:

P~i=Pi∑j=1kPj,∀i∈Topk.\tilde{P}_{i}=\frac{P_{i}}{\sum_{j=1}^{k}P_{j}},\quad\forall i\in\mathrm{Top}_{k}.

(7)

We then compute the Shannon entropy of this renormalized distribution:

H​(𝐏~)=−∑i=1kP~i​log⁡P~i.H(\tilde{\mathbf{P}})=-\sum_{i=1}^{k}\tilde{P}_{i}\log\tilde{P}_{i}.

(8)

The confidence score α∈[0,1]\alpha\in[0,1] is defined as the complement of the normalized entropy:

α=1−H​(𝐏~)log⁡k.\alpha=1-\frac{H(\tilde{\mathbf{P}})}{\log k}.

(9)

When the model is highly confident , the probability distribution is peaked on a single token; when completely uncertain, the distribution approaches uniform. This confidence score directly guides our adaptive tree construction strategy described in the next subsection.

3.2 Adaptive Tree Structure Generation

Based on the confidence score α\alpha, we dynamically adjust both the depth and width of the speculation tree. Our guiding principle is twofold. For high confidence scenarios (α→1\alpha\to 1), we construct deeper but narrower trees, since the draft model’s predictions are likely to be accepted and we can speculate further into the future while reducing unnecessary exploration of alternative branches. For low confidence scenarios (α→0\alpha\to 0), we construct shallower but wider trees, as it is more beneficial to explore diverse candidates at shallow depths rather than committing to deep paths that are unlikely to be accepted.

Adaptive Depth.

Given the confidence score α\alpha, we compute the target tree depth as a linear interpolation between minimum and maximum depths:

D​(α)=Dmin+α⋅(Dmax−Dmin),D(\alpha)=D_{\min}+\alpha\cdot(D_{\max}-D_{\min}),

(10)

where DminD_{\min} and DmaxD_{\max} are hyperparameters controlling the depth range.

Adaptive Width.

Conversely, the tree width is computed as:

W​(α)=Wmin+(1−α)⋅(Wmax−Wmin),W(\alpha)=W_{\min}+(1-\alpha)\cdot(W_{\max}-W_{\min}),

(11)

where WminW_{\min} and WmaxW_{\max} define the width range. This ensures that uncertain predictions trigger wider exploration at shallow levels.

Hierarchical Width Decay.

Beyond the first level, we apply a depth-dependent decay to the width at each subsequent level ll:

Wl=W​(α)⋅1l⋅(0.5+Pparent),W_{l}=W(\alpha)\cdot\frac{1}{l}\cdot(0.5+P_{\mathrm{parent}}),

(12)

where PparentP_{\mathrm{parent}} is the probability of the parent node. This design ensures that deeper levels have progressively fewer branches, avoiding exponential growth, and that higher-probability branches receive more exploration resources than lower-probability ones.

Dynamic Tree Construction.

Given the adaptive parameters, we construct the tree through a recursive expansion procedure starting from the root node. At each node, we first compute the local width WlW_{l} based on the current depth and parent probability, then select the top-WlW_{l} candidates from the draft model’s output distribution. For each candidate whose probability exceeds a depth-adaptive threshold θl=0.1⋅l/D​(α)\theta_{l}=0.1\cdot l/D(\alpha), we recursively expand its subtree. The expansion terminates when reaching either the maximum depth D​(α)D(\alpha) or the maximum node count NmaxN_{\max}.

The resulting tree 𝒯α\mathcal{T}_{\alpha} adapts its structure based on prediction confidence, containing more nodes along high-probability paths and fewer along uncertain ones. Formally, a tree path 𝐩=[p1,p2,…,pd]\mathbf{p}=[p_{1},p_{2},\ldots,p_{d}] is included in 𝒯α\mathcal{T}_{\alpha} if and only if for all levels l≤dl\leq d, the path index pl<Wlp_{l}<W_{l} and the cumulative probability P​(y^𝐩≤l)>θlP(\hat{y}_{\mathbf{p}_{\leq l}})>\theta_{l}.

Figure 2: Autocorrelation analysis of prediction entropy sequences on Qwen2.5-VL 7B. Left: VideoDetailedCaption dataset. Right: MVBench dataset. Lag-kk denotes the correlation between entropy values separated by kk decoding steps.

Temporal Correlation Exploitation.

An important empirical observation for long-form text generation in video understanding tasks is that prediction confidence exhibits strong temporal correlation—adjacent tokens in a sequence tend to have similar uncertainty levels. As shown in Figure 2, the Pearson correlation coefficients between entropy values at different lags remain consistently high across both VideoDetailedCaption and MVBench datasets (all p<0.05p<0.05), demonstrating strong temporal consistency of entropy. This allows us to use the entropy computed at step tt to optimize the tree structure for step t+1t+1 with negligible additional overhead.

3.3 Overall Inference Pipeline

Algorithm 1 presents the complete SAGE inference procedure. The process consists of three phases: initialization, iterative speculation-verification, and dynamic tree update.

Algorithm 1 The Overall Inference Pipeline of SAGE

0:  Target model ℳt\mathcal{M}_{t}, draft model ℳd\mathcal{M}_{d}, input (𝐙,𝐲<1)(\mathbf{Z},\mathbf{y}_{<1}), max generation tokens TT, Dynamic tree parameters: Dmin,Dmax,Wmin,Wmax,k,NmaxD_{\min},D_{\max},W_{\min},W_{\max},k,N_{\max}

1:  Initialize KV caches for ℳt\mathcal{M}_{t} and ℳd\mathcal{M}_{d}

2:  Initialize tree structure 𝒯\mathcal{T}

3:  Initialize confidence t←1t\leftarrow 1, α←0.5\alpha\leftarrow 0.5

4:  while t<Tt<T and not EOS do

5:  // Phase 1: Tree-based Draft Generation

6:  {y^𝐩}𝐩∈𝒯,𝐏out←ℳd​(yt−1,𝒯)\{\hat{y}_{\mathbf{p}}\}_{\mathbf{p}\in\mathcal{T}},\mathbf{P}_{\mathrm{out}}\leftarrow\mathcal{M}_{d}(y_{t-1},\mathcal{T})

7:  // Phase 2: Parallel Verification

8:  𝐌←\mathbf{M}\leftarrow TreeAttnMask(𝒯\mathcal{T})

9:  {Pℳt(⋅∣𝐩)}𝐩∈𝒯←ℳt({y^𝐩},𝐌)\{P_{\mathcal{M}_{t}}(\cdot\mid\mathbf{p})\}_{\mathbf{p}\in\mathcal{T}}\leftarrow\mathcal{M}_{t}(\{\hat{y}_{\mathbf{p}}\},\mathbf{M})

10:  𝐩∗,τ←\mathbf{p}^{*},\tau\leftarrow EvaluatePosterior(𝒯,Pℳt\mathcal{T},P_{\mathcal{M}_{t}})

11:  yt:t+τ←y^𝐩1:τ+1∗y_{t:t+\tau}\leftarrow\hat{y}_{\mathbf{p}^{*}_{1:\tau+1}}

12:  Update KV caches with accepted tokens

13:  t←t+τ+1t\leftarrow t+\tau+1

14:  // Phase 3: Dynamic Tree Update

15:  Compute Confidence α\alpha using 𝐏out\mathbf{P}_{\mathrm{out}} by Eq.(9)

16:  Update D←Dmin+α⋅(Dmax−Dmin)D\leftarrow D_{\min}+\alpha\cdot(D_{\max}-D_{\min}) by Eq.(10)

17:  Update W←Wmin+(1−α)⋅(Wmax−Wmin)W\leftarrow W_{\min}+(1-\alpha)\cdot(W_{\max}-W_{\min}) by Eq.(11)

18:  Update 𝒯\mathcal{T} using (𝐏out,D,W,Nmax\mathbf{P}_{\mathrm{out}},D,W,N_{\max})

19:  end while

20:  Return Generated sequence 𝐲1:t\mathbf{y}_{1:t}

Phase 1: Tree-based Draft Generation.

Given the current tree structure 𝒯\mathcal{T}, the draft model generates candidate tokens for all paths in the tree. The tree attention mask ensures proper causal attention within each path, allowing each token to only attend to its ancestors in the tree. We retain the output probability distribution 𝐏out\mathbf{P}_{\mathrm{out}} for entropy computation in the subsequent dynamic tree update phase.

Phase 2: Parallel Verification.

The target model verifies all candidate paths in a single forward pass using the tree attention mask. For each path in the tree, we compare the draft tokens against the target model’s greedy predictions at each position. The best path is selected as the one achieving the longest accepted prefix, where all draft tokens along the path match the corresponding greedy outputs from the target model. The accepted tokens are then appended to the generated sequence, and the key-value caches of both models are updated accordingly.

Phase 3: Dynamic Tree Update.

Using the retained probability distribution 𝐏out\mathbf{P}_{\mathrm{out}} from the draft generation phase, we compute the confidence score based on the normalized entropy as described in Eq.(9). This confidence score then determines the depth and width parameters for constructing the next tree structure through Eq.(10) and Eq.(11), enabling the speculation strategy to adapt to the current prediction uncertainty before the next iteration begins.

Adaptive History Tracking.

Beyond the entropy-based per-step adaptation, we introduce a feedback mechanism that adjusts the tree configuration based on historical acceptance performance. Specifically, we maintain a sliding window of recent acceptance lengths and compute their moving average. When the recent average acceptance length falls below a lower threshold, it indicates that the draft model’s predictions are frequently rejected by the target model, and we reduce the maximum depth to avoid wasting computation on deep speculation paths that are unlikely to be accepted. Conversely, when the recent average acceptance length exceeds an upper threshold, it suggests strong alignment between draft and target models, and we increase the maximum depth to capitalize on the high acceptance rate and speculate further into the future. When the average falls between these two thresholds, we maintain the current depth unchanged, providing hysteresis to prevent frequent oscillation between configurations. This history-based adaptation complements the entropy-guided strategy: while entropy captures instantaneous prediction uncertainty, the acceptance history reflects cumulative draft-target alignment quality over recent steps, enabling more robust adaptation to varying generation difficulty throughout the decoding.

Complexity Analysis.

The entropy computation adds 𝒪​(k)\mathcal{O}(k) operations per decoding step, which is negligible compared to the 𝒪​(|V|⋅d)\mathcal{O}(|V|\cdot d) complexity of the forward pass, where |V||V| denotes the vocabulary size and dd is the hidden dimension. The tree generation requires 𝒪​(Nmax)\mathcal{O}(N_{\max}) operations in the worst case. Since k,Nmax≪|V|⋅dk,N_{\max}\ll|V|\cdot d, our method introduces minimal overhead while enabling significant speedup through improved acceptance lengths. We provide a detailed time breakdown analysis in Appendix A.2.

4 Theoretical Analysis

In this section, we provide theoretical justifications for our entropy-guided adaptive tree construction. We establish the relationship between prediction entropy and token acceptance probability (§ 4.1), then analyze the optimal tree configuration (§ 4.2). Due to space constraints, detailed proofs are deferred to Appendix A.3.

4.1 Entropy and Acceptance Probability

We formalize the connection between output entropy and the probability that a draft token is accepted by the target model.

Definition 4.1 (Acceptance Event).

Let Pd(⋅|c)P_{d}(\cdot|c) and Pt(⋅|c)P_{t}(\cdot|c) denote the output distributions of the draft and target models conditioned on context cc, respectively. Under greedy decoding, a draft token y^=arg⁡maxy⁡Pd​(y|c)\hat{y}=\arg\max_{y}P_{d}(y|c) is accepted if and only if y^=arg⁡maxy⁡Pt​(y|c)\hat{y}=\arg\max_{y}P_{t}(y|c).

Assumption 4.2 (Bounded Distribution Divergence).

The draft and target models have bounded total variation distance:
DT​V(Pd(⋅|c),Pt(⋅|c))≤ϵD_{TV}(P_{d}(\cdot|c),P_{t}(\cdot|c))\leq\epsilon for some small ϵ≥0\epsilon\geq 0.

Lemma 4.3 (Confidence-Probability Relationship).

Let P~d\tilde{P}_{d} denote the renormalized top-kk distribution with probabilities p1≥p2≥⋯≥pkp_{1}\geq p_{2}\geq\cdots\geq p_{k}, and let α=1−H​(P~d)/log⁡k\alpha=1-H(\tilde{P}_{d})/\log k be the confidence score. Then:

p1≥1k+α⋅k−1k.p_{1}\geq\frac{1}{k}+\alpha\cdot\frac{k-1}{k}.

(13)

Theorem 4.4 (Acceptance Probability Lower Bound).

Under Assumption 4.2, acceptance is guaranteed when:

α>k−2+4​ϵ​k2​(k−1).\alpha>\frac{k-2+4\epsilon k}{2(k-1)}.

(14)

For k=10k=10 and ϵ=0.05\epsilon=0.05, this threshold is approximately 0.560.56.

This theorem establishes that high confidence (low entropy) provides a sufficient condition for token acceptance, justifying our use of entropy as a signal for tree construction.

4.2 Optimal Tree Configuration

We analyze the optimal tree depth and width to maximize speedup. We adopt a simplified model to derive qualitative insights that motivate our adaptive strategy.

Assumption 4.5 (Acceptance Probability Model).

The acceptance probability at depth ll is pl=p⋅γl−1p_{l}=p\cdot\gamma^{l-1} for base probability p∈(0,1]p\in(0,1] and decay factor γ∈(0,1]\gamma\in(0,1].

Theorem 4.6 (Expected Acceptance Length).

Under Assumption 4.5, for greedy-path speculation with depth DD:

𝔼​[τ]=∑l=1Dpl​γl​(l−1)/2.\mathbb{E}[\tau]=\sum_{l=1}^{D}p^{l}\gamma^{l(l-1)/2}.

(15)

Theorem 4.7 (Optimal Depth).

Under Assumption 4.5 with γ=1\gamma=1 and width Wl=1W_{l}=1, the optimal depth satisfies:

D∗=⌊log⁡(ct/cd)log⁡(1/p)−1⌋,D^{*}=\left\lfloor\frac{\log(c_{t}/c_{d})}{\log(1/p)}-1\right\rfloor,

(16)

where cd,ctc_{d},c_{t} are draft and target model costs. D∗D^{*} increases monotonically with pp.

Theorem 4.8 (Optimal Width).

For a depth-1 tree with acceptance probability qi=q1/iβq_{i}=q_{1}/i^{\beta}, the optimal width scales as:

W∗≈(q1cd)1/(1+β).W^{*}\approx\left(\frac{q_{1}}{c_{d}}\right)^{1/(1+\beta)}.

(17)

The relative benefit of large WW decreases as q1q_{1} increases.

Implications for Adaptive Strategy.

These theorems justify our adaptive tree construction:

•

High confidence (α→1\alpha\to 1): High acceptance probability pp leads to large optimal depth D∗D^{*}, while the marginal benefit of width decreases.

•

Low confidence (α→0\alpha\to 0): Low pp makes deep speculation inefficient, but wider exploration at shallow depths increases acceptance chances.

Our linear mappings D​(α)=Dmin+α⋅(Dmax−Dmin)D(\alpha)=D_{\min}+\alpha\cdot(D_{\max}-D_{\min}) and W​(α)=Wmin+(1−α)⋅(Wmax−Wmin)W(\alpha)=W_{\min}+(1-\alpha)\cdot(W_{\max}-W_{\min}) capture these relationships. While the theorems rely on simplified models, our empirical results in Section 5 confirm that this strategy consistently outperforms static configurations, validating the qualitative predictions of our analysis.

5 Experiments

Task
Method
TextVQA
GQA
ChartQA
SEED-Bench

τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup

Image

Vanilla AR
-
7.25
-
-
9.10
-
-
7.40
-
-
6.79
-

SD-Chain
4.01
15.86

2.19×\times

3.84
18.15

1.99×\times

4.21
16.60

2.24×\times

4.18
15.30

2.25×\times

SD-Tree
3.87
15.01

2.07×\times

4.05
19.39

2.13×\times

4.05
15.82

2.14×\times

3.80
14.01

2.06×\times

SpecVLM
3.93
16.48

2.27×\times

4.01
19.56

2.15×\times

3.84
16.42

2.22×\times

3.84
15.97

2.35×\times

SAGE\cellcolorgray!20

5.42\cellcolorgray!20

18.55\cellcolorgray!20

2.56×\times\cellcolorgray!20

\cellcolorgray!205.26

\cellcolorgray!2020.23

\cellcolorgray!202.22×\times

\cellcolorgray!205.60

\cellcolorgray!2018.95

\cellcolorgray!202.56×\times

\cellcolorgray!205.78

\cellcolorgray!2019.08

\cellcolorgray!202.81×\times

Task
Method
VideoDetailedCaption
MVBench
MVLU
LongVideoBench

τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup

Video

Vanilla AR
-
4.69
-
-
4.68
-
-
4.75
-
-
4.69
-

SD-Chain
3.10
8.26

1.76×\times

3.29
8.69

1.86×\times

3.97
9.99

2.10×\times

3.60
8.70

1.86×\times

SD-Tree
4.21
10.51

2.24×\times

3.50
9.19

1.96×\times

4.22
10.70

2.25×\times

3.91
9.90

2.11×\times

SpecVLM
4.11
13.62

2.90×\times

3.45
12.24

2.62×\times

4.07
14.07

2.96×\times

3.16
11.13

2.37×\times

SAGE\cellcolorgray!20

\cellcolorgray!205.74

\cellcolorgray!2015.75

\cellcolorgray!203.36×\times

\cellcolorgray!205.94

\cellcolorgray!2016.12

\cellcolorgray!203.44×\times

\cellcolorgray!205.19

\cellcolorgray!2015.68

\cellcolorgray!203.30×\times

\cellcolorgray!203.81

\cellcolorgray!2012.55

\cellcolorgray!202.68×\times

Table 1: Average accepted length τ\tau, decoding speed (tokens/s), and speedup of LLaVA-OneVision series (72B–8B) on image tasks TextVQA, GQA, ChartQA, and SEED-Bench, and on video tasks VideoDetailedCaption, MVBench, MVLU, and LongVideoBench. “Vanilla AR” refers to vanilla auto-regressive decoding, “SD-Chain” denotes speculative decoding with draft chains, and “SD-Tree” denotes speculative decoding with draft trees.

Setup
Method
VideoDetailedCaption
MVBench
LongVideoBench

τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup
τ\tau
Tokens/s
Speedup

Qwen2.5-VL

72B-7B

Vanilla AR
-
4.99
-
-
5.74
-
-
4.75
-

SD-Chain
3.67
10.38

2.08×\times

3.35
11.19

1.95×\times

3.10
8.67

1.83×\times

SD-Tree
4.18
11.28

2.26×\times

3.92
12.36

2.15×\times

3.70
9.72

2.05×\times

SpecVLM
3.90
13.48

2.70×\times

3.73
14.05

2.45×\times

3.27
11.51

2.42×\times

SAGE\cellcolorgray!20

\cellcolorgray!205.18

\cellcolorgray!2015.87

\cellcolorgray!203.18×\times

\cellcolorgray!204.53

\cellcolorgray!2015.52

\cellcolorgray!202.70×\times

\cellcolorgray!203.51

\cellcolorgray!2011.97

\cellcolorgray!202.52×\times

Table 2: Average accepted length τ\tau, decoding speed (tokens/s), and speedup of Qwen2.5-VL series on VideoDetailedCaption, MVBench, MVLU, and LongVideoBench. “Vanilla AR” refers to vanilla auto-regressive decoding, “SD-Chain” denotes speculative decoding with draft chains, and “SD-Tree” denotes speculative decoding with draft trees.

5.1 Experimental Setup

Benchmarks and Models. To verify the effectiveness of our method in accelerating the decoding phase, we evaluate it on both image and video understanding tasks that require generating long texts. For image understanding tasks, we use the TextVQA (Singh et al., 2019), GQA (Hudson and Manning, 2019), ChartQA (Masry et al., 2022) and SEED-Bench (Li et al., 2023a) dataset as the benchmark. For video understanding tasks, we select VideoDetailCaption (Ji et al., 2025), MVBench (Li et al., 2024b), MVLU (Zhou et al., 2024), and LongVideoBench (Wu et al., 2024) as benchmarks. We evaluate our method across various open-source VLM families, including LLaVA-OneVision series (Li et al., 2024a), Qwen2.5-VL series (Bai et al., 2025b) and Qwen3-VL series (Bai et al., 2025a).

Baselines. We compare SAGE against four methods: (1) Vanilla AR,
the standard auto-regressive decoding serving as the speedup baseline; (2) SD-Chain,
classic speculative decoding with a fixed-length linear draft sequence; (3) SD-Tree,
tree-based speculative decoding with a static predefined structure; and (4) SpecVLM (Ji et al., 2025),
a VLM-specific framework that combines visual token pruning with static tree drafting.

Evaluation Metrics. Since speculative decoding methods are all lossless, we focus on acceleration metrics. We select the following metrics to evaluate acceleration performance: (1) decoding speed, (2) speed-up ratio relative to vanilla auto-regressive decoding, and (3) average accepted length.

Implementation details. For the adaptive tree configuration, we set Dmax=8D_{\max}=8, Dmin=3D_{\min}=3, Wmax=10W_{\max}=10, Wmin=2W_{\min}=2, k=10k=10, and the maximum node count to 64. For adaptive history tracking, we use a sliding window of size 10 with lower and upper thresholds of 2 and 3 respectively. Following SpecVLM, we apply visual token pruning in the draft model with a pruning ratio of 90% for video tasks and 80% for image tasks. We use greedy decoding throughout. All experiments are conducted on a server equipped with NVIDIA H20 GPUs.

5.2 Performance Evaluation on Dense Models

Table 1 presents the performance comparison on the LLaVA-OneVision series with a 72B target model and an 8B draft model across four image benchmarks and four video benchmarks. On image tasks, SAGE consistently outperforms all baselines across all metrics. Specifically, SAGE achieves average acceptance lengths of 5.42, 5.26, 5.60, and 5.78 on TextVQA, GQA, ChartQA, and SEED-Bench respectively, representing improvements of 37.9%, 31.2%, 45.8%, and 50.5% over SpecVLM. The corresponding speedup ratios reach 2.56×\times, 2.22×\times, 2.56×\times, and 2.81×\times over vanilla auto-regressive decoding. On video tasks, the improvements are even more pronounced. SAGE achieves speedup ratios of 3.36×\times, 3.44×\times, 3.30×\times, and 2.68×\times on VideoDetailedCaption, MVBench, MVLU, and LongVideoBench respectively, substantially exceeding both SD-Tree and SpecVLM. Notably, on MVBench, SAGE attains an acceptance length of 5.94, which is 72.2% higher than SpecVLM’s 3.45, demonstrating the effectiveness of our entropy-guided adaptive tree construction in capturing prediction confidence and optimizing speculation depth accordingly.

To validate the generalizability of our approach, we further evaluate SAGE on the Qwen2.5-VL series with a 72B target model and a 7B draft model as shown in Table 2. SAGE consistently achieves the best performance across all three video benchmarks. On VideoDetailedCaption, SAGE obtains an acceptance length of 5.18 and a speedup of 3.18×\times, outperforming SpecVLM by 32.8% in acceptance length and 17.8% in speedup ratio. On MVBench, SAGE achieves 4.53 average acceptance length with 2.70×\times speedup, compared to 3.73 and 2.45×\times for SpecVLM. These results demonstrate that our entropy-guided adaptive strategy generalizes effectively across different VLM architectures, consistently improving both acceptance length and inference throughput without requiring any architecture-specific modifications or additional training.

5.3 Performance Evaluation on MoE Models

To investigate the applicability of our approach to Mixture-of-Experts (MoE) architectures, we conduct experiments on Qwen3-VL with 235B total parameters and 22B activated, using an 8B dense draft model. As shown in Table 3, SAGE achieves an acceptance length of 4.39 and a speedup of 1.32×\times on VideoDetailedCaption, outperforming both SD-Tree with 3.58 acceptance length and 1.22×\times speedup, and SpecVLM with 3.24 acceptance length and 1.14×\times speedup. The relatively modest speedup compared to dense models is expected, as MoE architectures already benefit from sparse activation during inference, reducing the computational gap between draft and target models. Nevertheless, SAGE consistently delivers the highest acceptance length and throughput, demonstrating that our entropy-guided adaptive strategy remains effective for MoE-based VLMs where the draft-target alignment poses additional challenges due to architectural heterogeneity.

Setup
Method
VideoDetailedCaption

τ\tau
Tokens/s
Speedup

Qwen3-VL

235B/A22B-8B

Vanilla AR
–
3.27
–

SD-Tree
3.58
3.99

1.22×\times

SpecVLM
3.24
3.74

1.14×\times

SAGE\cellcolorgray!20

\cellcolorgray!204.39

\cellcolorgray!204.33

\cellcolorgray!201.32×\times

Table 3: Performance evaluation on MoE architecture. Average accepted length τ\tau, decoding speed (tokens/s), and speedup of Qwen3-VL (235B total parameters with 22B activated, using 8B draft model) on VideoDetailedCaption.

Dataset
Method
Llama3 8B-1B

τ\tau
Tokens/s
Speedup

Gsm8k
Vanilla
–
21.01
–

Native-SD
3.12
44.59

2.12×\times

SAGE\cellcolorgray!20

4.05\cellcolorgray!20

48.31\cellcolorgray!20

2.30×\times\cellcolorgray!20

Humaneval
Vanilla
–
18.80
–

Native-SD
3.48
44.89

2.39×\times

SAGE\cellcolorgray!20

4.89\cellcolorgray!20

50.20\cellcolorgray!20

2.67×\times\cellcolorgray!20

Table 4: Performance evaluation on large language model tasks using Llama3 (8B target model with 1B draft model). SAGE consistently outperforms native speculative decoding on both mathematical reasoning (Gsm8k) and code generation (Humaneval) benchmarks.

5.4 Performance Evaluation on LLMs

To demonstrate that our entropy-guided adaptive strategy is not limited to vision-language models, we evaluate SAGE on pure language generation tasks using Llama3 with an 8B target model and a 1B draft model. As shown in Table 4, SAGE consistently outperforms native speculative decoding on both mathematical reasoning Gsm8k and code generation Humaneval benchmarks. On Gsm8k, SAGE achieves an acceptance length of 4.05 compared to 3.12 for Native-SD, yielding a speedup of 2.30×\times. On Humaneval, SAGE attains an acceptance length of 4.89, which is 40.5% higher than Native-SD, with a speedup of 2.67×\times. These results validate the generality of our approach beyond VLMs.

5.5 Ablation Studies

Setup
Method
VideoDetailedCaption

τ\tau
Tokens/s
Speedup

LLaVA-OV

72B-7B

Vanilla AR
-
4.69
-

SAGE0\text{SAGE}_{0}
5.60
11.11

2.37×\times

SAGE0.3\text{SAGE}_{0.3}
5.70
12.66

2.70×\times

SAGE0.6\text{SAGE}_{0.6}
5.81
14.67

3.13×\times

SAGE0.7\text{SAGE}_{0.7}
5.82
15.26

3.25×\times

SAGE0.8\text{SAGE}_{0.8}
5.83
15.73

3.35×\times

SAGE0.9\text{SAGE}_{0.9}
5.74
15.75

3.36×\times

SAGE0.95\text{SAGE}_{0.95}
5.55
15.16

3.23×\times

Table 5: Ablation study on the effect of visual token pruning ratio. Results are reported on VideoDetailedCaption using LLaVA-OneVision (72B-7B). The subscript denotes the proportion of visual tokens pruned.

Effect of Pruning Ratio. We investigate the impact of visual token pruning ratio on SAGE’s performance using LLaVA-OneVision 72B-8B on VideoDetailedCaption. As shown in Table 5, without pruning, SAGE0\text{SAGE}_{0} achieves an acceptance length of 5.60 but only 2.37×\times speedup due to the overhead of processing all visual tokens. As the pruning ratio increases, both throughput and speedup improve while the acceptance length remains stable. The optimal performance is achieved at pruning ratios between 0.8 and 0.9, where SAGE0.9\text{SAGE}_{0.9} attains 5.74 acceptance length with 3.36×\times speedup. However, excessive pruning with SAGE0.95\text{SAGE}_{0.95} leads to degradation as aggressive pruning removes informative visual tokens. These results suggest that moderate pruning effectively reduces draft model overhead while preserving sufficient visual context for accurate speculation.

Figure 3: Performance comparison between SAGE and SpecVLM across
different generation lengths on VideoDetailedCaption (LLaVA-OneVision 72B-7B).

Effect of Generation Token Length. As shown in Figure 3, we analyze how generation length affects performance by comparing SAGE and SpecVLM. Both methods exhibit improved acceptance lengths as generation length increases, attributed to more predictable generation patterns in longer sequences. However, SAGE consistently outperforms SpecVLM across all settings, with the performance gap widening as generation length grows. At 512 tokens, SAGE achieves 25.4% improvement over SpecVLM in acceptance length. This gap increases substantially at longer generations: 39.7% improvement at 1024 tokens and 56.8% at 2048 tokens. The throughput advantage follows a similar trend. These results demonstrate that our entropy-guided adaptive strategy becomes increasingly effective for longer generation tasks, as the method can better exploit the temporal patterns of prediction confidence over extended sequences.

6 Related Work

6.1 Efficient Inference for Vision-Language Models

Vision-language models such as BLIP-2 (Li et al., 2023b), LLaVA (Lin et al., 2024), and Qwen-VL (Bai et al., 2025a) achieve remarkable multimodal performance but suffer from substantial computational demands. Various acceleration techniques have been proposed, including token pruning (Ye et al., 2025) and quantization (Tong et al., 2025a, b). SpecVLM (Ji et al., 2025) introduces visual token pruning in the draft model to bridge the modality gap for speculative decoding in VLMs. Our work takes an orthogonal perspective by dynamically adapting the tree structure based on prediction uncertainty, which can be integrated with existing techniques.

6.2 Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Sun et al., 2023) accelerates auto-regressive generation by using a smaller draft model to generate candidate tokens verified in parallel by the target model. To improve acceptance rates, tree-structured drafting has been widely adopted. SpecInfer (Miao et al., 2024) organizes candidates as a tree for simultaneous verification. Medusa (Cai et al., 2024) adds multiple decoding heads for tree-structured generation, while EAGLE (Li et al., 2024d) employs feature-level auto-regressive drafting. However, most speculative decoding research focuses on text-only LLMs. The unique challenges of VLMs—including the modality gap and longer context lengths due to visual tokens—have received less attention. Only a few works such as SpecVLM (Ji et al., 2025) have explored this direction, yet they still rely on static tree configurations, which motivates our adaptive approach.

6.3 Adaptive Decoding Strategies

Adaptive methods that adjust decoding behavior based on runtime signals have shown promise in various contexts. CALM (Schuster et al., 2022) uses confidence for early exit in transformer layers, and entropy-based uncertainty estimation has been widely adopted for uncertainty quantification in language models (Kuhn et al., 2023) and model calibration (Guo et al., 2017). In LLM speculative decoding, EAGLE-2 (Li et al., 2024c) introduced context-aware dynamic draft trees by using token-level confidence scores to guide branch expansion during drafting. Our SAGE differs in three aspects: (1) we use distributional entropy rather than individual token probabilities as the confidence measure; (2) we adopt a lookahead strategy that pre-plans tree structure for the next round based on current entropy, exploiting temporal correlation across decoding steps; (3) we perform global resource allocation by jointly modulating depth and width inversely. To our knowledge, SAGE is the first entropy-guided adaptive tree construction method for speculative decoding in VLMs.

7 Conclusions

We introduce SAGE, a novel framework that dynamically adjusts the draft tree structure based on real-time prediction uncertainty for vision-language models. Our key insight is that output entropy serves as a natural confidence indicator: low entropy enables deeper speculation, while high entropy favors wider exploration at shallow depths. By adaptively constructing deeper-narrower trees for confident predictions and shallower-wider trees for uncertain ones, SAGE achieves improved acceptance lengths and faster acceleration compared to static tree baselines.

References

S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu (2025a)
Qwen3-vl technical report.

arXiv preprint arXiv:2511.21631.

Cited by: §5.1,
§6.1.

S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025b)
Qwen2. 5-vl technical report.

arXiv preprint arXiv:2502.13923.

Cited by: §5.1.

T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao (2024)
Medusa: simple llm inference acceleration framework with multiple decoding heads.

arXiv preprint arXiv:2401.10774.

Cited by: §6.2.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger (2017)
On calibration of modern neural networks.

In International conference on machine learning,

pp. 1321–1330.

Cited by: §6.3.

A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi (2020)
The curious case of neural text degeneration.

In International Conference on Learning Representations,

External Links: Link

Cited by: §1,
§3.1.

D. A. Hudson and C. D. Manning (2019)
Gqa: a new dataset for real-world visual reasoning and compositional question answering.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 6700–6709.

Cited by: §5.1.

Y. Ji, J. Zhang, H. Xia, J. Chen, L. Shou, G. Chen, and H. Li (2025)
Specvlm: enhancing speculative decoding of video llms via verifier-guided token pruning.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,

pp. 7216–7230.

Cited by: §A.1,
§1,
§5.1,
§5.1,
§6.1,
§6.2.

S. Kim, K. Mangalam, S. Moon, J. Malik, M. W. Mahoney, A. Gholami, and K. Keutzer (2023)
Speculative decoding with big little decoder.

Advances in Neural Information Processing Systems 36, pp. 39236–39256.

Cited by: §1.

L. Kuhn, Y. Gal, and S. Farquhar (2023)
Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation.

In The Eleventh International Conference on Learning Representations,

External Links: Link

Cited by: §6.3.

Y. Leviathan, M. Kalman, and Y. Matias (2023)
Fast inference from transformers via speculative decoding.

In International Conference on Machine Learning,

pp. 19274–19286.

Cited by: §1,
§1,
§6.2.

B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, et al. (2024a)
Llava-onevision: easy visual task transfer.

arXiv preprint arXiv:2408.03326.

Cited by: §1,
§5.1.

B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan (2023a)
Seed-bench: benchmarking multimodal llms with generative comprehension.

arXiv preprint arXiv:2307.16125.

Cited by: §5.1.

J. Li, D. Li, S. Savarese, and S. Hoi (2023b)
Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models.

In International conference on machine learning,

pp. 19730–19742.

Cited by: §6.1.

K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo, et al. (2024b)
Mvbench: a comprehensive multi-modal video understanding benchmark.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 22195–22206.

Cited by: §5.1.

Y. Li, F. Wei, C. Zhang, and H. Zhang (2024c)
Eagle-2: faster inference of language models with dynamic draft trees.

arXiv preprint arXiv:2406.16858.

Cited by: §1,
§3.1,
§6.3.

Y. Li, F. Wei, C. Zhang, and H. Zhang (2024d)
Eagle: speculative sampling requires rethinking feature uncertainty.

arXiv preprint arXiv:2401.15077.

Cited by: §6.2.

B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan (2024)
Video-llava: learning united visual representation by alignment before projection.

In Proceedings of the 2024 conference on empirical methods in natural language processing,

pp. 5971–5984.

Cited by: §6.1.

A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque (2022)
Chartqa: a benchmark for question answering about charts with visual and logical reasoning.

In Findings of the association for computational linguistics: ACL 2022,

pp. 2263–2279.

Cited by: §5.1.

X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, et al. (2024)
Specinfer: accelerating large language model serving with tree-based speculative inference and verification.

In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3,

pp. 932–949.

Cited by: §6.2.

T. Schuster, A. Fisch, J. Gupta, M. Dehghani, D. Bahri, V. Tran, Y. Tay, and D. Metzler (2022)
Confident adaptive language modeling.

Advances in Neural Information Processing Systems 35, pp. 17456–17472.

Cited by: §6.3.

A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach (2019)
Towards vqa models that can read.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 8317–8326.

Cited by: §5.1.

Z. Sun, A. T. Suresh, J. H. Ro, A. Beirami, H. Jain, and F. Yu (2023)
Spectr: fast speculative decoding via optimal transport.

Advances in Neural Information Processing Systems 36, pp. 30222–30242.

Cited by: §1,
§1,
§6.2.

Y. Tong, Y. Wang, J. Yuan, and C. Hu (2025a)
Robust machine unlearning for quantized neural networks via adaptive gradient reweighting with similar labels.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

pp. 20603–20612.

Cited by: §6.1.

Y. Tong, J. Yuan, T. Zhang, J. Liu, and C. Hu (2025b)
Data-free quantization of vision transformers via easy-to-hard synthesis and activation correction.

ACM Trans. Multimedia Comput. Commun. Appl..

Note: Just Accepted

External Links: ISSN 1551-6857,
Link,
Document

Cited by: §6.1.

P. K. A. Vasu, F. Faghri, C. Li, C. Koc, N. True, A. Antony, G. Santhanam, J. Gabriel, P. Grasch, O. Tuzel, and H. Pouransari (2025)
FastVLM: efficient vision encoding for vision language models.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 19769–19780.

Cited by: §1.

H. Wu, D. Li, B. Chen, and J. Li (2024)
Longvideobench: a benchmark for long-context interleaved video-language understanding.

Advances in Neural Information Processing Systems 37, pp. 28828–28857.

Cited by: §5.1.

H. Xia, T. Ge, P. Wang, S. Chen, F. Wei, and Z. Sui (2023)
Speculative decoding: exploiting speculative execution for accelerating seq2seq generation.

In Findings of the Association for Computational Linguistics: EMNLP 2023,

pp. 3909–3925.

Cited by: §1.

W. Ye, Q. Wu, W. Lin, and Y. Zhou (2025)
Fit and prune: fast and training-free visual token pruning for multi-modal large language models.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 39, pp. 22128–22136.

Cited by: §6.1.

J. Zhang, J. Huang, S. Jin, and S. Lu (2024)
Vision-language models for vision tasks: a survey.

IEEE transactions on pattern analysis and machine intelligence 46 (8), pp. 5625–5644.

Cited by: §1.

J. Zhou, Y. Shu, B. Zhao, B. Wu, S. Xiao, X. Yang, Y. Xiong, B. Zhang, T. Huang, and Z. Liu (2024)
Mlvu: a comprehensive benchmark for multi-task long video understanding.

arXiv e-prints, pp. arXiv–2406.

Cited by: §5.1.

Appendix A Appendix

A.1 Initialized draft tree structure

Following SpecVLM (Ji et al., 2025), we adopt the tree structure illustrated in Figure 4 for both static baselines and SAGE’s initial configuration. For SD-Tree and SpecVLM, this structure remains fixed during decoding. SAGE uses it only as initialization, then dynamically adjusts tree depth and width based on prediction entropy after each iteration.

Figure 4: Initialized draft tree structure.

A.2 Analysis of Computational Time Overhead

To understand the efficiency of SAGE, we conduct a detailed analysis of the computational time distribution across different operations during inference. Table 6 presents the time proportion of each operation. The decoding phase dominates the total inference time, with target model verification and draft model decoding accounting for 36.59% and 34.81% respectively, which aligns with the motivation of speculative decoding that aims to accelerate auto-regressive generation through parallel verification. The target model prefilling consumes 26.72% of the total time due to the processing of visual tokens from video frames, while the draft model prefilling only requires 0.77%. Most importantly, the additional overhead introduced by SAGE is negligible: the entropy-based dynamic tree update requires only 0.39% of the total time. This minimal overhead validates our design choice of using output entropy as a lightweight confidence indicator, as entropy computation leverages the probability distribution already available from the softmax operation without requiring additional forward passes.

Operation
Target Prefilling
Target Verification
Draft Prefilling
Draft Decoding
Token Pruning
Tree Update
Others
Total

Time Proportion
26.72%
36.59%
0.77%
34.81%
0.03%
0.39%
0.69%
100%

Table 6: Computational time breakdown of SAGE during inference.

A.3 Detailed Theoretical Proofs

Proof of Lemma 4.3.

We solve the optimization problem:

minp1,…,pk⁡p1s.t.H​(P~d)=(1−α)​log⁡k,∑ipi=1,pi≥0,p1≥p2≥⋯≥pk.\min_{p_{1},\ldots,p_{k}}p_{1}\quad\text{s.t.}\quad H(\tilde{P}_{d})=(1-\alpha)\log k,\quad\sum_{i}p_{i}=1,\quad p_{i}\geq 0,\quad p_{1}\geq p_{2}\geq\cdots\geq p_{k}.

(18)

For fixed p1p_{1}, entropy is maximized when the remaining mass (1−p1)(1-p_{1}) is distributed uniformly among the other k−1k-1 tokens, giving pi=(1−p1)/(k−1)p_{i}=(1-p_{1})/(k-1) for i≥2i\geq 2. The resulting entropy is:

H∗​(p1)=−p1​log⁡p1−(1−p1)​log⁡1−p1k−1.H^{*}(p_{1})=-p_{1}\log p_{1}-(1-p_{1})\log\frac{1-p_{1}}{k-1}.

(19)

Since H∗​(p1)H^{*}(p_{1}) is strictly decreasing in p1p_{1} for p1∈[1/k,1]p_{1}\in[1/k,1], and H∗​(1/k)=log⁡kH^{*}(1/k)=\log k while H∗​(1)=0H^{*}(1)=0, there exists a unique p1∗p_{1}^{*} satisfying H∗​(p1∗)=(1−α)​log⁡kH^{*}(p_{1}^{*})=(1-\alpha)\log k.

For the constraint H​(P~d)=(1−α)​log⁡kH(\tilde{P}_{d})=(1-\alpha)\log k to be achievable with p1p_{1} as small as possible, we need H∗​(p1)≥(1−α)​log⁡kH^{*}(p_{1})\geq(1-\alpha)\log k. Using the concavity of entropy and linear interpolation between the boundary cases:

p1≥1k+α⋅(1−1k)=1k+α⋅k−1k.p_{1}\geq\frac{1}{k}+\alpha\cdot\left(1-\frac{1}{k}\right)=\frac{1}{k}+\alpha\cdot\frac{k-1}{k}.

(20)

This bound is tight when α∈{0,1}\alpha\in\{0,1\}.
∎

Proof of Theorem 4.4.

By the definition of total variation distance, for any token yy:

|Pd(y|c)−Pt(y|c)|≤2DT​V(Pd(⋅|c),Pt(⋅|c))≤2ϵ.|P_{d}(y|c)-P_{t}(y|c)|\leq 2D_{TV}(P_{d}(\cdot|c),P_{t}(\cdot|c))\leq 2\epsilon.

(21)

Thus Pt​(y^|c)≥Pd​(y^|c)−2​ϵ=p1−2​ϵP_{t}(\hat{y}|c)\geq P_{d}(\hat{y}|c)-2\epsilon=p_{1}-2\epsilon.

For y^\hat{y} to be the target model’s argmax, we need Pt​(y^|c)>Pt​(y|c)P_{t}(\hat{y}|c)>P_{t}(y|c) for all y≠y^y\neq\hat{y}. In the worst case, the target model places all non-y^\hat{y} probability mass on a single alternative token y′y^{\prime}:

Pt​(y′|c)≤1−Pt​(y^|c)≤1−(p1−2​ϵ).P_{t}(y^{\prime}|c)\leq 1-P_{t}(\hat{y}|c)\leq 1-(p_{1}-2\epsilon).

(22)

The acceptance condition Pt​(y^|c)>Pt​(y′|c)P_{t}(\hat{y}|c)>P_{t}(y^{\prime}|c) is satisfied when:

p1−2​ϵ>1−(p1−2​ϵ)⟹p1>12+2​ϵ.p_{1}-2\epsilon>1-(p_{1}-2\epsilon)\implies p_{1}>\frac{1}{2}+2\epsilon.

(23)

Substituting the lower bound from Lemma 4.3:

1k+α⋅k−1k>12+2​ϵ.\frac{1}{k}+\alpha\cdot\frac{k-1}{k}>\frac{1}{2}+2\epsilon.

(24)

Solving for α\alpha:

α>kk−1​(12+2​ϵ−1k)=k−2+4​ϵ​k2​(k−1).\alpha>\frac{k}{k-1}\left(\frac{1}{2}+2\epsilon-\frac{1}{k}\right)=\frac{k-2+4\epsilon k}{2(k-1)}.

(25)

∎

Remark A.1.

The bound in Theorem 4.4 is conservative, providing a sufficient condition for guaranteed acceptance. In practice, acceptance can occur even when p1≤1/2+ϵp_{1}\leq 1/2+\epsilon if the target model’s probability mass is not adversarially distributed. Our empirical results show that acceptance probability increases smoothly with α\alpha, suggesting the actual relationship is stronger than this worst-case bound.

Corollary A.2 (Multi-Step Acceptance).

For a path of depth dd in the speculation tree, assuming independence across steps and that each step has confidence αl\alpha_{l} satisfying the threshold in Eq. 14, all dd tokens are guaranteed to be accepted. For confidence levels below the threshold, the acceptance probability decreases with depth.

Proof of Theorem 4.6.

Let AlA_{l} denote the event that the token at depth ll is accepted. The probability of accepting exactly ll tokens is:

P​(τ=l)=P​(A1,…,Al,Al+1¯)=(∏j=1lpj)​(1−pl+1),for ​l<D,P(\tau=l)=P(A_{1},\ldots,A_{l},\overline{A_{l+1}})=\left(\prod_{j=1}^{l}p_{j}\right)(1-p_{l+1}),\quad\text{for }l<D,

(26)

and P​(τ=D)=∏j=1DpjP(\tau=D)=\prod_{j=1}^{D}p_{j}.

The expected acceptance length is:

𝔼​[τ]\displaystyle\mathbb{E}[\tau]
=∑l=1D−1l⋅(∏j=1lpj)​(1−pl+1)+D⋅∏j=1Dpj\displaystyle=\sum_{l=1}^{D-1}l\cdot\left(\prod_{j=1}^{l}p_{j}\right)(1-p_{l+1})+D\cdot\prod_{j=1}^{D}p_{j}

(27)

=∑l=1D−1l⋅∏j=1lpj−∑l=1D−1l⋅∏j=1l+1pj+D⋅∏j=1Dpj\displaystyle=\sum_{l=1}^{D-1}l\cdot\prod_{j=1}^{l}p_{j}-\sum_{l=1}^{D-1}l\cdot\prod_{j=1}^{l+1}p_{j}+D\cdot\prod_{j=1}^{D}p_{j}

(28)

=∑l=1D−1l⋅∏j=1lpj−∑l=2D(l−1)⋅∏j=1lpj+D⋅∏j=1Dpj\displaystyle=\sum_{l=1}^{D-1}l\cdot\prod_{j=1}^{l}p_{j}-\sum_{l=2}^{D}(l-1)\cdot\prod_{j=1}^{l}p_{j}+D\cdot\prod_{j=1}^{D}p_{j}

(29)

=p1+∑l=2D−1∏j=1lpj+∏j=1Dpj=∑l=1D∏j=1lpj.\displaystyle=p_{1}+\sum_{l=2}^{D-1}\prod_{j=1}^{l}p_{j}+\prod_{j=1}^{D}p_{j}=\sum_{l=1}^{D}\prod_{j=1}^{l}p_{j}.

(30)

Substituting pj=p⋅γj−1p_{j}=p\cdot\gamma^{j-1}:

∏j=1lpj=pl⋅γ0+1+⋯+(l−1)=pl⋅γl​(l−1)/2.\prod_{j=1}^{l}p_{j}=p^{l}\cdot\gamma^{0+1+\cdots+(l-1)}=p^{l}\cdot\gamma^{l(l-1)/2}.

(31)

∎

Proof of Theorem 4.7.

With γ=1\gamma=1 and Wl=1W_{l}=1, the tree has |𝒯|=D|\mathcal{T}|=D nodes, and from Theorem 4.6:

𝔼​[τ]=∑l=1Dpl=p⋅1−pD1−p.\mathbb{E}[\tau]=\sum_{l=1}^{D}p^{l}=p\cdot\frac{1-p^{D}}{1-p}.

(32)

The speedup ratio becomes:

S​(D)=p⋅1−pD1−p+1cd⋅D+ct.S(D)=\frac{p\cdot\frac{1-p^{D}}{1-p}+1}{c_{d}\cdot D+c_{t}}.

(33)

The marginal benefit of increasing depth from DD to D+1D+1 is:

Δbenefit​(D)=pD+1.\Delta_{\text{benefit}}(D)=p^{D+1}.

(34)

The marginal benefit of increasing depth is pD+1p^{D+1} (additional expected accepted tokens), while the marginal cost is cdc_{d}. To compare these quantities in consistent units relative to a single target model forward pass (cost ctc_{t}), we require the benefit-to-cost ratio:

pD+1>cdct⟹(D+1)​log⁡p>log⁡cdct⟹D<log⁡(ct/cd)log⁡(1/p)−1.p^{D+1}>\frac{c_{d}}{c_{t}}\implies(D+1)\log p>\log\frac{c_{d}}{c_{t}}\implies D<\frac{\log(c_{t}/c_{d})}{\log(1/p)}-1.

(35)

Thus the optimal depth is D∗=⌊log⁡(ct/cd)/log⁡(1/p)⌋D^{*}=\lfloor\log(c_{t}/c_{d})/\log(1/p)\rfloor. As pp increases, log⁡(1/p)\log(1/p) decreases, so D∗D^{*} increases.
∎

Proof of Theorem 4.8.

With depth D=1D=1 and width WW, the probability that the ii-th candidate is accepted is qi=q1/iβq_{i}=q_{1}/i^{\beta}. The expected acceptance is upper-bounded by:

𝔼​[accept]≤∑i=1Wqi=q1​∑i=1W1iβ≈q1⋅W1−β1−βfor ​β<1.\mathbb{E}[\text{accept}]\leq\sum_{i=1}^{W}q_{i}=q_{1}\sum_{i=1}^{W}\frac{1}{i^{\beta}}\approx q_{1}\cdot\frac{W^{1-\beta}}{1-\beta}\quad\text{for }\beta<1.

(36)

The speedup ratio is:

S​(W)=𝔼​[accept]+1cd⋅W+ct.S(W)=\frac{\mathbb{E}[\text{accept}]+1}{c_{d}\cdot W+c_{t}}.

(37)

Taking the derivative and setting to zero, after simplification we obtain:

q1​W−β​(cd​W+ct)=cd​(q1​W1−β1−β+1).q_{1}W^{-\beta}(c_{d}W+c_{t})=c_{d}\left(\frac{q_{1}W^{1-\beta}}{1-\beta}+1\right).

(38)

For the regime where the “+1+1” term dominates (low acceptance), this simplifies to:

q1​cd​W1−β≈cd⟹W∗≈(1q1)1/(1−β).q_{1}c_{d}W^{1-\beta}\approx c_{d}\implies W^{*}\approx\left(\frac{1}{q_{1}}\right)^{1/(1-\beta)}.

(39)

More generally, dimensional analysis suggests W∗∼(q1/cd)1/(1+β)W^{*}\sim(q_{1}/c_{d})^{1/(1+\beta)}. The key qualitative insight is that when q1q_{1} is high, the first candidate is likely correct, reducing the marginal benefit of additional width.
∎

Remark A.3.

Our theoretical analysis makes several simplifying assumptions, including independence across decoding steps and specific functional forms for acceptance probability. The primary contribution of this analysis is to establish that the direction of our adaptations—deeper trees for confident predictions, wider trees for uncertain ones—has a principled foundation. The consistent empirical improvements across diverse settings (Section 5) suggest these design principles are robust even when the simplified assumptions do not hold exactly.

Generated on Sat Jan 31 05:31:38 2026 by LaTeXML