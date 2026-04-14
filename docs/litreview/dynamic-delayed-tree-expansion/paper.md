Dynamic Delayed Tree Expansion For Improved Multi-Path Speculative Decoding

1 Introduction

2 Background

2.1 Performance Comparisons

2.2 Improving Drafting

Context-dependent tree structures.

Offline tree optimization for block efficiency.

Dynamic draft length control.

Training-based tree policy.

Hardware-Aware Tree Decoding.

3 Verification Algorithms

3.1 Single-Path Algorithms

Naive speculative sampling.

Tree verification.

Block verification (BV).

3.2 Multi-Path Algorithms

OT-based algorithms.

Traversal verification.

3.3 Other Algorithms

4 Comparison of Existing Verification Algorithms

4.1 Experimental Setup

4.2 Results

5 Delayed Tree Expansion

6 Neural Delay-and-Branch Predictor

6.1 Experiments

7 Conclusion

8 Impact Statement

A Orthogonal Drafting Work

Improving drafting without tree control.

Pipelining and parallelism.

Alternative multi-token prediction.

B Review of OT-Based Methods

B.1 NSS

B.2 Naive

B.3 SpecTr

B.4 SpecInfer

B.5 Khisti

C OTLP Acceptances

D OTLP Branching

E Neural Selector Details

F Extended Online Experimental Results

Dynamic Delayed Tree Expansion

For Improved Multi-Path Speculative Decoding

Rahul Thomas

Teo Kitanovski

Micah Goldblum

Arka Pal

Abstract

Multi-path speculative decoding accelerates lossless sampling from a target model by using a cheaper draft model to generate a draft tree of tokens, and then applies a verification algorithm that accepts a subset of these. While prior work has proposed various verification algorithms for i.i.d rollouts, their relative performance under matched settings remains unclear. In this work, we firstly present a systematic evaluation of verification strategies across model families, tasks, and sampling regimes, and find that Traversal Verification dominates consistently, with OT-based methods lagging far behind. Our analysis uncovers that this occurs because OT-based methods achieve high multi-token acceptance near the root of the draft tree, while multi-token gains are most impactful deeper in the draft tree, where draft and target distributions diverge. Based on this insight, we propose delayed tree expansion, which drafts a partial single path, delaying the i.i.d. branching point. We show that delayed tree expansion preserves the target distribution and improves on root-node i.i.d rollouts. Further, we develop a dynamic neural selector that estimates the expected block efficiency of optimal-transport-based verification methods from draft and target features, enabling context-dependent expansion decisions. Our neural selector allows OT-based methods like SpecInfer to outperform Traversal Verification for the first time, achieving 5% higher average throughput across a wide range of models, datasets, and sampling settings.

Machine Learning, ICML

1 Introduction

In recent years, large language models (LLMs) have demonstrated impressive performance across a wide range of domains, including but not limited to translation, reasoning, mathematics, and coding (Zhu et al., 2024; Kasneci et al., 2023; Hendrycks et al., 2021; Chen et al., 2021). Despite such rapid progress, end-to-end latency remains a significant bottleneck in real LLM deployments. Most LLM families, such as Gemma (Team, 2024; Team et al., 2025a), Qwen (Bai et al., 2023; Hui et al., 2024), and Llama (Touvron et al., 2023b, a; Grattafiori et al., 2024), are autoregressive and require an expensive forward pass for each output token.

Exploiting the fact that the forward pass is memory-bound (Fu et al., 2024), speculative decoding (Chen et al., 2023; Leviathan et al., 2023) improves GPU utilization by decoding multiple tokens per pass in a three-step process. First, during drafting, a cheap draft model proposes multiple future tokens. Then, the original LLM performs a target forward pass over these tokens in parallel. Finally, during verification, some tokens are rejected to maintain the target distribution. Throughput gains depend on the drafting and target pass walltimes, as well as block efficiency – the number of average accepted tokens per target model call.

Recent work has generalized speculative decoding to the multi-path setting, where the draft model proposes a tree of tokens and the target pass uses a tree attention mask. Multi-path methods improve block efficiency by diversifying the draft token pool. In order to maintain the target distribution, specific multi-path verification algorithms are required. These are usually based on optimal transport (OT) solvers that traverse the draft tree in a top-down manner (Sun et al., 2023; Khisti et al., 2025; Miao et al., 2024), although some incorporate bottom-up traversal (Sun et al., 2024c; Weng et al., 2025). Concurrently, there have been efforts to improve the efficacy of drafting for a fixed verification method via confidence-guided tree construction, adaptive drafting depth, and pipelining (Li et al., 2024a; Brown et al., 2024; Xiong et al., 2025; Chen et al., 2024a; Wang et al., 2025; Huang et al., 2024; Liu et al., 2024b; Guan et al., 2025).

However, despite the significant body of prior work on multi-path speculative decoding, there has been no systematic and controlled comparison of multi-path verification methods. Most papers introducing new algorithms test against only a handful of relevant baselines, or compare only to single-path drafts. Moreover, prior works which introduce new tree construction methods (Li et al., 2024a; Brown et al., 2024; Xiong et al., 2025; Chen et al., 2024a; Wang et al., 2025; Huang et al., 2024; Liu et al., 2024b; Guan et al., 2025) can often only be used with specific verification algorithms in order to maintain the target distribution. Therefore, an important open question remains: which multi-path verification method performs best, and under what conditions? Prior works often compare such methods under different drafting methods, datasets, and model families, or only consider the most basic verification methods, making it difficult to measure the true progress of verification gains.

In this work, we compare multi-path verification methods under matched i.i.d. draft settings across diverse model families, tasks, and sampling regimes. We find that Traversal Verification consistently dominates all other methods. Our analysis reveals that this is the result of wasteful expansion. These other methods, which are all OT-based, achieve highest token acceptances near the root of the draft tree, but the most promising improvements occur deeper in the tree when draft and target distributions diverge. Generating a draft tree from i.i.d. root rollouts forces token diversity into shallow nodes, even when it is more beneficial at deeper nodes.

This motivates us to introduce delayed expansion, where we draft a partial single path, and then expand into i.i.d. rollouts at a ”branching point”. This drafting policy allows explicit control of three parameters: the single path depth, the expansion factor, and the expansion depth. Optimizing these is a careful balancing act between block efficiency and tree size, because larger trees accept more tokens on average but also incur higher drafting and target pass costs. Following adaptive drafting methods that maximize expected block efficiency offline (Chen et al., 2024a; Brown et al., 2024), we train a lightweight MLP to predict the optimal values of these parameters from root node features and system-specific draft and target pass times. With this neural predictor, OT-based methods can finally outperform Traversal Verification. In summary, our contributions are as follows:

•

We provide the first systematic comparison of existing i.i.d. multi-path speculative decoding verification methods across varying model families, datasets, and sampling settings. We find Traversal Verification performs far better than all other methods (OT-based).

•

To explain the pitfalls of OT-based methods, we analyze acceptance rates and find that the marginal performance improvements between these methods are caused by increasingly diverging target and draft distributions deeper in the tree. Branching early in the tree, where target and draft distributions are most similar, yields relatively less useful acceptance improvements.

•

To mitigate this, we utilize delayed expansion, where the draft tree is formed by drafting a path and then branching into many i.i.d. paths. We train a context-dependent neural predictor on offline block efficiency estimates to predict optimal path and branch parameters from model and latency features. In online deployments, our method can be applied to all OT-based i.i.d. multi-path methods. Across a multitude of settings, our neural predictor with the OT-based method SpecInfer improves average throughput over Traversal Verification by ∼5%\sim 5\%.

2 Background

Speculative decoding uses a cheap draft model to decode multiple tokens in one target model call, in three steps:

1.

Drafting. Using only the draft model, we expand a tree of tokens from the current context and compute next-token draft distributions on the tree nodes.

2.

Target Pass. The target model performs a batched forward pass over the draft tree, with a custom attention mask that respects ancestor-only dependencies, to obtain next-token target distributions on tree nodes.

3.

Verification. Based on the target and draft distributions, we randomly accept a single node on the draft tree and append an additional correction token, such that the output matches the target model distribution.

The acceptance length τ\tau is the depth of the accepted node, which grows as the target and draft distributions coincide. The block efficiency 𝔼​[τ+1]\mathbb{E}[\tau+1] is the average number of decoded tokens per target call. When drafting is cheap and the batched target pass takes as long as a target forward pass, block efficiency accurately represents speedup.

2.1 Performance Comparisons

The literature lacks comprehensive comparisons of speculative decoding algorithms. Xia et al. (2024) surveys drafting and verification strategies in speculative decoding, and releases Spec-Bench, a third-party testing environment used to compare SOTA methods under common environment setups. However, their dataset is relatively small, they only support a few verification algorithms, and their evaluations do not incorporate temperature or nucleus sampling variations. More recently, Liu et al. (2025) gives the first production-grade vLLM study (Kwon et al., 2023) on the effects of batch size, model family, and workloads on speculative decoding and variants like EAGLE, EAGLE-3, and multi-token prediction. Still, they use only basic verification algorithms and greedy decoding. We are the first to compare all verification algorithms across models, tasks, and sampling settings.

2.2 Improving Drafting

Existing work has focused on improving block efficiency by modifying drafting or verification. Verification works build novel algorithms that alter the node selection scheme to improve acceptance lengths. Note that verification and draft improvements can be integrated together when compatible, and many of these drafting methods rely on specific verification methods like NSS (Miao et al., 2024), Naive (Chen et al., 2023; Leviathan et al., 2023), and SpecInfer (Miao et al., 2024). We discuss these in Section 3 and focus on drafting here. We focus on works that alter the draft tree construction itself, as these are most relevant to our work.
We provide further details on orthogonal work in Appendix A.

Context-dependent tree structures.

Various works alter the draft topology online to only expand nodes with meaningful acceptance improvements. EAGLE-2 (Li et al., 2024a) deterministically expands top-kk tokens from nodes with the highest global draft probabilities, using draft confidence as a proxy for acceptance. Dynamic Depth Decoding (Brown et al., 2024) improves EAGLE-2 by adaptively selecting how deep to expand within its deterministic tree structure. EDD and PCT (Zheng and Wang, 2025) prune candidate branches with low draft confidence, so that the target model compute is spent on branches likely to be accepted. Instead of fixing a deterministic draft tree, DySpec (Xiong et al., 2025) dynamically allocates a token tree expansion budget and then samples a draft tree from the token tree online. Our methods also utilize context-dependent tree topologies, but they can more effectively push towards high acceptance regions by relying on offline block efficiency supervision rather than simple proxy metrics.

Offline tree optimization for block efficiency.

Another line of work uses explicit objectives for expected accepted length to select a set of optimal trees offline. Sequoia (Chen et al., 2024a) uses dynamic programming to select deterministic draft structures under various budget constraints, which optimize SpecInfer block efficiency for sampling without replacement. OPT-Tree (Wang et al., 2025) instead maximizes expected acceptance under NSS. While these methods are similar to ours in that they optimize expected block efficiency, their fixed offline tree collection remains highly sensitive to the dataset and sampling settings, a pitfall which our per-context neural selector naturally overcomes.

Dynamic draft length control.

Because the search over tree topologies is complex, many draft tree selectors decide how far to draft rather than changing branching logic. SpecDec++ (Huang et al., 2024) learns an adaptive stopping policy to decide when further draft tokens bring minimal benefits. SVIP (Zhang et al., 2024) uses a lightweight draft confidence rule to terminate speculation early when rejections seem likely, to avoid wasted target forward pass work. FailFast (Pan et al., 2025) extends this by using dLLMs for rapid parallelized draft generation, and also rapidly increases draft length in high-acceptance regions to accept massive numbers of tokens. AdaSD (Lu et al., 2025) adapts draft length by using the divergence between target and draft distributions as a signal to reduce manual tuning.

Training-based tree policy.

Draft model language modeling performance is not highly correlated with speculative decoding performance (Yan et al., 2025). Thus, some works train draft models on objectives more aligned with speculative decoding acceptance. The earliest line of work in this regard, DistillSpec (Zhou et al., 2023), uses knowledge distillation to train the draft model to better align with the target model for Naive acceptance. Online speculative decoding (Liu et al., 2023) uses knowledge distillation to update the draft model concurrently with speculative decoding inference, dynamically clustering queries by common domain or topic to improve alignment. AdaSpec (Hu et al., 2025b) adapts speculative decoding to latency constraints in model serving, prioritizing response time over block efficiency. Group Tree Optimization (Hu et al., 2025a) enhances draft distillation by better aligning the single-path distillation objective with test-time tree-based decoding policies, using a draft tree reward based on expected NSS acceptance and a PPO-style objective that contrasts frozen and evolving draft trees. Our work differs from these in that it expands training specific to NSS and Naive objectives to more diverse verification methods, and in that we learn a compact tree selector model instead of finetuning a large draft model.

Hardware-Aware Tree Decoding.

Some works accelerate the batched forward pass or pick draft tree shapes that are highly compatible with compilers. Sequoia (Chen et al., 2024a) uses hardware-dependent target and draft times to inform its tree depth and branch selections. Yggdrasil (Guan et al., 2025) uses an equal-growth tree drafting algorithm to select tree width and depth and verification width to optimize latency and graph compiler optimizations, and improves speculative decoding stage scheduling for minimal GPU to CPU transfer. DeFT (Yao et al., 2024) uses high-utilization QKV grouping and custom kernels to accelerate tree decoding, which is highly relevant to the drafting phase. While we train our neural selector using draft and target pass times, and are hardware-aware like these works, our approach remains highly effective because it incorporates a diversity of verification objectives that these works do not.

3 Verification Algorithms

We now review prior verification algorithms. For the rest of paper, we denote the target model by MpM_{p} and the draft model by MqM_{q}, which share a vocabulary 𝒱\mathcal{V}. We use p(⋅|𝒄)p(\cdot|\boldsymbol{c}) and q(⋅|𝒄)q(\cdot|\boldsymbol{c}) to denote their next-token distributions over 𝒱\mathcal{V}. All of these algorithms require a formal notion of a draft tree, which we give below. Here, (𝒄1,𝒄2)(\boldsymbol{c}_{1},\boldsymbol{c}_{2}) denotes the concatenation of contexts. We ignore the root context 𝒄\boldsymbol{c} in our p,qp,q and node notation when it is implicitly clear.

Definition 3.1.

A draft tree rooted at 𝒄\boldsymbol{c} is a directed tree of nodes represented by distinct contexts 𝒄′\boldsymbol{c}^{\prime} with root node 𝒄\boldsymbol{c}, such that for each parent and child node pair (𝒄p,𝒄c)(\boldsymbol{c}_{p},\boldsymbol{c}_{c}), we have 𝒄c=(𝒄p,t)\boldsymbol{c}_{c}=(\boldsymbol{c}_{p},t) for some token t∈𝒱t\in\mathcal{V}. We denote the list of child nodes by ch​(𝒄)\text{ch}(\boldsymbol{c}), which can duplicate nodes111This does not follow the standard definition of a child node list in a tree, as child nodes have multiplicity and their order matters..

For a verification algorithm to preserve the target distribution, it must be compatible with drafting. We cover those which apply to single-path, multi-path, and other drafting regimes. Single-path means the tree is a path sampled from MqM_{q}, and multi-path is the union of i.i.d. single-paths.

Algorithm
Appears In
Multi-Path
OT-Based

NaiveTree
Leviathan et al. (2023)
✓
✓

NSS
Miao et al. (2024)
✓
✓

SpecTr
Sun et al. (2023)
✓
✓

SpecInfer
Miao et al. (2024)
✓
✓

Khisti
Khisti et al. (2025)
✓
✓

Naive
Chen et al. (2023)

TV
Hu and Huang (2024)

BV
Sun et al. (2024c)

Traversal
Weng et al. (2025)
✓

Table 1: Summary of all single-path and multi-path verification algorithms, and whether multi-path algorithms are OT-based.

3.1 Single-Path Algorithms

To the best of our knowledge, there are only three unique222All multi-path algorithms apply here, but all OT-based ones except NSS degenerate to Naive, and Traversal degenerates to BV. NSS degenerates to a simple algorithm: sample from pp until the trajectory no longer matches the draft block. verification algorithms that apply when the draft tree is an autoregressively sampled length LL block a1:L∼q(⋅|𝒄)a_{1:L}\sim q(\cdot|\boldsymbol{c}).

Naive speculative sampling.

Proposed by Chen et al. (2023); Leviathan et al. (2023), speculative sampling independently accepts each node a1:ia_{1:i} with probability min⁡(1,p​(ai|𝒄,a1:i−1)/q​(ai|𝒄,a1:i−1))\min(1,p(a_{i}|\boldsymbol{c},a_{1:i-1})/q(a_{i}|\boldsymbol{c},a_{1:i-1})). Then, it selects the maximal depth node a1:τa_{1:\tau} with all ancestor nodes accepted. Finally, it samples a correction token from a residual distribution, which is proportional to max{p(⋅|a1:τ)−q(⋅|a1:τ),0}\max\{p(\cdot|a_{1:\tau})-q(\cdot|a_{1:\tau}),0\} if τ<L\tau<L and p(⋅|a1:τ)p(\cdot|a_{1:\tau}) otherwise.

Tree verification.

Tree verification (Hu and Huang, 2024) provably improves upon naive speculative sampling by using Monte Carlo tree sampling to increase 𝔼​[τ]\mathbb{E}[\tau].

Block verification (BV).

Block verification (Sun et al., 2024c) provably improves upon naive speculative sampling by relaxing the requirement that all ancestors must be accepted. They recursively define node weights w​(a1:i)=min⁡(1,w​(a1:i−1)​p​(ai|a1:i−1)/q​(ai|a1:i−1))w(a_{1:i})=\min(1,w(a_{1:i-1})p(a_{i}|a_{1:i-1})/q(a_{i}|a_{1:i-1})) and independently accept each node a1:ia_{1:i} according to these weights. They return the maximal depth node a1:τa_{1:\tau} and sample the correction token from a ww-weighted naive residual.

As Sun et al. (2024c) show in their paper, block verification achieves the highest block efficiency among any single-path verification algorithm that only takes in pp and qq data on a1:La_{1:L}, so it is also provably better than tree verification.

3.2 Multi-Path Algorithms

Now, we consider draft trees which are the union of KK i.i.d. length LL paths a[k]1:L∼qL(⋅|𝒄)a[k]_{1:L}\sim q_{L}(\cdot|\boldsymbol{c}). When paths overlap, for every parent-child node pair (𝒄p,𝒄c)(\boldsymbol{c}_{p},\boldsymbol{c}_{c}) ,the multiplicity of 𝒄c\boldsymbol{c}_{c} in the child node list of 𝒄p\boldsymbol{c}_{p} is the number of times 𝒄c\boldsymbol{c}_{c} appears as a prefix of some a​[k]a[k]. Such a draft tree remains practical for speculative decoding due to GPU parallelization. For moderate KK and LL, there is little overhead in performing the batched forward pass across KK sequences, relative to a single forward pass (Agrawal et al., 2024; Dao et al., 2022).

The majority of multi-path algorithms use optimal transport linear program (OTLP) solvers, which is a specialized next-token predictor tailored to the draft distribution.

Definition 3.2.

An OTLP solver fp,q,k:𝒱k→𝒱f_{p,q,k}:\mathcal{V}^{k}\to\mathcal{V} on distributions p,q∈Δ​(𝒱)p,q\in\Delta(\mathcal{V}) with multiplicity k∈ℕk\in\mathbb{N} is a probabilistic function such that f​(X1,…,Xk)f(X_{1},\ldots,X_{k}) follows the pp distribution for i.i.d. X1,…,Xk∼qX_{1},\ldots,X_{k}\sim q.

OT-based algorithms.

Any OTLP solver can be used to perform a top-down traversal of the draft tree, starting at the root. Specifically, at each node 𝒄\boldsymbol{c}, taking in p(⋅|𝒄),q(⋅|𝒄)p(\cdot|\boldsymbol{c}),q(\cdot|\boldsymbol{c}) and the child node list ch​(𝒄)=[(𝒄,xi)]i=1k\text{ch}(\boldsymbol{c})=[(\boldsymbol{c},x_{i})]_{i=1}^{k}, an OT-based solver computes a feasible solution of an optimal transport linear program to probabilistically append a token tt to 𝒄\boldsymbol{c}, progressing to the child node (𝒄,t)(\boldsymbol{c},t) if t∈{x1,…,xk}t\in\{x_{1},\ldots,x_{k}\} and terminating otherwise. To the best of our knowledge, the unique OT-based algorithms in the i.i.d. multi-path setting are NSS (Miao et al., 2024), SpecInfer (Miao et al., 2024), SpecTr (Sun et al., 2023), and Khisti (Khisti et al., 2025). Also, while naive speculative sampling (Leviathan et al., 2023; Chen et al., 2023) was originally presented as a single-path algorithm, it is OT-based and can also be applied to multi-path methods: to distinguish this from Naive, we call it NaiveTree. We review these details in Appendix B.

Traversal verification.

Weng et al. (2025) gives the only multi-path algorithm that traverses the draft tree from the bottom-up. When K=1K=1, this reduces to block verification.

We stress that these algorithms may outperform each other in different (p,q)(p,q) regimes, which inherently depend on the model, dataset, and pp-sampling setting. On the OT-based side, recent work (Hu et al., 2025c; Thomas and Pal, 2025) has computed the acceptance rate of an optimal OTLP solver or a near-optimal OTLP solver in certain cases, so a fully optimal OT-based remains infeasible. To the best of our knowledge, there are no existing comparisons between traversal verification and all OT-based methods.

3.3 Other Algorithms

In the deterministic tree setting, only NSS (Miao et al., 2024) works, so it is used in works like EAGLE (Li et al., 2024b). Some works also mix i.i.d. and deterministic constructions. For example, Hu et al. (2025c) devise an OT-based algorithm when the top-(k−1)(k-1) draft tokens are chosen deterministically and the last is selected randomly without replacement, with SpecHub (Sun et al., 2024b) having prior considered the case k=2k=2. We leave exploration of these methods in the context of dynamic drafting to future work.

Table 2: Average block efficiency across datasets and sampling configurations of existing verification algorithms. For more detailed data, see Appendix F.

Method
Qwen
Gemma
Llama
Average

NSS
4.44
1.99
5.72
4.05

BV
4.24
3.30
5.37
4.30

Khisti
4.90
2.05
6.29
4.41

NaiveTree
5.00
2.01
6.50
4.50

Naive
4.60
3.44
5.52
4.52

SpecInfer
5.13
2.07
6.55
4.58

SpecTr
5.11
2.05
6.67
4.61

Traversal
5.33
3.81
6.78
5.31

4 Comparison of Existing Verification Algorithms

In this section, we compare the verification algorithms mentioned in the previous sections. We include two single-path algorithms: original (naive) speculative sampling and block verification (BV); and 6 multi-path algorithms: NSS, NaiveTree, SpecTr, SpecInfer, Khisti, and Traversal.

4.1 Experimental Setup

We perform our experiments on three target-draft model pairs: Llama-3 70B/8B-Instruct (Grattafiori et al., 2024), Gemma-3 27B/270M-IT (Team et al., 2025b), and Qwen-2.5 32B/0.5B-Instruct (Qwen et al., 2025). Our selection covers three different model families, target model sizes, and different magnitudes of target to draft size ratios (∼\sim9:1, 100:1, and 64:1 respectively).

We conduct our experiments on 5 datasets: MATH500 (Lightman et al., 2023), OlympiadBench (He et al., 2024), LiveCodeBench (Jain et al., 2024), LitBench (Fein et al., 2025), and Opus (Zhang et al., 2020; Tiedemann, 2012). In LiveCodeBench, we use the ‘code_generation_lite’ subset; for Opus, we select half of our prompts each from ‘opus_books‘ and ‘open_subtitles‘ with a uniform split of translations between English, French, Spanish, and Italian. Each of these datasets covers a different generative setting. MATH500 and OlympiadBench both cover math, but at differing levels of difficulty; LiveCodeBench is focused on coding; we use the prompts from LitBench to test creative writing; and we use Opus to test performance for translation.

We evaluate methods on 50 prompts per dataset. We test on 8 different sampling configurations: sampling from MpM_{p} with temperatures {0.2,0.4,0.6,0.8,1.0,1.2}\{0.2,0.4,0.6,0.8,1.0,1.2\} and no nucleus sampling, and sampling from MpM_{p} with temperature 1.0 and nucleus sampling with {0.9,0.99}\{0.9,0.99\}. We evaluate throughput on machines with two A100-80GB GPUs and an Intel Xeon Gold 6342 CPU (12 cores, 2.80 GHz).

We focus on two important metrics for comparing speculative decoding methods: block efficiency and throughput. Block efficiency is the average number of accepted tokens per speculated block; the greater this number, the fewer target forward passes are required. Block efficiency improves as nodes are added to a tree; however, speedups in practice tend instead to follow a U-curve, as larger trees slow down both drafting and target model forward passes. As such, throughput, measured in tokens per second, is more relevant in deployment. However, its shortcoming is that it is highly sensitive to system-level parameters, such as GPU speed/memory, inference engines, and attention kernels.

Table 3: Average throughput (tokens per second) across datasets and sampling configurations of existing verification algorithms. For more detailed data, see Appendix F.

Method
Qwen
Gemma
Llama
Average

NSS
16.88
5.94
12.91
11.91

Naive
16.55
9.20
11.58
12.44

Khisti
18.67
5.93
13.66
12.75

NaiveTree
19.16
6.00
14.71
13.29

SpecTr
19.01
6.09
14.86
13.32

BV
17.30
10.57
12.20
13.36

SpecInfer
19.58
6.21
14.86
13.55

Traversal
21.35
11.56
14.77
15.89

4.2 Results

Our results comparing existing verification algorithms are summarized in Table 2 and Table 3. These tables average across all datasets and sampling configurations per model; the detailed breakdowns are in Appendix F. In reporting our results, for each set of sampling parameters and dataset, we select the branching factor K∈[1,4]K\in[1,4] and block length L∈[0,8]L\in[0,8] that maximizes the block-efficiency or throughput.

We see that Traversal is the best-performing verification algorithm in both block efficiency and throughput, outperforming the second-best method in each by ∼15%\sim 15\%. Traversal shows particularly strong performance on Gemma, but is also the top performer for Qwen and only slightly below the best throughput for Llama. Further analysis of full results in Appendix F shows that Traversal is consistently the best algorithm in both throughput and block efficiency across all sampling configurations tested. We hypothesize this occurs because the top-down approach in OT-based methods makes deep traversal in tree exponentially difficult, whereas the bottom-up approach in Traversal starts at leaf nodes and has a higher chance of accepting longer sequences.

After Traversal, we find that SpecTr, BV, and SpecInfer all perform similarly in average throughput. NSS performs worst, which is expected because it does not use draft probabilities to guide verification.

5 Delayed Tree Expansion

While our results in Section 4 show Traversal significantly outperforms OT-based methods in all settings, interestingly, there is no dominant winner among OT-based methods. For throughput, SpecInfer wins for Qwen, Naive wins for Gemma, and SpecTr and SpecInfer tie on Llama, with Khisti not lagging far behind. To explain this phenomenon, we examine acceptance rates of OT-based methods at tree nodes.

The acceptance rate measures how often an OT-based method appends a token that remains on the draft tree. This incorporates the randomness of draft tokens, so it does not actually depend on the observed child nodes, rather only on pp and qq. We show how to compute these rates in Appendix C.

Definition 5.1.

For any OTLP solver fp,q,kf_{p,q,k}, its acceptance rate α​(fp,q,k)=ℙ​(f​(X1,…,Xk)∈{X1,…,Xk})\alpha(f_{p,q,k})=\mathbb{P}(f(X_{1},\ldots,X_{k})\in\{X_{1},\ldots,X_{k}\}) is the probability that the OTLP solver output token lies among its input tokens, over i.i.d. input samples X1,…,Xk∼qX_{1},\ldots,X_{k}\sim q.

We employ the same experimental setup as in Section 4, but conduct our analysis on offline trees generated from over 200,000200,000 fixed-spaced draft roots along fixed target model trajectories. This allows us to use vLLM (Kwon et al., 2023) to speed up data generation. Our aggregate results across all 88 sampling settings for LiveCodeBench, MATH500, and OlympiadBench on Llama are shown in Figure 1, and we also include L1 distance between target and draft distributions.

Figure 1: We generate 200,000+ draft (Llama-3 8B-Instruct) trees from roots of target model (Llama-3 70B-Instruct) trajectories and compute both L1 target-draft distance and average OTLP acceptances across varying draft tree depths. The divergence between target and draft distributions spikes deeper in the tree, and acceptances across all OTLP methods consistently decrease with depth.

We directly observe for all OT-based methods that node-wise acceptance rate rapidly degrade with depth. In other words, while we branch early in the tree often, the acceptance rate improvements later in the tree are often beneficial. This is directly explained by the fact that L1 target-draft deviations increase with depth. For example, the Naive acceptance rate for k=1k=1 linearly decreases with L1 distance.

Motivated by these findings, we define a delayed tree, which drafts a partial path and then expands. OTLP solvers can still be used on such a tree while preserving the target distribution, because conditioned on their current node, their output token follows the pp distribution.

Definition 5.2.

Fix K,L1,L2∈ℕK,L_{1},L_{2}\in\mathbb{N} with K≥1K\geq 1. A random (K,L1,L2)(K,L_{1},L_{2})-delayed-tree at context 𝒄\boldsymbol{c} is formed by sampling a length-L1L_{1} path a1:L1∼q(⋅|𝒄)a_{1:L_{1}}\sim q(\cdot|\boldsymbol{c}) and then branching into KK i.i.d. length-L2L_{2} paths a1:L2(k)∼q(⋅|𝒄,a1:L1)a^{(k)}_{1:L_{2}}\sim q(\cdot|\boldsymbol{c},a_{1:L_{1}}). Child node lists can have duplicates if paths overlap after the branched node (𝒄,a1:L1)(\boldsymbol{c},a_{1:L_{1}}), but not before.

Now that we have motivated delayed expansion, we discuss its evaluation. How do we estimate the block efficiency of an OT-based method on a delayed tree, when both drafting and verification involve randomness? Motivated by adjacent work in adaptive drafting (Chen et al., 2024a; Brown et al., 2024), we define the notion of branching probability for an OTLP solver.

Definition 5.3.

For any OTLP solver fp,q,kf_{p,q,k}, list of kk tokens 𝒙\boldsymbol{x}, and token t∈𝒱t\in\mathcal{V}, the branching probability to tt, B​(fp,q,k,𝒙,t)B(f_{p,q,k},\boldsymbol{x},t), is the probability that fp,q,k​(𝒙)=tf_{p,q,k}(\boldsymbol{x})=t.

This can be used to compute expected block efficiency by following the law of conditional expectation to separate draft tree randomness and verification randomness: the outer expectation is taken over random (K,L1,L2)(K,L_{1},L_{2})-delayed-trees 𝒯\mathcal{T}, the inner expectation is taken over the randomness present in the OTLP solver, and p,qp,q in fp,q,kf_{p,q,k} represent the implied node distributions p(⋅|𝒄,t1,…,tj−1)p(\cdot|\boldsymbol{c},t_{1},\ldots,t_{j-1}) and q(⋅|𝒄,t1,…,tj−1)q(\cdot|\boldsymbol{c},t_{1},\ldots,t_{j-1}). The last equality holds from the fact that OT-based methods iteratively append the output of fp,q,kf_{p,q,k} to progress down the draft tree.

𝔼​[τ+1]=𝔼𝒯​[𝔼​[τ+1|𝒯]]=\displaystyle\mathbb{E}[\tau+1]=\mathbb{E}_{\mathcal{T}}\left[\mathbb{E}[\tau+1|\mathcal{T}]\right]=

(1)

𝔼𝒯​[∑𝒄′∈𝒯ℙ​(OTLP solver reaches ​𝒄′|𝒯)]=\displaystyle\mathbb{E}_{\mathcal{T}}\left[\sum_{\boldsymbol{c}^{\prime}\in\mathcal{T}}\mathbb{P}(\text{OTLP solver reaches }\boldsymbol{c}^{\prime}|\mathcal{T})\right]=

(2)

𝔼𝒯​[∑(𝒄,t1,…,td)∈𝒯∏j=1dB​(fp,q,k,ch​(𝒄,t1,…,tj−1),tj)]\displaystyle\mathbb{E}_{\mathcal{T}}\left[\sum_{\begin{subarray}{c}(\boldsymbol{c},t_{1},\ldots,\\
t_{d})\in\mathcal{T}\end{subarray}}\prod_{j=1}^{d}B\left(f_{p,q,k},\text{ch}(\boldsymbol{c},t_{1},\ldots,t_{j-1}),t_{j}\right)\right]

(3)

In practice, we estimate this quantity by taking a simple average of the inner sum over s=4s=4 i.i.d. delayed tree samples 𝒯1,…,𝒯s\mathcal{T}_{1},\ldots,\mathcal{T}_{s}. This estimator does not eliminate drafting variance, but eliminates verification variance and is unbiased.

6 Neural Delay-and-Branch Predictor

Delayed tree expansion, as described in Section 5, introduces discrete design choices that can strongly influence end-to-end throughput. For a fixed verification procedure, the throughput of a delayed expansion policy is a tradeoff between two variables: acceptance, representing how many tokens in the draft tree are verified, and latency, which is the combined cost of drafting and the target pass. Because the divergence between the pp and qq distributions varies significantly across contexts, a single static configuration of all three parameters (K,L1,L2)(K,L_{1},L_{2}) is not necessarily optimal across all requests. Therefore, we introduce a context-conditioned parameter selector that predicts which delayed expansion parameters to use at each decoding step.

Formally, at each decoding step with root context 𝒄\boldsymbol{c}, our neural selector chooses the parameters a=(K,L1,L2)∈𝒜a=(K,L_{1},L_{2})\in\mathcal{A}, where 𝒜={1,2,3,4}×{0,…,8}2\mathcal{A}=\{1,2,3,4\}\times\{0,\ldots,8\}^{2} is the action space representing all supported configurations of the parameters described in Section 5. After choosing aa, we draft the corresponding delayed tree, run a target forward pass, and carry out verification. Ultimately, we choose aa to maximize a throughput objective which trades off between block efficiency and latency (Equation 9). As 𝒜\mathcal{A} is discrete and small, we implement π\pi as a categorical policy.

Since the selector must choose aa before constructing the draft tree, we restrict it to the following features that are readily available at the root, described in Appendix E: (i) draft and target model hidden states at the root and draft states at the previous token, (ii) scalar uncertainty and divergence statistics between pp and qq, such as entropy and KL divergence, (iii) local sampling parameters and context length, and (iv) latency estimates for draft and target forward passes.

We design this policy from a lightweight MLP which independently projects hidden state inputs to a shared dimension d=128d=128, applies layer normalization, concatenates standardized scalar features, and applies a two-layer MLP to produce logits over supported actions. To train the selector as described above, we construct a target signal for each action from an offline dataset of speculative decoding traces, taking a root every 16 tokens. For each root 𝒄\boldsymbol{c} and action aa, we store an empirical estimate E^​[τ​(𝒄,a)+1]\widehat{E}[\tau(\boldsymbol{c},a)+1] of block efficiency for an aa-delayed-tree at 𝒄\boldsymbol{c} following Equation 3. To incorporate throughput considerations, we estimate the runtime for each action using latency measurements derived from a microbenchmark ”warm-up run”, then approximate the total wall-clock time for all necessary forward passes T^\hat{T} as shown in Equation 11.

Given the logits zθ​(xi)z_{\theta}(x_{i}), we can then define the policy’s per-sample offline throughput estimate at train time as the ratio of expected block efficiency to expected time:

TPS^π​(𝒄)=∑a∈𝒜πθ​(a|𝒄)​E^​[τ​(𝒄,a)+1]∑a∈𝒜πθ​(a|𝒄)​T^​(𝒄,a).\widehat{\text{TPS}}_{\pi}(\boldsymbol{c})=\frac{\sum_{a\in\mathcal{A}}{\pi_{\theta}(a|\boldsymbol{c})\widehat{E}[\tau(\boldsymbol{c},a)+1]}}{\sum_{a\in\mathcal{A}}{\pi_{\theta}(a|\boldsymbol{c})\widehat{T}(\boldsymbol{c},a)}}.

(4)

For each training example, we also have a static baseline configuration aib​a​s​ea_{i}^{base} associated with its sampling parameters (temperature and nucleus threshold). Then, we optimize a baseline-aware objective by minimizing

−log​TPS^π​(𝒄)TPS^b​a​s​e​(𝒄).-\text{log}\frac{\widehat{\text{TPS}}_{\pi}(\boldsymbol{c})}{\widehat{\text{TPS}}_{base}(\boldsymbol{c})}.

(5)

In addition to this primary term in the objective, we add the penalty term described in Appendix E which encourages the selector to avoid severe regressions below the baseline, even if they occur infrequently.

At inference time, the selector produces πθ(⋅∣𝒄)\pi_{\theta}(\cdot\mid\boldsymbol{c}) once per decoding step using the root-level features. We then select the action with the highest assigned probability argmaxa∈𝒜​πθ​(a∣𝒄)\text{argmax}_{a\in\mathcal{A}}\pi_{\theta}(a\mid\boldsymbol{c}), which corresponds to the (K,L1,L2)(K,L_{1},L_{2}) parameters necessary for delayed expansion, before proceeding with the drafting and verification processes.

6.1 Experiments

We now evaluate the effectiveness of our neural selector. We follow a similar experimental setup to that described in Section 4.1. We train our parameter selector on offline data generated from 350 prompts per dataset, and measure its performance on a held-out test set.

First, we compare the efficacy of our neural selector (NDE) to the baseline algorithms. The results of this are shown in Table 4 and Table 5. We compute the ratio of improvement in block efficiency and TPS per verification method. We see that NDE improves on the baseline algorithms by around 10%10\% and 25%25\% in block efficiency and throughput on average, across the models tested. Significant improvements are obtained in particular on Gemma. We speculate that this is because our Gemma target/draft model pair has the highest ratio of size difference, and therefore likely has the most divergent proposal distribution, which our neural selector is able to navigate by dynamic parameter selection.

We then compare the best performing existing verification method, Traversal, to our new algorithms. The results of this are shown in Table 6 and Table 7. We see that the NDE is effective at reducing the gap of the OT methods to Traversal; and now, SpecInfer NDE is able to outperform Traversal in throughput by ∼5%\sim 5\% on average, achieving a new state-of-the-art.

Table 4: Block efficiency ratio improvement of our NDE (neural dynamic expansion) approach over the baseline algorithms.

Method
Qwen
Gemma
Llama
Average

Khisti NDE
1.01
1.90
1.09
1.19

NaiveTree NDE
1.07
0.99
1.17
1.09

NSS NDE
0.97
1.68
1.01
1.11

SpecInfer NDE
0.98
1.65
1.01
1.10

SpecTr NDE
0.97
1.68
1.01
1.10

Table 5: Throughput (tokens per second) ratio improvement of our NDE (neural dynamic expansion) approach over the baseline algorithms.

Method
Qwen
Gemma
Llama
Average

Khisti NDE
0.99
2.17
0.97
1.16

NaiveTree NDE
1.31
1.40
1.21
1.30

NSS NDE
1.20
2.09
0.96
1.26

SpecInfer NDE
1.15
2.14
0.96
1.23

SpecTr NDE
1.14
2.17
0.96
1.23

Table 6: Average block efficiency across datasets and sampling configurations. Traversal is best existing algorithm; NDE (neural delayed expansion) is ours. For more detailed data, see Appendix F.

Method
Qwen
Gemma
Llama
Average

Traversal
5.33
3.81
6.78
5.31

NSS NDE
4.31
3.34
5.78
4.48

NaiveTree NDE
4.94
3.42
6.47
4.94

SpecInfer NDE
5.03
3.42
6.64
5.03

SpecTr NDE
4.97
3.44
6.74
5.05

Khisti NDE
4.97
3.89
6.84
5.23

Table 7: Average throughput (tokens per second) across datasets and sampling configurations. Traversal is best existing algorithm; NDE (neural delayed expansion) is ours. For more detailed data, see Appendix F.

Method
Qwen
Gemma
Llama
Average

Traversal
21.35
11.56
14.77
15.89

Khisti NDE
18.41
12.84
13.24
14.83

NSS NDE
20.29
12.40
12.45
15.05

NaiveTree NDE
21.67
12.86
14.01
16.18

SpecTr NDE
21.74
13.22
14.23
16.40

SpecInfer NDE
22.54
13.26
14.27
16.69

7 Conclusion

We conducted the first systematic analysis of all i.i.d. verification algorithms in multi-path speculative decoding. We discovered that Traversal Verification significantly outperforms all other methods in our evaluation, which are OT-based. By analyzing target-draft divergence and tree node acceptance rates, we found this was caused by aggressive branching early in the draft tree, which wastes compute on minimal acceptance gains. To overcome this shortcoming, we introduce the idea of dynamic delayed draft tree expansion, and trained a neural selector on each OT-based method to optimally select when and how to expand i.i.d. rollouts in a tree based on a corresponding acceptance length objective. With this neural selector, we demonstrated that OT-based methods can outperform Traversal Verification. Future work could explore a similar neural selector for Traversal Verification, or consider more robust training objectives for expected acceptance length.

8 Impact Statement

This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.

References

A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee (2024)
Taming throughput-latency tradeoff in llm inference with sarathi-serve.

In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24),

External Links: Link

Cited by: §3.2.

Z. Ankner, R. Parthasarathy, A. Nrusimha, C. Rinard, J. Ragan-Kelley, and W. Brandon (2024)
Hydra: sequentially-dependent draft heads for medusa decoding.

arXiv preprint arXiv:2402.05109.

Cited by: Appendix A.

J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. (2023)
Qwen technical report.

arXiv preprint arXiv:2309.16609.

Cited by: §1.

O. Brown, Z. Wang, A. Do, N. Mathew, and C. Yu (2024)
Dynamic depth decoding: faster speculative decoding for llms.

arXiv preprint arXiv:2409.00142.

Cited by: §1,
§1,
§1,
§2.2,
§5.

T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao (2024)
Medusa: simple llm inference acceleration framework with multiple decoding heads.

arXiv preprint arXiv:2401.10774.

Cited by: Appendix A.

C. Chen, S. Borgeaud, G. Irving, J. Lespiau, L. Sifre, and J. Jumper (2023)
Accelerating large language model decoding with speculative sampling.

arXiv preprint arXiv:2302.01318.

Cited by: §B.2,
§1,
§2.2,
§3.1,
§3.2,
Table 1.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. Pinto, J. Kaplan, et al. (2021)
Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374.

Cited by: §1.

Z. Chen, A. May, R. Svirschevski, Y. Huang, M. Ryabinin, Z. Jia, and B. Chen (2024a)
Sequoia: scalable, robust, and hardware-aware speculative decoding.

arXiv preprint arXiv:2402.12374.

Cited by: §1,
§1,
§1,
§2.2,
§2.2,
§5.

Z. Chen, X. Yang, J. Lin, C. Sun, K. Chang, and J. Huang (2024b)
Cascade speculative drafting for even faster llm inference.

Advances in Neural Information Processing Systems 37, pp. 86226–86242.

Cited by: Appendix A.

T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré (2022)
FlashAttention: fast and memory-efficient exact attention with io-awareness.

In Advances in Neural Information Processing Systems (NeurIPS 2022),

External Links: Link

Cited by: §3.2.

M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman, et al. (2024)
LayerSkip: enabling early exit inference and self-speculative decoding.

arXiv preprint arXiv:2404.16710.

Cited by: Appendix A.

D. Fein, S. Russo, V. Xiang, K. Jolly, R. Rafailov, and N. Haber (2025)
LitBench: a benchmark and dataset for reliable evaluation of creative writing.

External Links: 2507.00769,
Link

Cited by: §4.1.

Y. Fu, P. Bailis, I. Stoica, and H. Zhang (2024)
Break the sequential dependency of llm inference using lookahead decoding.

arXiv preprint arXiv:2402.02057.

Cited by: Appendix A,
§1.

F. Gloeckle, B. Y. Idrissi, B. Rozière, D. Lopez-Paz, and G. Synnaeve (2024)
Better & faster large language models via multi-token prediction.

arXiv preprint arXiv:2404.19737.

Cited by: Appendix A.

A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)
The llama 3 herd of models.

arXiv preprint arXiv:2407.21783.

Cited by: §1,
§4.1.

Y. Guan, C. Yu, S. Fang, W. Hu, Z. Pan, Z. Wang, Z. Liu, Y. Zhou, Y. Ding, M. Guo, et al. (2025)
Yggdrasil: bridging dynamic speculation and static runtime for latency-optimal tree-based llm decoding.

arXiv preprint arXiv:2512.23858.

Cited by: §1,
§1,
§2.2.

C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun (2024)
OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.

External Links: 2402.14008

Cited by: §4.1.

Z. He, Z. Zhong, T. Cai, J. D. Lee, and D. He (2023)
Rest: retrieval-based speculative decoding.

arXiv preprint arXiv:2311.08252.

Cited by: Appendix A.

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2021)
Measuring mathematical problem solving with the math dataset.

arXiv preprint arXiv:2103.03874.

Cited by: §1.

D. Hendrycks and K. Gimpel (2023)
Gaussian error linear units (gelus).

External Links: 1606.08415,
Link

Cited by: Appendix E.

S. Hu, J. Li, Z. Lu, and P. Zhou (2025a)
Bridging draft policy misalignment: group tree optimization for speculative decoding.

arXiv preprint arXiv:2509.22134.

Cited by: §2.2.

Y. Hu, J. Guo, X. Feng, and T. Zhao (2025b)
AdaSPEC: selective knowledge distillation for efficient speculative decoders.

arXiv preprint arXiv:2510.19779.

Cited by: §2.2.

Z. Hu and H. Huang (2024)
Accelerated speculative sampling based on tree monte carlo.

In Forty-first International Conference on Machine Learning,

Cited by: §3.1,
Table 1.

Z. Hu, T. Zheng, V. Viswanathan, Z. Chen, R. A. Rossi, Y. Wu, D. Manocha, and H. Huang (2025c)
Towards optimal multi-draft speculative decoding.

arXiv preprint arXiv:2502.18779.

Cited by: §3.2,
§3.3.

K. Huang, X. Guo, and M. Wang (2024)
Specdec++: boosting speculative decoding via adaptive candidate lengths.

arXiv preprint arXiv:2405.19715.

Cited by: §1,
§1,
§2.2.

B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Lu, et al. (2024)
Qwen2. 5-coder technical report.

arXiv preprint arXiv:2409.12186.

Cited by: §1.

N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica (2024)
LiveCodeBench: holistic and contamination free evaluation of large language models for code.

arXiv preprint arXiv:2403.07974.

Cited by: §4.1.

E. Kasneci, K. Seßler, S. Küchemann, M. Bannert, D. Dementieva, F. Fischer, U. Gasser, G. Groh, S. Günnemann, E. Hüllermeier, S. Krusche, G. Kutyniok, T. Michaeli, C. Nerdel, J. Pfeffer, O. Poquet, M. Sailer, A. Schmidt, T. Seidel, and G. Kasneci (2023)
ChatGPT for good? on opportunities and challenges of large language models for education.

Learning and Individual Differences 103, pp. 102274.

External Links: Document

Cited by: §1.

A. J. Khisti, M. Ebrahimi, H. Dbouk, A. Behboodi, R. Memisevic, and C. Louizos (2025)
Multi-draft speculative sampling: canonical decomposition and theoretical limits.

In The Thirteenth International Conference on Learning Representations,

Cited by: §B.5,
§1,
§3.2,
Table 1.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica (2023)
Efficient memory management for large language model serving with pagedattention.

In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,

Cited by: §2.1,
§5.

Y. Leviathan, M. Kalman, and Y. Matias (2023)
Fast inference from transformers via speculative decoding.

In International Conference on Machine Learning,

pp. 19274–19286.

Cited by: §B.2,
§1,
§2.2,
§3.1,
§3.2,
Table 1.

Y. Li, F. Wei, C. Zhang, and H. Zhang (2024a)
Eagle-2: faster inference of language models with dynamic draft trees.

arXiv preprint arXiv:2406.16858.

Cited by: §1,
§1,
§2.2.

Y. Li, F. Wei, C. Zhang, and H. Zhang (2024b)
Eagle: speculative sampling requires rethinking feature uncertainty.

arXiv preprint arXiv:2401.15077.

Cited by: Appendix A,
§3.3.

Y. Li, F. Wei, C. Zhang, and H. Zhang (2025)
Eagle-3: scaling up inference acceleration of large language models via training-time test.

arXiv preprint arXiv:2503.01840.

Cited by: Appendix A.

H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023)
Let’s verify step by step.

arXiv preprint arXiv:2305.20050.

Cited by: §4.1.

F. Liu, Y. Tang, Z. Liu, Y. Ni, D. Tang, K. Han, and Y. Wang (2024a)
Kangaroo: lossless self-speculative decoding for accelerating llms via double early exiting.

Advances in Neural Information Processing Systems 37, pp. 11946–11965.

Cited by: Appendix A.

T. Liu, Y. Li, Q. Lv, K. Liu, J. Zhu, W. Hu, and X. Sun (2024b)
Pearl: parallel speculative decoding with adaptive draft length.

arXiv preprint arXiv:2408.11850.

Cited by: Appendix A,
§1,
§1.

X. Liu, L. Hu, P. Bailis, A. Cheung, Z. Deng, I. Stoica, and H. Zhang (2023)
Online speculative decoding.

arXiv preprint arXiv:2310.07177.

Cited by: §2.2.

X. Liu, J. Yu, J. Park, I. Stoica, and A. Cheung (2025)
Speculative decoding: performance or illusion?.

arXiv preprint arXiv:2601.11580.

Cited by: §2.1.

K. Lu, D. Hong, and P. Liu (2025)
AdaSD: adaptive speculative decoding for efficient language model inference.

arXiv preprint arXiv:2512.11280.

Cited by: §2.2.

Z. Ma, I. Gim, and L. Zhong (2025)
Cacheback: speculative decoding with nothing but cache.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,

pp. 31067–31072.

Cited by: Appendix A.

X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, et al. (2024)
Specinfer: accelerating large language model serving with tree-based speculative inference and verification.

In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3,

pp. 932–949.

Cited by: §B.1,
§B.4,
§1,
§2.2,
§3.2,
§3.3,
Table 1,
Table 1,
footnote 3.

R. Pan, Z. Chen, and R. Netravali (2025)
Fail fast, win big: rethinking the drafting strategy in speculative decoding via diffusion llms.

arXiv preprint arXiv:2512.20573.

Cited by: §2.2.

Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu (2025)
Qwen2.5 technical report.

External Links: 2412.15115,
Link

Cited by: §4.1.

M. Samragh, A. Kundu, D. Harrison, K. Nishu, D. Naik, M. Cho, and M. Farajtabar (2025)
Your llm knows the future: uncovering its multi-token prediction potential.

arXiv preprint arXiv:2507.11851.

Cited by: Appendix A.

Y. Shen, J. Shen, Q. Kong, T. Liu, Y. Lu, and C. Wang (2025)
Speculative decoding via hybrid drafting and rollback-aware branch parallelism.

arXiv preprint arXiv:2506.01979.

Cited by: Appendix A.

M. Stern, N. Shazeer, and J. Uszkoreit (2018)
Blockwise parallel decoding for deep autoregressive models.

Advances in Neural Information Processing Systems 31.

Cited by: Appendix A.

H. Sun, Z. Chen, X. Yang, Y. Tian, and B. Chen (2024a)
Triforce: lossless acceleration of long sequence generation with hierarchical speculative decoding.

arXiv preprint arXiv:2404.11912.

Cited by: Appendix A.

R. Sun, T. Zhou, X. Chen, and L. Sun (2024b)
Spechub: provable acceleration to multi-draft speculative decoding.

arXiv preprint arXiv:2411.05289.

Cited by: §3.3.

Z. Sun, U. Mendlovic, Y. Leviathan, A. Aharoni, J. H. Ro, A. Beirami, and A. T. Suresh (2024c)
Block verification accelerates speculative decoding.

arXiv preprint arXiv:2403.10444.

Cited by: §1,
§3.1,
§3.1,
Table 1.

Z. Sun, A. T. Suresh, J. H. Ro, A. Beirami, H. Jain, and F. Yu (2023)
Spectr: fast speculative decoding via optimal transport.

Advances in Neural Information Processing Systems 36, pp. 30222–30242.

Cited by: §B.3,
§1,
§3.2,
Table 1.

G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, et al. (2025a)
Gemma 3 technical report.

arXiv preprint arXiv:2503.19786.

Cited by: §1.

G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, L. Rouillard, T. Mesnard, G. Cideron, J. Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot, I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai, A. Tsitsulin, R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y. Gao, B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J. Peter, D. Sinopalnikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin, R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. Steiner, A. Friesen, A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade, A. Feng, A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. György, A. S. Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy, A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen, C. L. Lan, C. A. Choquette-Choo, C. Carey, C. Brick, D. Deutsch, D. Eisenbud, D. Cattle, D. Cheng, D. Paparas, D. S. Sreepathihalli, D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov, F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi, H. Klimczak-Plucińska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne, I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan, J. Ji, J. Singh, K. Black, K. Yu, K. Hui, K. Vodrahalli, K. Greff, L. Qiu, M. Valentine, M. Coelho, M. Ritter, M. Hoffman, M. Watson, M. Chaturvedi, M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev, N. Chauhan, N. Sachdeva, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein, P. Culliton, P. Schmid, P. G. Sessa, P. Xu, P. Stanczyk, P. Tafti, R. Shivanna, R. Wu, R. Pan, R. Rokni, R. Willoughby, R. Vallu, R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth, S. Põder, S. Bhatnagar, S. R. Panyam, S. Eiger, S. Zhang, T. Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci, V. Misra, V. Roseberry, V. Feinberg, V. Kolesnikov, W. Han, W. Kwon, X. Chen, Y. Chow, Y. Zhu, Z. Wei, Z. Egyed, V. Cotruta, M. Giang, P. Kirk, A. Rao, K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins, O. Sanseviero, L. Gonzalez, Z. Gleicher, T. Warkentin, V. Mirrokni, E. Senter, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y. Matias, D. Sculley, S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J. Alayrac, R. Anil, Dmitry, Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev, C. Hardin, R. Dadashi, and L. Hussenot (2025b)
Gemma 3 technical report.

External Links: 2503.19786,
Link

Cited by: §4.1.

G. Team (2024)
Gemma 2: improving open language models at a practical size.

arXiv preprint arXiv:2408.00118.

Cited by: §1.

R. K. Thomas and A. Pal (2025)
Global resolution: optimal multi-draft speculative sampling via convex minimization.

arXiv preprint arXiv:2511.15898.

Cited by: §3.2.

J. Tiedemann (2012)
Parallel data, tools and interfaces in OPUS.

In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12), N. Calzolari, K. Choukri, T. Declerck, M. U. Doğan, B. Maegaard, J. Mariani, A. Moreno, J. Odijk, and S. Piperidis (Eds.),

Istanbul, Turkey, pp. 2214–2218.

External Links: Link

Cited by: §4.1.

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, N. Goyal, E. Hambro, H. Azhar, A. Rodriguez, et al. (2023a)
Llama 2: open foundation and fine-tuned chat models.

arXiv preprint arXiv:2310.11387.

Cited by: §1.

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, H. Azhar, et al. (2023b)
LLaMA: open and efficient foundation language models.

arXiv preprint arXiv:2302.13971.

Cited by: §1.

J. Wang, Y. Su, J. Li, Q. Xia, Z. Ye, X. Duan, Z. Wang, and M. Zhang (2025)
Opt-tree: speculative decoding with adaptive draft tree structure.

Transactions of the Association for Computational Linguistics 13, pp. 188–199.

Cited by: §1,
§1,
§2.2.

Y. Weng, Q. Hu, X. Chen, L. Liu, D. Mei, H. Qiu, J. Tian, and Z. Shi (2025)
Traversal verification for speculative tree decoding.

arXiv preprint arXiv:2505.12398.

Cited by: §1,
§3.2,
Table 1.

H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui (2024)
Unlocking efficiency in large language model inference: a comprehensive survey of speculative decoding.

arXiv preprint arXiv:2401.07851.

Cited by: §2.1.

Y. Xiong, R. Zhang, Y. Li, and L. Zou (2025)
DySpec: faster speculative decoding with dynamic token tree structure.

World Wide Web 28 (3), pp. 36.

Cited by: §1,
§1,
§2.2.

M. Yan, S. Agarwal, and S. Venkataraman (2025)
Decoding speculative decoding.

In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),

pp. 6460–6473.

Cited by: §2.2.

J. Yao, K. Zhang, K. Chen, J. You, Z. Wang, B. Yuan, and T. Lin (2024)
DeFT: flash tree-attention with io-awareness for efficient tree-search-based llm inference.

In ICLR 2024 Workshop: How Far Are We From AGI,

Cited by: §2.2.

B. Zhang, P. Williams, I. Titov, and R. Sennrich (2020)
Improving massively multilingual neural machine translation and zero-shot translation.

In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault (Eds.),

Online, pp. 1628–1639.

External Links: Link,
Document

Cited by: §4.1.

J. Zhang, J. Wang, H. Li, L. Shou, K. Chen, G. Chen, and S. Mehrotra (2023)
Draft & verify: lossless large language model acceleration via self-speculative decoding.

arXiv preprint arXiv:2309.08168.

Cited by: Appendix A.

Z. Zhang, J. Xu, T. Liang, X. Chen, Z. He, R. Wang, and Z. Tu (2024)
Draft model knows when to stop: a self-verification length policy for speculative decoding.

arXiv preprint arXiv:2411.18462.

Cited by: §2.2.

W. Zhao, Y. Huang, X. Han, W. Xu, C. Xiao, X. Zhang, Y. Fang, K. Zhang, Z. Liu, and M. Sun (2024)
Ouroboros: generating longer drafts phrase by phrase for faster speculative decoding.

arXiv preprint arXiv:2402.13720.

Cited by: Appendix A.

H. Zheng and X. Wang (2025)
Faster speculative decoding via effective draft decoder with pruned candidate tree.

In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),

pp. 9856–9868.

Cited by: §2.2.

Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh, S. Kumar, J. Kagy, and R. Agarwal (2023)
Distillspec: improving speculative decoding via knowledge distillation.

arXiv preprint arXiv:2310.08461.

Cited by: §2.2.

W. Zhu, H. Liu, Q. Dong, J. Xu, S. Huang, L. Kong, J. Chen, and L. Li (2024)
Multilingual machine translation with large language models: empirical results and analysis.

External Links: 2304.04675,
Link

Cited by: §1.

Appendix A Orthogonal Drafting Work

Various avenues of work improve drafting or its integration with other phases without altering the tree construction. Some even fundamentally alter the draft-verify regime. We emphasize that many of these works are orthogonal to ours, and can be combined with dynamic delayed drafting to further improve throughput. We leave exploration of these ideas to future work.

Improving drafting without tree control.

EAGLE (Li et al., 2024b) trains a lightweight draft model on features to propose many high-quality draft tokens per step. EAGLE-3 (Li et al., 2025) scales EAGLE with test-time compute to improve draft quality under fixed latency. Kangaroo (Liu et al., 2024a) employs self-speculation to train an adaptive layer on a subnetwork of the target model, to use as a draft model. Cacheback (Ma et al., 2025) uses an LRU cache of n-grams to accelerate drafting. Cascade drafting (Chen et al., 2024b) uses horizontal and vertical cascading, with larger drafts for crucial earlier tokens and smaller drafts for later tokens. Ouroboros (Zhao et al., 2024) drafts and verifies phrases rather than tokens. Previous work also considers retrieval (He et al., 2023), hierarchical drafting (Sun et al., 2024a), and layer skipping (Zhang et al., 2023; Elhoushi et al., 2024). Our work is agnostic to how draft candidates are produced, so we do not compare against these methods.

Pipelining and parallelism.

Some hardware-aware work focuses on reducing rollbacks and pipeline stalls. Pearl (Liu et al., 2024b) reduces mutual waiting between drafting and target passes by overlapping computation with pre/post-verification pipelining, and tunes adaptive draft lengths for end-to-end latency instead of block efficiency. SpecBranch (Shen et al., 2025) improves Pearl by introducing branch-parallel speculative branches with rollback-aware hybrid drafts, explicitly trading off between parallelism and rollback costs. These are orthogonal to and can be integrated into our work, since we optimize for drafting and these optimize for scheduling once drafting is fixed.

Alternative multi-token prediction.

Speculative decoding is not the only method that decodes multiple tokens per target call. Parallel decoding (Stern et al., 2018) proposes many tokens and uses parallel scoring to reduce autoregressive dependence, but does not faithfully maintain the target distribution. Medusa (Cai et al., 2024) trains multiple independent heads over the draft model to propose draft tokens, and Hydra (Ankner et al., 2024) extends on Medusa with sequential decoding. Multi-token prediction (Gloeckle et al., 2024; Samragh et al., 2025) trains auxiliary heads without a draft model to predict multiple future tokens from a single forward pass. Lookahead decoding (Fu et al., 2024) generates many disjoint n-grams in one parallelized step, and performs verification to maintain the target distribution.

Appendix B Review of OT-Based Methods

Here, we review the five OT-based verification algorithms. Each involves traversing the draft tree top-down based on its corresponding OTLP solver, as described in Section 3.2. Below, pp and qq denote the target and draft distributions, and we use the shorthand notation 𝒙+=max⁡{𝒙,0}\boldsymbol{x}_{+}=\max\{\boldsymbol{x},0\} for vectors 𝒙\boldsymbol{x}. We also use ∝\propto to denote normalizing a distribution.

B.1 NSS

NSS333Miao et al. (2024) called this ”Naive Speculative Sampling”, but we refer to standard speculative sampling as naive, and call this NSS. is the simplest OT-based algorithm, explicitly defined in (Miao et al., 2024). Ignoring the draft tokens, NSS always directly samples a token from the target distribution, as in Algorithm 1. Because NSS does not depend at all on draft model probabilities, it can be used in any drafting regime. Thus, methods like EAGLE-2, which use deterministic trees, use NSS.

Algorithm 1 NSS OTLP Solver

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), i.i.d. draft tokens X1,…,Xk∼qX_{1},\ldots,X_{k}\sim q

0: Token Y∈𝒱Y\in\mathcal{V}

1: Sample Y∼pY\sim p

2: return YY

B.2 Naive

Naive speculative sampling, as we described in Section 3.1, was originally formulated as a single-path method (Leviathan et al., 2023; Chen et al., 2023). However, one can observe that independently accepting draft tree nodes, and then selecting the maximum depth node with all accepted ancestors, is equivalent to performing a top-down OT-based draft tree traversal when the draft tree is a single path. The output of the OTLP solver here is given by either accepting the single child token, or sampling from a residual. This can be extended to a multi-path method in Algorithm 2 by following the same OTLP solver only on the first draft token X1X_{1}, but still allowing one to output and traverse to other draft tokens X2,…,XkX_{2},\ldots,X_{k} when sampling from the residual. Note that while the output of NSS is independent of all draft tokens, the output of Naive is independent of all but the first draft token.

Algorithm 2 Naive OTLP Solver (Speculative Decoding)

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), i.i.d. draft tokens X1,…,Xk∼qX_{1},\ldots,X_{k}\sim q

0: Token Y∈𝒱Y\in\mathcal{V}

1: Sample U∼𝒰​([0,1])U\sim\mathcal{U}([0,1])

2: if U≤p​(X1)/q​(X1)U\leq p(X_{1})/q(X_{1}) then

3:  return X1X_{1}

4: end if

5: pres∝(p−q)+p_{\text{res}}\propto(p-q)_{+}

6: Sample Y∼presY\sim p_{\text{res}}

7: return YY

B.3 SpecTr

SpecTr uses the OTLP solver K-SEQ, described in detail by Sun et al. (2023), so we recap the sampling procedure below and forgo proofs of losslessness. First, we compute the division factor ρ⋆\rho^{\star}. For each ρ\rho, define the quantities

β​(ρ)\displaystyle\beta(\rho)
=∑x∈𝒱min⁡(ρ−1​p​(x),q​(x)),\displaystyle=\sum_{x\in\mathcal{V}}\min\left(\rho^{-1}p(x),q(x)\right),

(6)

pacc​(ρ)\displaystyle p_{\text{acc}}(\rho)
=1−(1−β​(ρ))k.\displaystyle=1-(1-\beta(\rho))^{k}.

(7)

The division factor is the root of ρ↦pacc​(ρ)−ρ​β​(ρ)\rho\mapsto p_{\text{acc}}(\rho)-\rho\beta(\rho), which is monotone decreasing on [1,k][1,k], so binary search can be used to efficiently compute ρ⋆\rho^{\star}. Then, a ρ⋆\rho^{\star}-weighted version of the Naive OTLP solver is performed for multiple rounds, as in Algorithm 3. This OTLP solver precisely reduces to the Naive solver when k=1k=1. Note that ρ⋆\rho^{\star} up to kk can be increased and this will remain an OTLP solver, but the optimal acceptance rate (see Appendix C) is provably achieved at ρ⋆\rho^{\star}.

Algorithm 3 SpecTr OTLP Solver (K-SEQ)

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

0: Token Y∈𝒱Y\in\mathcal{V}

1: Solve pacc​(ρ)=ρ​β​(ρ)p_{\mathrm{acc}}(\rho)=\rho\beta(\rho) for ρ⋆\rho^{\star} by binary search

2: β←∑t∈𝒱min⁡(p​(t)/ρ⋆,q​(t))\beta\leftarrow\sum_{t\in\mathcal{V}}\min\left(p(t)/\rho^{\star},\,q(t)\right)

3: pacc←1−(1−β)kp_{\mathrm{acc}}\leftarrow 1-(1-\beta)^{k}

4: γ←pacc/β\gamma\leftarrow p_{\mathrm{acc}}/\beta

5: for i=1i=1 to kk do

6:  Sample Ui∼𝒰​([0,1])U_{i}\sim\mathcal{U}([0,1])

7:  if ρ⋆​Ui≤p​(Xi)/q​(Xi)\rho^{\star}U_{i}\leq p(X_{i})/q(X_{i}) then

8:   return XiX_{i}

9:  end if

10: end for

11: pres∝(p−min⁡(p/ρ⋆,q)⋅γ)+p_{\text{res}}\propto(p-\min(p/\rho^{\star},q)\cdot\gamma)_{+}

12: Sample Y∼presY\sim p_{\text{res}}

13: return YY

B.4 SpecInfer

SpecInfer is described alongside NSS in Miao et al. (2024). Similar to SpecTr, this algorithm reduces to Naive when k=1k=1. While both SpecTr and SpecInfer involve at most kk rounds of potential acceptances before sampling from the residual, SpecTr only computes the residual at the end, whereas SpecInfer updates it at each round via uniform child selection.

Algorithm 4 SpecInfer OTLP Solver

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

0: Token Y∈𝒱Y\in\mathcal{V}

1: S←[X1,…,Xk]S\leftarrow[X_{1},\dots,X_{k}]

2: while S≠[]S\neq[\ ] do

3:  Sample x∼𝒰​(S)x\sim\mathcal{U}(S) and U∼𝒰​([0,1])U\sim\mathcal{U}([0,1])

4:  if U≤p​(x)/q​(x)U\leq p(x)/q(x) then

5:   return xx

6:  end if

7:  p∝(p−q)+p\propto(p-q)_{+}

8:  Remove one occurrence of xx from SS

9: end while

10: Sample Y∼pY\sim p

11: return YY

B.5 Khisti

Khisti first solves a truncated OTLP based on p,q,kp,q,k to generate an importance-weighted distribution rr. Then, they sample a token x∼rx\sim r by using pairwise tournament selection on X1:kX_{1:k}, and perform naive speculative sampling, with rr replacing the draft qq and [x][x] replacing the draft tokens [X1,…,Xk][X_{1},\ldots,X_{k}]. For further details, see Section 4 of Khisti et al. (2025).

Algorithm 5 Khisti OTLP Solver

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

0: Token Y∈𝒱Y\in\mathcal{V}

1: r←Khisti_Importance_Sample​(p,q,k)r\leftarrow\textsc{Khisti\_Importance\_Sample}(p,q,k)

2: x←Khisti_Tournament_Select​(r,X1:k)x\leftarrow\textsc{Khisti\_Tournament\_Select}(r,X_{1:k})

3: return Naive_OTLP_Solver(p,r,[x])(p,r,[x])

Appendix C OTLP Acceptances

Now, for each of the five OT-based verification algorithms from Appendix B, we show how to compute the acceptance rate as defined in Definition 5.1. We use the exact same notation. These algorithms can all be obtained by following the logic of the OTLP solver and adding the probabilities of accepting tokens at various rounds and sampling draft tokens from the residual. We have empirically confirmed their accuracy with Monte Carlo sampling.

There is one caveat: one cannot get an exact acceptance rate for Khisti efficiently. However, it is possible to obtain a lower bound by simply ignoring the residual contribution for draft tokens other than xx.

Algorithm 6 NSS Acceptance

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), k≥1k\geq 1

1: return ∑t∈𝒱p​(t)​(1−(1−q​(t))k)\sum_{t\in\mathcal{V}}p(t)\big(1-(1-q(t))^{k}\big)

Algorithm 7 Naive Acceptance

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), k≥1k\geq 1

1: return ∑t∈𝒱min⁡(p​(t),q​(t))+∑t∈𝒱(p​(t)−q​(t))+​(1−(1−q​(t))k−1)\sum_{t\in\mathcal{V}}\min(p(t),q(t))+\sum_{t\in\mathcal{V}}(p(t)-q(t))_{+}\big(1-(1-q(t))^{k-1}\big)

Algorithm 8 SpecTr Acceptance

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), k≥1k\geq 1

1: Solve pacc​(ρ)=ρ​β​(ρ)p_{\mathrm{acc}}(\rho)=\rho\beta(\rho) for ρ⋆\rho^{\star} by binary search

2: β←∑t∈𝒱min⁡(p​(t)/ρ⋆,q​(t))\beta\leftarrow\sum_{t\in\mathcal{V}}\min\left(p(t)/\rho^{\star},\,q(t)\right)

3: pacc←1−(1−β)kp_{\mathrm{acc}}\leftarrow 1-(1-\beta)^{k}

4: γ←pacc/β\gamma\leftarrow p_{\mathrm{acc}}/\beta

5: pres∝(p−min⁡(p/ρ⋆,q)⋅γ)+p_{\text{res}}\propto(p-\min(p/\rho^{\star},q)\cdot\gamma)_{+}

6: r←(q−p/ρ⋆)+/(1−β)r\leftarrow(q-p/\rho^{\star})_{+}/(1-\beta)

7: return pacc+(1−pacc)​∑t∈𝒱pres​(t)​(1−(1−r​(t))k)p_{\mathrm{acc}}+(1-p_{\mathrm{acc}})\sum_{t\in\mathcal{V}}p_{\text{res}}(t)\big(1-(1-r(t))^{k}\big)

Algorithm 9 SpecInfer Acceptance

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), k≥1k\geq 1

1: prej←1,m←𝟏p_{\mathrm{rej}}\leftarrow 1,m\leftarrow\mathbf{1}

2: for i=1i=1 to kk do

3:  r←∑t∈𝒱min⁡(p​(t),q​(t))r\leftarrow\sum_{t\in\mathcal{V}}\min(p(t),q(t))

4:  prej←prej⋅(1−r)p_{\mathrm{rej}}\leftarrow p_{\mathrm{rej}}\cdot(1-r)

5:  m←m⊙(𝟏−(q−p)+/(1−r))m\leftarrow m\odot(\mathbf{1}-(q-p)_{+}/(1-r))

6:  p∝(p−q)+p\propto(p-q)_{+}

7: end for

8: α←(1−prej)+prej​∑t∈𝒱p​(t)​(1−m​(t))\alpha\leftarrow(1-p_{\mathrm{rej}})+p_{\mathrm{rej}}\sum_{t\in\mathcal{V}}p(t)\big(1-m(t)\big)

9: return α\alpha

Algorithm 10 Khisti Acceptance Lower Bound

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), k≥1k\geq 1

1: r←Khisti_Importance_Sample​(p,q,k)r\leftarrow\textsc{Khisti\_Importance\_Sample}(p,q,k)

2: return ∑t∈𝒱min⁡(p​(t),r​(t))\sum_{t\in\mathcal{V}}\min(p(t),r(t))

Appendix D OTLP Branching

Next, for the five OT-based verification algorithms from Appendix B, we compute branching probabilities from Definition 5.3. We again use the notation from Appendix B. These algorithms can similarly be obtained by following the logic of the OTLP solver and mixing the branching probabilities from round-wise token acceptances with the final residual. We have confirmed these algorithms are accurate with Monte Carlo sampling. Unlike acceptance computation, Khisti does have an exact algorithm, by extending tournament selection to a distribution on the output of tournament selection conditioned on the draft tokens.

Algorithm 11 NSS Branching

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

1: return {Xi↦p​(Xi)}\{X_{i}\mapsto p(X_{i})\}

Algorithm 12 Naive Branching

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

1: a←min⁡(1,p​(X1)/q​(X1))a\leftarrow\min(1,p(X_{1})/q(X_{1}))

2: pres∝(p−q)+p_{\text{res}}\propto(p-q)_{+}

3: return {Xi↦(1−a)​pres​(Xi)+a⋅𝟏​{Xi=X1}}\{X_{i}\mapsto(1-a)p_{\text{res}}(X_{i})+a\cdot\mathbf{1}\{X_{i}=X_{1}\}\}

Algorithm 13 SpecTr Branching

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

1: Solve pacc​(ρ)=ρ​β​(ρ)p_{\mathrm{acc}}(\rho)=\rho\beta(\rho) for ρ⋆\rho^{\star} by binary search

2: β←∑t∈𝒱min⁡(p​(t)/ρ⋆,q​(t))\beta\leftarrow\sum_{t\in\mathcal{V}}\min\left(p(t)/\rho^{\star},\,q(t)\right)

3: pacc←1−(1−β)kp_{\mathrm{acc}}\leftarrow 1-(1-\beta)^{k}

4: γ←pacc/β\gamma\leftarrow p_{\mathrm{acc}}/\beta

5: pres∝(p−min⁡(p/ρ⋆,q)⋅γ)+p_{\text{res}}\propto(p-\min(p/\rho^{\star},q)\cdot\gamma)_{+}

6: for i=1i=1 to kk do

7:  ai←min⁡(1,p​(Xi)/(ρ⋆​q​(Xi)))a_{i}\leftarrow\min(1,p(X_{i})/(\rho^{\star}q(X_{i})))

8: end for

9: return {Xi↦∑j=1kaj​𝟏​{Xj=Xi}​∏l<j(1−al)+pres​(Xi)​∏l=1k(1−al)}\{X_{i}\mapsto\sum_{j=1}^{k}a_{j}\mathbf{1}\{X_{j}=X_{i}\}\prod_{l<j}(1-a_{l})+p_{\text{res}}(X_{i})\prod_{l=1}^{k}(1-a_{l})\}

Algorithm 14 SpecInfer Branching

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

1: p0←pp_{0}\leftarrow p

2: for i=1i=1 to kk do

3:  pi∝(pi−1−q)+p_{i}\propto(p_{i-1}-q)_{+}

4:  ai←min⁡(𝟏,pi−1/q)a_{i}\leftarrow\min(\boldsymbol{1},p_{i-1}/q)

5: end for

6: for i=ki=k to 0 do

7:  for multisets S⊆[X1,…,Xk]S\subseteq[X_{1},\ldots,X_{k}] s.t. |S|=k−i|S|=k-i do

8:   for x∈{X1,…,Xk}x\in\{X_{1},\ldots,X_{k}\} do

9:    if i=ki=k then

10:     ℬi​(S;x)←pk​(x)\mathcal{B}_{i}(S;x)\leftarrow p_{k}(x)

11:    else

12:     ℬi​(S;x)←|S|−1​∑t∈S(ai​(t)​𝟏​{t=x}+(1−ai​(t))​ℬi+1​(S∖{t};x))\mathcal{B}_{i}(S;x)\leftarrow|S|^{-1}\sum_{t\in S}(a_{i}(t)\mathbf{1}\{t=x\}+(1-a_{i}(t))\mathcal{B}_{i+1}(S\setminus\{t\};x))

13:    end if

14:   end for

15:  end for

16: end for

17: return {Xi↦ℬ0​([X1,…,Xk];Xi)}\{X_{i}\mapsto\mathcal{B}_{0}([X_{1},\dots,X_{k}];X_{i})\}

Algorithm 15 Khisti Branching

0: Target distribution p∈Δ​(𝒱)p\in\Delta(\mathcal{V}), draft distribution q∈Δ​(𝒱)q\in\Delta(\mathcal{V}), draft tokens X1,…,XkX_{1},\dots,X_{k}

1: r←Khisti_Importance_Sample​(p,q,k)r\leftarrow\textsc{Khisti\_Importance\_Sample}(p,q,k)

2: for x∈{X1,…,Xk}x\in\{X_{1},\ldots,X_{k}\} do

3:  πx←ℙ​(Khisti_Tournament_Select​(r,X1:k)=x)\pi_{x}\leftarrow\mathbb{P}(\textsc{Khisti\_Tournament\_Select}(r,X_{1:k})=x)

4: end for

5: return ∑x∈{X1,…,Xk}πx⋅Naive_Branching​(p,r,[x])\sum_{x\in\{X_{1},\dots,X_{k}\}}\pi_{x}\cdot\textsc{Naive\_Branching}(p,r,[x])

Appendix E Neural Selector Details

Let 𝒜\mathcal{A} be the space of all supported actions for delayed expansion, which we define as the Cartesian product

A={1,…,Kmax}×{0,…,L1,max}×{0,…,L2,max},A=\{1,...,K_{\text{max}}\}\times\{0,...,L_{\text{1,max}}\}\times\{0,...,L_{\text{2,max}}\},

(8)

where an action a=(K,L1,L2)∈𝒜a=(K,L_{1},L_{2})\in\mathcal{A} specifies the branching factor KK, the trunk draft length L1L_{1} and the branch draft length L2L_{2}. When using delayed tree expansion, we aim to maximize the throughput as measured in tokens per second (TPS):

maxπ𝔼𝒄[𝔼​[τ+1∣𝒄,a]𝔼​[T​(𝒄,a)]],a∼π(⋅∣𝒄),\max_{\pi}\mathbb{E}_{\boldsymbol{c}}\!\left[\frac{\mathbb{E}[\tau+1\mid\boldsymbol{c},a]}{\mathbb{E}[T(\boldsymbol{c},a)]}\right],\qquad a\sim\pi(\cdot\mid\boldsymbol{c}),

(9)

where 𝔼​[τ+1∣𝒄,a]\mathbb{E}[\tau+1\mid\boldsymbol{c},a] is the block efficiency at the context 𝒄\boldsymbol{c} having chosen the drafting action aa, and 𝔼​[T​(𝒄,a)]\mathbb{E}[T(\boldsymbol{c},a)] represents the expected wall-time (in seconds) of the necessary draft and target passes performed at context 𝒄\boldsymbol{c} under the drafting action aa. The outer expectation is taken over the distributions of contexts 𝒄\boldsymbol{c} we would observe across all inputs.

To this end, we define a selector policy which takes in the following features as input:

•

Hidden-state features hprevph^{p}_{\mathrm{prev}}, hprevqh^{q}_{\mathrm{prev}} of the target and draft model at the preceding token, and hcurqh^{q}_{\mathrm{cur}}, the draft model features at the root token444We cannot use target features here because the root token is missing a KV cache row from the last iteration, which would require an extra target forward pass. This is also the case for drafting, but an extra draft forward pass is relatively cheap., all of which are later projected to a shared dimension dd,

•

Scalar uncertainty features, which include the entropies H​(pp​r​e​v),H​(qp​r​e​v),H​(qr​o​o​t)H(p_{prev}),H(q_{prev}),H(q_{root}), local divergences K​L​(pp​r​e​v∥qp​r​e​v),K​L​(qp​r​e​v∥pp​r​e​v)KL(p_{prev}\|q_{prev}),KL(q_{prev}\|p_{prev}), and a local L1L^{1} distance ‖pp​r​e​v−qp​r​e​v‖\|p_{prev}-q_{prev}\|,

•

Local parameter features, i.e. the context length |x||x| and the sampling hyperparameters – temperature TT and nucleus threshold pt​o​pp_{top}, and

•

Latency estimates for a draft and target forward pass at the current context length, to capture hardware-dependent effects.

We design this selector as an MLP policy with separate projections for each hidden-state feature block. Let ϕ1,ϕ2,ϕ3\phi_{1},\phi_{2},\phi_{3} be the linear projections from each representation vector to a shared dimension d=128d=128, each followed by layer normalization (LN), and s​(x)s(x) be the (standardized) scalar feature vector. The policy then computes

zθ​(x)=M​L​P​([L​N​(ϕ1​(hprevp));L​N​(ϕ2​(hprevq));L​N​(ϕ3​(hcurq));s​(x)]),z_{\theta}(x)=MLP\left(\big[LN(\phi_{1}(h^{p}_{\text{prev}}));LN(\phi_{2}(h^{q}_{\text{prev}}));LN(\phi_{3}(h^{q}_{\text{cur}}));s(x)\big]\right),

(10)

where the MLP has two hidden layers (of sizes 512 and 32 respectively), with GELU activations (Hendrycks and Gimpel, 2023) and dropout. The output layer of this model is a vector of |A||A| logits, representing the probabilities of each action in AA being optimal with respect to throughput. Using this structure, the overhead for the selector forward pass is negligible compared to any draft or target model pass.

Let tp​(l)t_{p}(l) be the measured target model time in seconds for a forward pass at context length ll, and let tq​(l)t_{q}(l) be the draft model time. Then, for an action a=(K,L1,L2)a=(K,L_{1},L_{2}) at context length l=|x|l=|x| we approximate total wall-clock time as

T^​(x,a)=∑j=0L1−1tq​(l+j)⏟trunk drafting+∑j=0L2−1tq​(l+L1+j∗K)⏟branch drafting+tp​(l+L1+K​L2)⏟target forward pass.\hat{T}(x,a)=\underbrace{\sum_{j=0}^{L_{1}-1}t_{q}(l+j)}_{\text{trunk drafting}}+\underbrace{\sum_{j=0}^{L_{2}-1}t_{q}(l+L_{1}+j*K)}_{\text{branch drafting}}+\underbrace{t_{p}(l+L_{1}+KL_{2})}_{\text{target forward pass}}.

(11)

which we use to estimate total throughput as shown in Equation 4. Given this estimate T​P​S^π​(𝒄)\widehat{TPS}_{\pi}(\boldsymbol{c}) and the throughput of the baseline action T​P​S^b​a​s​e​(𝒄)\widehat{TPS}_{base}(\boldsymbol{c}), we ultimately define the policy’s objective as minimizing:

ℒ=1B​∑i=1B(−log⁡T​P​S^π​(xi)T​P​S^b​a​s​e​(xi))+λ⋅1|ℐα|​∑i∈ℐα(max⁡{1−T​P​S^π​(xi)T​P​S^b​a​s​e​(xi),0})2,\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}\left(-\log\frac{\widehat{TPS}_{\pi}(x_{i})}{\widehat{TPS}_{base}(x_{i})}\right)\;+\;\lambda\cdot\frac{1}{|\mathcal{I}_{\alpha}|}\sum_{i\in\mathcal{I}_{\alpha}}\left(\max\left\{1-\frac{\widehat{TPS}_{\pi}(x_{i})}{\widehat{TPS}_{base}(x_{i})},0\right\}\right)^{2},

(12)

where BB is the minibatch size, λ\lambda is the penalty weight for regressing below the baseline, and ℐα\mathcal{I}_{\alpha} are the indices of the largest α\alpha fraction of the penalty terms (max⁡{1−T​P​S^π​(xi)T​P​S^b​a​s​e​(xi),0})2(\max\left\{1-\frac{\widehat{TPS}_{\pi}(x_{i})}{\widehat{TPS}_{base}(x_{i})},0\right\})^{2} within the minibatch, which penalize worst-case throughput regressions (relative to the baseline) to prevent the policy from trading occasional but large throughput regressions for marginal average improvements.

Appendix F Extended Online Experimental Results

Qwen-2.5 (32B / 500M)
Gemma-3 (27B / 270M)
Llama-3 (70B / 8B)

Writing
Coding
Translation
Math (E)
Math (H)
Writing
Coding
Translation
Math (E)
Math (H)
Writing
Coding
Translation
Math (E)
Math (H)

Khisti, delayed expansion
14.21
17.81
14.16
23.05
22.80
8.93
9.90
10.73
17.39
17.28
11.00
14.24
13.93
13.75
13.26

Khisti
11.22
19.87
13.88
24.29
24.07
3.53
5.08
8.88
6.60
5.58
11.71
14.25
14.23
14.18
13.92

NaiveTree, delayed expansion
14.57
21.91
16.29
28.89
26.69
9.95
8.45
11.95
15.84
18.10
11.31
15.66
15.04
14.76
13.29

NaiveTree
10.25
19.08
13.28
27.52
25.67
3.48
5.26
9.74
6.48
5.04
11.77
16.10
15.71
15.68
14.28

Naive
7.90
17.47
9.76
24.84
22.77
4.72
6.79
7.30
13.15
14.07
8.65
13.05
12.72
12.42
11.04

NSS, delayed expansion
11.85
20.61
16.93
26.96
25.10
10.17
6.39
11.89
15.82
17.76
9.77
14.75
14.39
12.25
11.10

NSS
8.00
16.28
12.67
24.68
22.76
3.50
5.34
9.87
6.13
4.85
10.08
15.36
15.03
12.86
11.23

SpecInfer, delayed expansion
15.52
23.16
17.39
29.31
27.33
10.02
9.85
12.10
16.53
17.77
11.80
15.77
15.08
14.79
13.90

SpecInfer
10.77
19.76
13.43
27.81
26.14
3.59
5.34
9.92
6.55
5.67
12.29
16.43
15.77
15.52
14.27

SpecTr, delayed expansion
15.58
22.03
17.26
27.33
26.50
10.00
9.87
12.03
16.24
17.97
11.73
15.59
14.85
15.04
13.92

SpecTr
10.94
19.31
13.29
26.56
24.96
3.54
5.38
9.80
6.44
5.26
12.21
16.23
15.63
15.70
14.55

BV
9.53
18.24
11.22
24.69
22.81
5.82
7.66
8.57
15.58
15.25
9.57
13.38
12.94
13.07
12.04

Traversal, KK=2

11.11
20.47
12.94
27.44
25.56
6.31
8.01
9.07
14.24
16.08
10.90
14.81
14.17
14.74
13.84

Traversal, KK=3

11.47
21.34
14.23
29.05
26.63
6.51
8.55
9.46
15.09
17.21
11.95
15.39
14.66
14.55
13.86

Traversal, KK=4

11.95
22.90
14.97
29.41
27.49
6.56
8.94
9.73
15.35
17.23
12.35
15.74
15.19
15.39
15.18

Table 8: Tokens per second by dataset and algorithm

Qwen-2.5 (32B / 500M)
Gemma-3 (27B / 270M)
Llama-3 (70B / 8B)

Writing
Coding
Translation
Math (E)
Math (H)
Writing
Coding
Translation
Math (E)
Math (H)
Writing
Coding
Translation
Math (E)
Math (H)

Khisti, delayed expansion
2.56
5.21
3.22
7.04
6.81
2.01
3.51
2.69
5.72
5.53
5.59
7.48
7.24
7.06
6.85

Khisti
2.78
5.16
3.50
6.59
6.50
1.21
1.77
3.09
2.26
1.92
5.26
6.64
6.63
6.57
6.35

NaiveTree, delayed expansion
2.41
5.31
3.42
7.06
6.49
1.89
2.75
2.82
4.62
5.00
5.15
7.36
6.98
6.75
6.10

NaiveTree
2.62
5.14
3.63
7.00
6.60
1.17
1.78
3.25
2.19
1.68
5.18
7.19
7.01
6.88
6.24

Naive
2.18
4.91
2.73
6.91
6.30
1.74
2.60
2.80
4.91
5.17
4.08
6.33
6.12
5.85
5.22

NSS, delayed expansion
2.00
4.48
3.12
6.24
5.74
1.79
2.71
2.75
4.69
4.75
4.45
7.01
6.68
5.61
5.11

NSS
2.06
4.40
3.46
6.34
5.93
1.17
1.81
3.28
2.04
1.63
4.45
6.86
6.69
5.69
4.93

SpecInfer, delayed expansion
2.64
5.39
3.29
7.20
6.61
2.00
2.84
2.74
4.66
4.87
5.39
7.57
6.99
6.81
6.43

SpecInfer
2.79
5.33
3.67
7.12
6.76
1.20
1.80
3.27
2.17
1.90
5.39
7.31
7.00
6.84
6.21

SpecTr, delayed expansion
2.69
5.32
3.37
6.87
6.59
2.02
2.73
2.77
4.64
5.02
5.43
7.58
7.04
7.08
6.58

SpecTr
2.85
5.33
3.69
7.04
6.65
1.20
1.84
3.30
2.16
1.77
5.40
7.36
7.07
7.08
6.45

BV
2.31
4.48
2.78
6.02
5.59
1.79
2.45
2.76
4.76
4.73
4.18
5.93
5.78
5.71
5.25

Traversal, KK=2

2.70
5.09
3.22
6.63
6.20
2.00
2.67
3.02
4.64
5.17
4.90
6.71
6.48
6.59
6.17

Traversal, KK=3

2.83
5.38
3.60
7.10
6.51
2.08
2.84
3.20
4.96
5.52
5.40
7.10
6.78
6.57
6.31

Traversal, KK=4

2.99
5.80
3.83
7.24
6.78
2.12
3.02
3.29
5.05
5.56
5.62
7.36
7.07
6.96
6.86

Table 9: Block efficiency by dataset and algorithm

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
19.59
19.94
20.06
19.42
17.36
13.58
19.50
17.80

Khisti
18.23
18.66
18.90
19.07
19.37
18.53
18.11
18.47

NaiveTree, delayed expansion
22.17
22.41
22.52
22.01
21.47
19.28
21.88
21.65

NaiveTree
19.75
20.20
19.82
19.52
18.98
16.90
19.03
19.06

Naive
15.40
15.86
16.58
16.85
16.74
16.05
17.51
17.39

NSS, delayed expansion
22.56
22.60
21.88
21.07
19.57
15.93
19.21
19.49

NSS
19.31
18.97
18.55
17.76
15.93
12.72
15.77
16.04

SpecInfer, delayed expansion
22.59
22.67
22.68
22.94
22.82
21.06
22.69
22.87

SpecInfer
19.81
19.99
20.17
20.07
19.61
17.67
19.72
19.62

SpecTr, delayed expansion
21.06
21.88
22.15
22.47
22.35
20.21
21.89
21.93

SpecTr
18.70
19.49
19.72
19.25
19.14
17.19
19.15
19.46

BV
16.07
17.24
17.64
17.81
17.90
15.91
17.96
17.84

Traversal, KK=2

18.93
19.61
20.14
20.20
19.83
17.99
19.68
19.65

Traversal, KK=3

20.14
20.78
20.88
21.18
20.87
18.68
20.93
20.90

Traversal, KK=4

21.09
21.56
21.90
21.63
21.52
19.79
21.68
21.60

Table 10: Throughput (tokens per second) for Qwen

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
4.85
5.02
5.21
5.42
5.28
3.91
4.88
5.18

Khisti
5.13
5.25
5.37
5.28
4.51
3.40
5.09
5.21

NaiveTree, delayed expansion
5.15
5.14
5.25
5.02
4.86
4.26
4.95
4.89

NaiveTree
5.16
5.20
5.17
5.11
4.97
4.44
4.98
4.96

Naive
4.48
4.65
4.84
4.93
4.61
4.18
4.59
4.56

NSS, delayed expansion
5.01
5.01
4.73
4.47
4.05
3.09
4.05
4.09

NSS
5.05
4.98
4.89
4.65
4.19
3.38
4.16
4.22

SpecInfer, delayed expansion
5.11
5.12
5.11
5.13
5.13
4.53
5.01
5.06

SpecInfer
5.18
5.23
5.27
5.27
5.15
4.64
5.17
5.15

SpecTr, delayed expansion
4.79
5.12
5.09
5.20
5.19
4.40
4.96
4.99

SpecTr
5.04
5.24
5.28
5.20
5.15
4.62
5.14
5.21

BV
3.96
4.20
4.30
4.36
4.37
3.93
4.39
4.38

Traversal, KK=2

4.65
4.78
4.89
4.93
4.83
4.40
4.84
4.82

Traversal, KK=3

4.98
5.13
5.17
5.28
5.17
4.62
5.17
5.15

Traversal, KK=4

5.25
5.38
5.40
5.43
5.40
4.97
5.40
5.40

Table 11: Block Efficiencies for Qwen

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
12.73
12.52
13.36
12.76
12.96
12.76
12.79
12.87

Khisti
5.95
5.87
5.86
5.82
6.01
6.00
5.98
5.95

NaiveTree, delayed expansion
13.36
12.97
12.55
12.53
12.75
12.93
12.94
12.82

NaiveTree
6.03
5.96
5.86
6.01
6.10
5.94
6.01
6.11

Naive
8.99
9.19
9.37
9.45
9.06
9.04
9.25
9.29

NSS, delayed expansion
12.72
12.38
12.54
12.47
12.04
12.40
12.05
12.64

NSS
5.95
5.96
6.31
5.84
5.83
5.88
5.88
5.86

SpecInfer, delayed expansion
13.58
13.47
13.25
13.06
13.04
13.25
13.29
13.11

SpecInfer
6.13
6.20
6.21
6.09
6.35
6.24
6.27
6.21

SpecTr, delayed expansion
13.74
12.99
13.36
13.16
12.80
13.05
13.55
13.13

SpecTr
5.72
5.91
5.77
6.44
6.25
6.05
6.21
6.34

BV
10.44
10.34
10.80
10.84
10.32
10.92
10.43
10.49

Traversal, KK=2

10.99
10.96
10.71
10.53
10.80
10.40
10.74
10.81

Traversal, KK=3

11.44
11.09
12.03
11.11
11.31
11.88
11.27
11.32

Traversal, KK=4

12.08
12.01
11.75
11.49
11.16
11.64
11.16
11.22

Table 12: Throughput (tokens per second) for Gemma

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
3.82
3.78
4.06
3.90
3.91
3.89
3.86
3.92

Khisti
2.05
2.04
2.02
2.01
2.07
2.06
2.09
2.06

NaiveTree, delayed expansion
3.51
3.41
3.34
3.34
3.43
3.51
3.37
3.43

NaiveTree
2.02
2.00
1.97
2.02
2.04
1.98
2.04
2.04

Naive
3.47
3.34
3.40
3.53
3.50
3.57
3.37
3.37

NSS, delayed expansion
3.50
3.40
3.38
3.38
3.18
3.26
3.25
3.37

NSS
2.01
1.98
2.09
1.95
1.97
1.95
1.97
1.97

SpecInfer, delayed expansion
3.44
3.40
3.42
3.45
3.42
3.51
3.36
3.38

SpecInfer
2.05
2.06
2.06
2.05
2.10
2.08
2.08
2.08

SpecTr, delayed expansion
3.47
3.35
3.50
3.44
3.39
3.56
3.42
3.37

SpecTr
1.94
1.99
1.94
2.16
2.11
2.06
2.11
2.11

BV
3.27
3.27
3.37
3.37
3.24
3.40
3.24
3.24

Traversal, KK=2

3.58
3.55
3.46
3.49
3.49
3.45
3.49
3.49

Traversal, KK=3

3.71
3.62
3.93
3.70
3.68
3.93
3.68
3.68

Traversal, KK=4

3.96
3.97
3.83
3.82
3.65
3.90
3.67
3.67

Table 13: Block Efficiencies for Gemma

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
13.20
13.14
13.56
13.44
13.52
11.44
13.91
13.67

Khisti
13.52
13.46
13.82
13.89
13.79
12.75
14.06
13.99

NaiveTree, delayed expansion
13.79
14.11
13.99
14.13
14.24
13.56
14.05
14.22

NaiveTree
14.22
14.39
14.41
15.32
14.50
14.34
15.35
15.12

Naive
10.61
11.09
11.57
11.96
12.08
11.38
12.06
11.87

NSS, delayed expansion
13.44
13.43
13.31
12.77
11.90
11.01
11.69
12.05

NSS
13.77
13.36
13.36
13.57
11.93
12.01
12.69
12.59

SpecInfer, delayed expansion
13.88
14.10
14.50
14.48
14.51
14.05
14.24
14.38

SpecInfer
13.97
14.30
14.59
15.65
14.74
15.05
15.12
15.42

SpecTr, delayed expansion
13.92
13.94
14.57
14.60
14.59
13.47
14.18
14.55

SpecTr
14.11
14.27
14.30
15.33
14.68
15.22
15.45
15.54

BV
10.97
11.63
11.98
12.55
12.79
12.24
12.74
12.70

Traversal, KK=2

12.91
12.83
13.87
13.69
14.01
14.03
14.19
14.01

Traversal, KK=3

13.62
14.02
13.83
14.17
14.18
14.31
14.25
14.27

Traversal, KK=4

13.97
14.52
15.04
15.01
14.77
15.07
14.99
14.79

Table 14: Throughput (tokens per second) for Llama

Temperature (top-p=1)
Top-p

0.2
0.4
0.6
0.8
1.0
1.2
0.90
0.99

Khisti, delayed expansion
6.49
6.52
6.77
6.97
7.32
6.59
6.88
7.19

Khisti
6.58
6.58
6.75
6.85
5.72
4.25
6.78
6.81

NaiveTree, delayed expansion
6.34
6.54
6.46
6.50
6.60
6.27
6.50
6.55

NaiveTree
6.42
6.51
6.51
6.61
6.56
6.20
6.65
6.52

Naive
5.06
5.30
5.54
5.70
5.73
5.41
5.76
5.68

NSS, delayed expansion
6.24
6.24
6.21
5.94
5.49
5.06
5.46
5.56

NSS
6.26
6.05
6.06
5.82
5.41
5.20
5.49
5.48

SpecInfer, delayed expansion
6.42
6.60
6.71
6.72
6.76
6.53
6.68
6.69

SpecInfer
6.35
6.46
6.60
6.71
6.63
6.45
6.53
6.66

SpecTr, delayed expansion
6.61
6.58
7.07
6.91
6.88
6.35
6.69
6.87

SpecTr
6.50
6.56
6.60
6.69
6.76
6.66
6.78
6.82

BV
4.81
5.11
5.29
5.54
5.60
5.39
5.64
5.59

Traversal, KK=2

5.83
5.79
6.23
6.19
6.29
6.30
6.41
6.31

Traversal, KK=3

6.21
6.37
6.39
6.47
6.50
6.51
6.49
6.50

Traversal, KK=4

6.40
6.70
6.86
6.84
6.80
6.90
6.88
6.83

Table 15: Block Efficiencies for Llama

Generated on Thu Feb 19 01:35:10 2026 by LaTeXML