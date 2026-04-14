HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding

1 Introduction

2 Preliminaries

2.1 Speculative Decoding

2.2 EAGLEs and its Draft Tree Construction

3 Observations

3.1 The Verification Bottleneck

3.2 Characterizing Heterogeneity in Draft Acceptance

3.3 Implications for Further Optimization

4 Methodology

4.1 Cumulative Meta-Path Top-K𝐾Kitalic_K Entropy

4.2 Hierarchical Entropy Binning via Data-Driven Decision Trees

4.3 Entropy-Bin-Driven Adaptive Optimization

5 Experiments

5.1 Effectiveness

5.2 Ablation Study

5.3 Hyperparameter Study

5.4 More Discussion on HeteroSpec

6 Related Work

7 Conclusion

A Appendix

A.1 Limitations and Future Work

HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding

Siran Liu1,2, Yang Ye1, Qianchao Zhu1, Zheng Cao2, Yongchao He2

1Peking University, 2SCITIX (SGP) TECH PTE. LTD.

liusr25@stu.pku.edu.cn

Corresponding Author.

Abstract

Autoregressive decoding, the standard approach for Large Language Model (LLM) inference, remains a significant bottleneck due to its sequential nature. While speculative decoding algorithms mitigate this inefficiency through parallel verification, they fail to exploit the inherent heterogeneity in linguistic complexity—a key factor leading to suboptimal resource allocation.
We address this by proposing HeteroSpec, a heterogeneity-adaptive speculative decoding framework that dynamically optimizes computational resource allocation based on linguistic context complexity.
HeteroSpec introduces two key mechanisms: (1) A novel cumulative meta-path Top-K𝐾Kitalic_K entropy metric for efficiently identifying predictable contexts. (2) A dynamic resource allocation strategy based on data-driven entropy partitioning, enabling adaptive speculative expansion and pruning tailored to local context difficulty.
Evaluated on five public benchmarks and four models, HeteroSpec achieves an average speedup of 4.26×\times×. It consistently outperforms state-of-the-art EAGLE-3 across speedup rates, average acceptance length, and verification cost. Notably, HeteroSpec requires no draft model retraining, incurs minimal overhead, and is orthogonal to other acceleration techniques. It demonstrates enhanced acceleration with stronger draft models, establishing a new paradigm for context-aware LLM inference acceleration.

1 Introduction

Autoregressive decoding is the backbone of modern large language models (LLMs), enabling them to generate high-quality and coherent text across myriad applications, including dialogue systems, summarization, and question answering brown2020language ; touvron2023llama ; vaswani2023attentionneed ; achiam2023gpt . While this decoding strategy has been key to unlocking LLMs’ potential, its inherent inefficiency presents a major bottleneck for practical deployment. Autoregressive decoding necessitates a sequential generation process, executing a full forward pass through the large target model at every step, leading to considerable computational cost and latency kasai2021deepencodershallowdecoder ; shazeer2019fasttransformerdecodingwritehead . Consequently, effective strategies to accelerate autoregressive decoding without compromising output quality or correctness are essential for meeting the growing demand for faster, more scalable LLM inference, especially in latency-sensitive and resource-constrained settings.

To mitigate the bottleneck of sequential autoregressive decoding, a promising direction is speculative decoding chen2023acceleratinglargelanguagemodel ; leviathan2023fastinferencetransformersspeculative . This approach accelerates generation by employing a smaller, faster draft model to propose a sequence of candidate tokens in a single step. Crucially, these proposed tokens are then validated in parallel by the full target model. By accepting multiple tokens simultaneously when the target model agrees with the draft, this method significantly reduces the number of sequential target model forward passes while provably preserving the exact sampling distribution. Building on this effective paradigm, subsequent research has introduced various enhancements. Medusa cai2024medusasimplellminference uses MLPs trained to predict tokens in parallel with LLM features. And EAGLE-2 li2024eagle2fasterinferencelanguage adopts a dynamic draft tree guided by confidence scores, leading to improved draft acceptance rates and acceleration. The state-of-the-art, EAGLE-3 li2025eagle3scalinginferenceacceleration , further improves draft performance by aggregating contextual information from multiple hidden layers and removing feature loss constraints.

Although advanced speculative decoding has greatly accelerated LLM inference, fundamental challenges remain due to the inherent dynamic and heterogeneous nature of token prediction. This heterogeneity is well-documented by Zipf’s Law zipf1949human and is clearly reflected in the highly variable output distributions of language models. Consequently, text generation is far from uniformly predictable; it constantly transitions between highly frequent, easily anticipatable patterns and complex, low-frequency structures. This intrinsic variability leads to dynamically fluctuating "decoding difficulty" during generation. Predicting frequent patterns is straightforward, potentially allowing for long, high-confidence draft sequences, while predicting rare or complex structures is highly uncertain, requiring more cautious processing. Although current dynamic methods brown2024dynamicdepthdecodingfaster ; zhang2024draftmodelknowsstop ; huang2024specdecboostingspeculativedecoding ; zhang2024adaeagleoptimizingspeculativedecoding recognize the need to adapt, often employing metrics or trained models to control drafting length or stopping point, they primarily focus on optimizing draft generation, often overlooking the equally crucial need for fine-grained adaptation during the verification phase, which remains a significant bottleneck. Moreover, static thresholds or pre-trained predictors struggle with the context-dependent complexity of language, making these controls fragile and poorly generalizable across diverse tasks.

To gain a deeper understanding of this challenge, we empirically profile the draft acceptance process in EAGLE-3. Our analysis reveals a pronounced heterogeneity in draft acceptance outcomes: a small fraction of high-confidence, top-ranked draft candidates are disproportionately responsible for accepted tokens and contribute significantly to acceleration, while the majority yield minimal or no accepted prefixes. This highlights the inefficiency of uniformly processing all candidates and strongly suggests that dynamically allocating computational resources, particularly the computationally expensive verification effort, based on predicted confidence and linguistic complexity, can substantially improve efficiency by prioritizing the most promising candidates.

Motivated by these insights, we introduce HeteroSpec (Heterogeneity-Adaptive Speculative Decoding), a plug-and-play framework designed to dynamically optimize speculative decoding by explicitly adapting computational resource allocation based on the predicted complexity and confidence of linguistic contexts. Crucially, HeteroSpec requires no retraining of the draft model, introduces minimal overhead, and is orthogonal to existing acceleration techniques. First, we propose a cumulative meta-path Top-K𝐾Kitalic_K entropy metric that efficiently quantifies the predictability of potential generation paths, enabling the identification of high-confidence tokens situated in simple contexts.
Second, we introduce a dynamic resource allocation framework that employs data-driven decision tree entropy partitioning to adaptively control speculative expansion and facilitate efficient re-pruning optimizations across diverse prediction paths based on their assessed difficulty. Extensive experiments on five public benchmarks using four open-source LLMs demonstrate the effectiveness of HeteroSpec. It achieves an average speedup of 4.26×\times× and consistently surpasses state-of-the-art speculative decoding methods. Our main contributions are summarized as follows:

•

We provide empirical evidence revealing the pronounced heterogeneity in speculative decoding’s draft acceptance process: a small fraction of high-confidence candidates disproportionately drives acceptance and acceleration, underscoring the inefficiency of uniformly processing all proposals.

•

We introduce HeteroSpec, a heterogeneity-adaptive speculative decoding framework. It dynamically optimizes speculative expansion and re-pruning decisions by adapting to the "decoding difficulty", enabled by core techniques: a cumulative meta-path Top-K𝐾Kitalic_K entropy metric to quantify prediction difficulty and a data-driven dynamic resource allocation framework.

•

Experiments on diverse benchmarks and open-source LLMs show that HeteroSpec consistently outperforms state-of-the-art EAGLE-3 in speedup, acceptance length, and verification overhead.

2 Preliminaries

2.1 Speculative Decoding

Speculative decoding chen2023acceleratinglargelanguagemodel ; leviathan2023fastinferencetransformersspeculative is a leading technique for accelerating autoregressive decoding in LLMs, enabling faster text generation while guaranteeing the output tokens follow the exact probability distribution of the target LLM. It uses a lightweight draft model to propose a sequence of k𝑘kitalic_k candidate tokens, T^j+1:j+ksubscript^𝑇:
𝑗1
𝑗𝑘\hat{T}_{j+1:j+k}over^ start_ARG italic_T end_ARG start_POSTSUBSCRIPT italic_j + 1 : italic_j + italic_k end_POSTSUBSCRIPT, following a prefix T1:jsubscript𝑇:1𝑗T_{1:j}italic_T start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT. The target model then processes this entire draft in a single parallel forward pass. Drafted tokens are validated sequentially: token t^j+isubscript^𝑡
𝑗𝑖\hat{t}_{j+i}over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT is accepted with probability Aj=min⁡(1,pj+i⁢(t^j+i)p^j+i⁢(t^j+i))subscript𝐴𝑗1
subscript𝑝
𝑗𝑖subscript^𝑡
𝑗𝑖subscript^𝑝
𝑗𝑖subscript^𝑡
𝑗𝑖A_{j}=\min\left(1,\frac{p_{j+i}(\hat{t}_{j+i})}{\hat{p}_{j+i}(\hat{t}_{j+i})}\right)italic_A start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = roman_min ( 1 , divide start_ARG italic_p start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ) end_ARG start_ARG over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ) end_ARG ), where p𝑝pitalic_p and p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG are the target and draft model distributions, respectively. If accepted, validation continues; if rejected, a token is sampled from the residual distribution pj+i−p^j+isubscript𝑝
𝑗𝑖subscript^𝑝
𝑗𝑖p_{j+i}-\hat{p}_{j+i}italic_p start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT - over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT to maintain fidelity, and subsequent draft tokens are discarded. Decoding then resumes from position j+i
𝑗𝑖j+iitalic_j + italic_i.

2.2 EAGLEs and its Draft Tree Construction

Building upon the foundation of standard speculative decoding, the EAGLE family of algorithms introduces significant advancements in accelerating LLM inference. A key innovation is its dynamic draft tree construction approach, first prominently featured in EAGLE-2 li2024eagle2fasterinferencelanguage . This dynamic method moves beyond fixed-length drafts by adaptively proposing candidate sequences in a tree structure. The process is broadly divided into two distinct stages: Expansion and Reranking. The subsequent state-of-the-art, EAGLE-3 li2025eagle3scalinginferenceacceleration , inherits and refines this dynamic framework, notably enhancing predictive power by removing feature loss constraints and incorporating richer information from multiple intermediate hidden layers. The dynamic tree construction process in EAGLE-2 consists of the following two phases:

•

Expansion: In this initial phase, EAGLE-2 constructs a preliminary draft tree, denoted as T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT. Starting from the root, nodes are expanded by generating the top-k tokens based on the draft model’s predicted probability distribution p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG. This expansion is selective, prioritizing branches estimated to lead to high global acceptance during the subsequent verification step. The global acceptance value Visubscript𝑉𝑖V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT for a node i𝑖iitalic_i (i.e., a path from the root) is defined as the product of individual token acceptance probabilities along that path, and is approximated during expansion by replacing the true acceptance probability Ajsubscript𝐴𝑗A_{j}italic_A start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT at each step j𝑗jitalic_j with a confidence score cjsubscript𝑐𝑗c_{j}italic_c start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT from the draft model, i.e., Vi≈∏tj∈Path⁢(root,i)cjsubscript𝑉𝑖subscriptproductsubscript𝑡𝑗Pathroot𝑖subscript𝑐𝑗V_{i}\approx\prod_{t_{j}\in\text{Path}(\text{root},i)}c_{j}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ≈ ∏ start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ∈ Path ( root , italic_i ) end_POSTSUBSCRIPT italic_c start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT. The expansion is limited by a maximum tree depth d𝑑ditalic_d.

•

Reranking: Upon completion of the expansion phase, all nodes in T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT are re-evaluated based on their approximated global acceptance values Visubscript𝑉𝑖V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. The Top-N𝑁Nitalic_N nodes with the highest Visubscript𝑉𝑖V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT are selected from anywhere in T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT to form the pruned subtree T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. This selection naturally yields a valid subtree due to the property that Visubscript𝑉𝑖V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is bounded by its ancestors. T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT represents the most promising candidate sequences, which are then submitted to the target model for efficient parallel verification. This verification leverages the tree structure and a tree-mask attention to compute the target probabilities p𝑝pitalic_p for all tokens in T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT simultaneously, respecting their contextual dependencies.

3 Observations

Figure 1: Key empirical observations with EAGLE-3. (a) Illustration of the Terminal Confidence Rank (TCR). (b) Breakdown of runtime overhead during single-turn speculative decoding for models of different sizes, highlighting the dominant cost of the verification stage. (c) Distribution of Terminal Confidence Rank within the Top-N𝑁Nitalic_N draft candidates. (d) Correlation between Average Acceptance Length and Terminal Confidence Rank. (e) Quantile analysis of the Terminal Confidence Rank distribution, showing concentration within the top percentiles of Top-N𝑁Nitalic_N.

3.1 The Verification Bottleneck

From a computational perspective, the target model verification stage is the primary bottleneck in end-to-end LLM inference using speculative decoding. As evidenced by experiments on models of different sizes (Figure 1(b)), the verification phase constitutes a significant portion of total runtime overhead, ranging from 67% to 90%. This substantial cost is primarily due to the large size and complexity of the target model. Consequently, improving overall inference efficiency critically depends on optimizing this stage. This can be achieved through two main avenues: (1) reducing the total number of target model verifications required, and (2) decreasing the computational cost associated with each individual verification call.

3.2 Characterizing Heterogeneity in Draft Acceptance

To gain insights into the dynamics of successful draft acceptance within EAGLE speculative decoding, we introduce the metric of Terminal Confidence Rank (TCR). As depicted in Figure 1(a), TCR is defined as the rank (among the Top-N𝑁Nitalic_N candidate sequences generated by the reranking phase) of the longest prefix ultimately accepted by the target model in a given decoding iteration.

Empirical analysis using EAGLE3-LLAMA3.1-8B on MT_bench Zheng2023JudgingLW reveals significant heterogeneity in the speculative decoding process. Figure 1(c) and (e) show that Terminal Confidence Rank is heavily concentrated among the top 25% of Top-N𝑁Nitalic_N candidates. Furthermore, Figure 1(d) shows a strong correlation between lower TCRs and longer average accepted lengths, often approaching the maximum draft depth. These findings indicate that sequences originating from high-confidence, top-ranked draft candidates are substantially more likely to be accepted and yield greater length gains.

This observed heterogeneity aligns with the nature of language and phenomena like Zipf’s Law, whereby a small number of high-frequency patterns (such as common words and punctuation) make up the majority of natural language text. Simple, high-frequency linguistic patterns are more accurately predicted by the draft model, resulting in higher confidence (lower TCR) and higher acceptance probabilities leading to longer accepted prefixes. Conversely, complex or low-frequency structures are harder to predict, resulting in lower confidence (higher TCR) and shorter accepted lengths. As draft models improve, they are better able to capture these simple, predictable patterns, potentially amplifying this effect.

3.3 Implications for Further Optimization

The empirical evidence of heterogeneity in draft acceptance suggests a clear strategy for optimizing the verification bottleneck: dynamically prioritize the verification of highly confident, top-ranked draft paths. Given that the most successful and longest accepted sequences predominantly originate from candidates with low TCRs (i.e., high confidence ranks) as shown in Figure 1(c-e), focusing target model evaluations on these promising branches first can significantly reduce redundant computation on less likely candidates.

Leveraging this heterogeneity allows for a more efficient allocation of computational resources. By prioritizing high-confidence, potentially long sequences, we can maximize the accepted length per expensive target model call. This approach, which aligns verification effort with the empirical likelihood of success and the potential for significant acceleration, offers a promising direction for improving the overall efficiency and throughput of speculative decoding for LLMs.

4 Methodology

HeteroSpec comprises three key components designed to efficiently leverage contextual heterogeneity for speculative decoding. Section 4.1 introduces the cumulative meta-path Top-K𝐾Kitalic_K entropy metric. Section 4.2 presents a data-driven partition of context complexity into hierarchical entropy bins using shallow decision trees. Section 4.3 presents entropy-bin-driven adaptive optimization strategies. Figure 2 illustrates the HeteroSpec framework.

Figure 2: Illustration of the HeteroSpec framework, where ②, ③, and ④ represent our three unique modules. We demonstrate the main differences between HeteroSpec and EAGLE-3 in the inference pipeline using an example of an EAGLE drafting tree with Top-K𝐾Kitalic_K=2, Top-N𝑁Nitalic_N=20, and Depth=5.

4.1 Cumulative Meta-Path Top-K𝐾Kitalic_K Entropy

The central challenge is to effectively and efficiently distinguish critical paths during speculative decoding, specifically separating simple, well-structured contexts from complex or linguistically challenging ones. Notably, we observe that in these highly simple and well-structured instances, the token probability distributions produced by the draft model at each decoding step are typically highly skewed, with most of the probability mass concentrated on a single token, resulting in very low per-step entropy. This observation motivates our key statistical feature.

Cumulative Meta-Path Top-K𝐾Kitalic_K Entropy. For a candidate path 𝒫=(x1,x2,…,xT)𝒫subscript𝑥1subscript𝑥2…subscript𝑥𝑇\mathcal{P}=(x_{1},x_{2},\ldots,x_{T})caligraphic_P = ( italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT ), we define the cumulative path Top-K𝐾Kitalic_K entropy as:

Hpath(Top⁢-⁢K)⁢(𝒫)=∑t=1THt(Top⁢-⁢K)=−∑t=1T∑i=1Top⁢-⁢Kp~t,i⁢log⁡p~t,i,superscriptsubscript𝐻pathTop-𝐾𝒫superscriptsubscript𝑡1𝑇superscriptsubscript𝐻𝑡Top-𝐾superscriptsubscript𝑡1𝑇superscriptsubscript𝑖1Top-𝐾subscript~𝑝
𝑡𝑖subscript~𝑝
𝑡𝑖H_{\mathrm{path}}^{(\mathrm{Top}\text{-}K)}(\mathcal{P})=\sum_{t=1}^{T}H_{t}^{%
(\mathrm{Top}\text{-}K)}=-\sum_{t=1}^{T}\sum_{i=1}^{\mathrm{Top}\text{-}K}%
\tilde{p}_{t,i}\log\tilde{p}_{t,i},italic_H start_POSTSUBSCRIPT roman_path end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( roman_Top - italic_K ) end_POSTSUPERSCRIPT ( caligraphic_P ) = ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_H start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( roman_Top - italic_K ) end_POSTSUPERSCRIPT = - ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT roman_Top - italic_K end_POSTSUPERSCRIPT over~ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_t , italic_i end_POSTSUBSCRIPT roman_log over~ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_t , italic_i end_POSTSUBSCRIPT ,

(1)

where p~t,isubscript~𝑝
𝑡𝑖\tilde{p}_{t,i}over~ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_t , italic_i end_POSTSUBSCRIPT are normalized probabilities of the Top-K𝐾Kitalic_K tokens.
Calculating only the Top-K𝐾Kitalic_K entropy aims to ensure negligible computational overhead, this approximation has negligible impact for our focus on low-entropy patterns. For the operational metric, we focus on the candidate path 𝒫∗superscript𝒫∗\mathcal{P}^{\ast}caligraphic_P start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT with the highest final-token confidence, i.e., 𝒫∗=arg⁡max𝒫⁡pT,Top⁢-⁢1(|𝒫|)superscript𝒫∗subscript𝒫subscriptsuperscript𝑝𝒫
𝑇Top-1\mathcal{P}^{\ast}=\arg\max_{\mathcal{P}}p^{(\lvert\mathcal{P}\rvert)}_{T,%
\mathrm{Top}\text{-}1}caligraphic_P start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = roman_arg roman_max start_POSTSUBSCRIPT caligraphic_P end_POSTSUBSCRIPT italic_p start_POSTSUPERSCRIPT ( | caligraphic_P | ) end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_T , roman_Top - 1 end_POSTSUBSCRIPT. We compute Hpath(Top⁢-⁢K)⁢(𝒫∗)superscriptsubscript𝐻pathTop-𝐾superscript𝒫∗H_{\mathrm{path}}^{(\mathrm{Top}\text{-}K)}(\mathcal{P^{\ast}})italic_H start_POSTSUBSCRIPT roman_path end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( roman_Top - italic_K ) end_POSTSUPERSCRIPT ( caligraphic_P start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT ) as the confidence indicator for the speculation tree at that step.

4.2 Hierarchical Entropy Binning via Data-Driven Decision Trees

To facilitate fine-grained control and adaptive strategy deployment, we partition the total path entropy into quantized intervals. Rather than manually specifying entropy thresholds, we adopt a data-driven approach utilizing a shallow decision tree, inspired by the interpretability and discretization strengths of Classification and Regression Tree (CART) algorithmsbrei .

We train a 3-layer CART regression tree on a large corpus drawn from the ShareGPT dataset, which was originally utilized for draft model pre-training. We extract fully accepted draft paths, constructing a dataset 𝒟={(𝐱(i),y(i))}𝒟superscript𝐱𝑖superscript𝑦𝑖\mathcal{D}=\{(\mathbf{x}^{(i)},y^{(i)})\}caligraphic_D = { ( bold_x start_POSTSUPERSCRIPT ( italic_i ) end_POSTSUPERSCRIPT , italic_y start_POSTSUPERSCRIPT ( italic_i ) end_POSTSUPERSCRIPT ) }, where 𝐱(i)superscript𝐱𝑖\mathbf{x}^{(i)}bold_x start_POSTSUPERSCRIPT ( italic_i ) end_POSTSUPERSCRIPT denotes the cumulative meta-path Top-K𝐾Kitalic_K entropy and y(i)superscript𝑦𝑖y^{(i)}italic_y start_POSTSUPERSCRIPT ( italic_i ) end_POSTSUPERSCRIPT denotes the rank (among the Top-N𝑁Nitalic_N draft candidates) of its final-token confidence for path 𝒫∗superscript𝒫∗\mathcal{P^{\ast}}caligraphic_P start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT.

For each split, the tree selects a threshold s𝑠sitalic_s to divide D𝐷Ditalic_D into two subsets, Dleftsubscript𝐷leftD_{\text{left}}italic_D start_POSTSUBSCRIPT left end_POSTSUBSCRIPT (with x≤s𝑥𝑠x\leq sitalic_x ≤ italic_s) and Drightsubscript𝐷rightD_{\text{right}}italic_D start_POSTSUBSCRIPT right end_POSTSUBSCRIPT (with x>s𝑥𝑠x>sitalic_x > italic_s), aiming to minimize the total within-bin variance of the confidence rank:

L⁢(s)=1∣Dleft∣⁢∑(x,y)∈Dleft(y−cleft)2+1∣Dright∣⁢∑(x,y)∈Dright(y−cright)2,𝐿𝑠

1delimited-∣∣subscript𝐷leftsubscript𝑥𝑦subscript𝐷leftsuperscript𝑦subscript𝑐left2
1delimited-∣∣subscript𝐷rightsubscript𝑥𝑦subscript𝐷rightsuperscript𝑦subscript𝑐right2L(s)=\frac{1}{\mid D_{\text{left}}\mid}\sum_{(x,y)\in D_{\text{left}}}(y-c_{%
\text{left}})^{2}+\frac{1}{\mid D_{\text{right}}\mid}\sum_{(x,y)\in D_{\text{%
right}}}(y-c_{\text{right}})^{2},italic_L ( italic_s ) = divide start_ARG 1 end_ARG start_ARG ∣ italic_D start_POSTSUBSCRIPT left end_POSTSUBSCRIPT ∣ end_ARG ∑ start_POSTSUBSCRIPT ( italic_x , italic_y ) ∈ italic_D start_POSTSUBSCRIPT left end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_y - italic_c start_POSTSUBSCRIPT left end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + divide start_ARG 1 end_ARG start_ARG ∣ italic_D start_POSTSUBSCRIPT right end_POSTSUBSCRIPT ∣ end_ARG ∑ start_POSTSUBSCRIPT ( italic_x , italic_y ) ∈ italic_D start_POSTSUBSCRIPT right end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_y - italic_c start_POSTSUBSCRIPT right end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ,

(2)

where cleftsubscript𝑐leftc_{\text{left}}italic_c start_POSTSUBSCRIPT left end_POSTSUBSCRIPT and crightsubscript𝑐rightc_{\text{right}}italic_c start_POSTSUBSCRIPT right end_POSTSUBSCRIPT denote the average rank in each subset. By recursively applying this splitting process to a tree of depth 3, we obtain 8888 non-overlapping entropy intervals {ℬ0,ℬ1,…,ℬ7}subscriptℬ0subscriptℬ1…subscriptℬ7\{\mathcal{B}_{0},\mathcal{B}_{1},\ldots,\mathcal{B}_{7}\}{ caligraphic_B start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , caligraphic_B start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , caligraphic_B start_POSTSUBSCRIPT 7 end_POSTSUBSCRIPT }. In practice, we focus on the lowest three entropy bins ℬl=(ℬ0∪ℬ1∪ℬ2)subscriptℬ𝑙subscriptℬ0subscriptℬ1subscriptℬ2\mathcal{B}_{l}=(\mathcal{B}_{0}\cup\mathcal{B}_{1}\cup\mathcal{B}_{2})caligraphic_B start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ( caligraphic_B start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ∪ caligraphic_B start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∪ caligraphic_B start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) for aggressive speculative strategies, as these bins predominantly correspond to simple, high-frequency language segments according to Zipf’s law. Furthermore, training such a decision tree incurs negligible computational overhead, typically taking only a few seconds. Once trained, the tree is automatically integrated into the system and applied across all tasks, enabling an efficient and fully automated deployment process.

4.3 Entropy-Bin-Driven Adaptive Optimization

For each draft iteration, we calculate the cumulative meta-path Top-K𝐾Kitalic_K entropy Hpath(Top−K)⁢(𝒫∗)superscriptsubscript𝐻pathTop𝐾superscript𝒫∗H_{\mathrm{path}}^{(\mathrm{Top}-K)}(\mathcal{P}^{\ast})italic_H start_POSTSUBSCRIPT roman_path end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( roman_Top - italic_K ) end_POSTSUPERSCRIPT ( caligraphic_P start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT ) and assign the corresponding path to an entropy bin ℬisubscriptℬ𝑖\mathcal{B}_{i}caligraphic_B start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. Next, decoding strategies are adaptively selected based on the assigned entropy bin, with specialized optimizations applied to low-entropy bins. Figure 2 presents an example inference pipeline, highlighting the main differences between EAGLE-3 and HeteroSpec.

Dynamic Extended Drafting. When assigned to a low-entropy bin ℬlsubscriptℬ𝑙\mathcal{B}_{l}caligraphic_B start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, we adopt an aggressive strategy by extending the speculative tree beyond the initial depth d𝑑ditalic_d by α−i𝛼𝑖\alpha-iitalic_α - italic_i layers (where i𝑖iitalic_i is the bin index), yielding a new depth d′=d+(α−i)superscript𝑑′
𝑑𝛼𝑖d^{\prime}=d+(\alpha-i)italic_d start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_d + ( italic_α - italic_i ).
Given the persistent prevalence of high-frequency and structurally simple patterns in natural language, when the current draft path falls into a low-entropy bin, subsequent speculative tokens are also likely to remain within low-entropy regions. In this scenario, increasing the speculative depth extends the expected length of accepted segments, thereby reducing the total number of verifications required. Even in rare cases where the speculative extension departs from the low-entropy regime, the verification cost is only paid once for the entire segment, amortizing the computational overhead.

Top-N𝑁Nitalic_N Pruning. When assigned to a low-entropy bin ℬlsubscriptℬ𝑙\mathcal{B}_{l}caligraphic_B start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, we further restrict the set of candidate branches per draft step. Instead of using a fixed, tuning-dependent Top_⁢NdefaultTop_subscript𝑁default\text{Top\_}N_{\text{default}}Top_ italic_N start_POSTSUBSCRIPT default end_POSTSUBSCRIPT, we dynamically adjust the number of candidates for each bin as
Top_⁢Ni=γi⋅Top_⁢Ndefault+(α−i)Top_subscript𝑁𝑖
⋅subscript𝛾𝑖Top_subscript𝑁default𝛼𝑖\text{Top\_}N_{i}=\gamma_{i}\cdot\text{Top\_}N_{\text{default}}+(\alpha-i)Top_ italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⋅ Top_ italic_N start_POSTSUBSCRIPT default end_POSTSUBSCRIPT + ( italic_α - italic_i ),
where γisubscript𝛾𝑖\gamma_{i}italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is a safety upper bound for bin ℬlsubscriptℬ𝑙\mathcal{B}_{l}caligraphic_B start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, specifically set as γ1=0.3subscript𝛾10.3\gamma_{1}=0.3italic_γ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 0.3, γ2=0.6subscript𝛾20.6\gamma_{2}=0.6italic_γ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.6, and γ3=1subscript𝛾31\gamma_{3}=1italic_γ start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT = 1 for the three low-entropy bins, and (α−i)𝛼𝑖(\alpha-i)( italic_α - italic_i ) reflects the increased speculative depth previously described. This mapping avoids the brittleness and complexity of manual hyperparameter tuning by adopting a conservative upper-bound policy: by retaining a relatively large quantile of candidates for each bin, we reliably capture the most likely accepted branches while significantly reducing unnecessary verification on unlikely candidates.

Dynamic Graph Optimization. The dynamic extended drafting and Top-N𝑁Nitalic_N pruning strategies introduce per-iteration variations in the computation graph structure. To maintain inference efficiency under such dynamic control flow, modern graph optimization methods are adopted. By employing just-in-time graph tracing and compilation, specialized computation graphs are generated and cached for distinct speculative configurations, enabling effective operator fusion and efficient reuse. This ensures that high-throughput inference is preserved, even as speculative depth and candidate set size dynamically change during decoding.

5 Experiments

Tasks. To ensure a fair comparison with EAGLE-3, we aligned our task and model settings accordingly, adopting identical weights for all tasks without performing task-specific fine-tuning. Specifically, we evaluated multi-turn dialogue, code generation, mathematical reasoning, instruction following, summarization, and question answering using the public datasets MT-bench Zheng2023JudgingLW , HumanEval chen2021evaluating , GSM8K cobbe2021training , Alpaca alpaca and CNN/Daily Mail nallapati2016abstractive , respectively.

Models. We conduct experiments on Vicuna 13B vicuna2023 , LLaMAInstruct 3.1 8B, LLaMA-Instruct 3.3 70B dubey2024llama , and DeepSeek-R1-Distill-LLaMA
8B deepseek2025deepseek . For the 8B/13B models, experiments are conducted on 1×1\times1 ×NVIDIA A800 80G GPU. For the 70B model, we use 2×2\times2 ×A800 GPUs due to memory limitations.

Metrics.
HeteroSpec is intended to reduce the verification cost of the target model; therefore, we introduce two device-independent metrics: total validation calls and total verification tokens. In addition, we do not modify the target model’s weights or architecture, nor do we relax the acceptance conditions for speculative decoding. Therefore, evaluation of generation quality is unnecessary.

•

Speedup Ratio: the actual speedup ratio compared to vanilla decoding.

•

Average Acceptance Length (τ𝜏\tauitalic_τ): the average number of new tokens generated per drafting-verification cycle.

•

Total Validation Calls (Calls) / Total Verification Tokens (Tokens): Total validation calls denotes the total number of times the target model is invoked for validation during decoding. Total verification tokens denotes the cumulative number of tokens processed by the target model in all validation steps. These metrics together reflect the computational cost associated with verifying candidate outputs in speculative decoding.

Comparison. We compare our approach with the state-of-the-art EAGLE-3li2025eagle3scalinginferenceacceleration . For key hyperparameters such as Depth, Top-K𝐾Kitalic_K, and Top-N𝑁Nitalic_N, we adopt the same settings as those used in the official implementation of EAGLE-3. Since this study does not involve training the draft model, and to avoid the confounding effect of randomness on the interpretation of our method’s effectiveness, only the case of temperature=0 is considered by default throughout the following analysis.

5.1 Effectiveness

Table 1: Speedup ratios, average acceptance lengths (τ𝜏\tauitalic_τ), total validation calls (Calls), and total verification tokens (Tokens) of different methods. V represents Vicuna, L31 represents LLaMA-Instruct 3.1, L33 represents LLaMA-Instruct 3.3, and DSL represents DeepSeek-R1-Distill-LLaMA.

MT-bench
HumanEval
GSM8K
Alpaca
CNN/DM
Mean

Model
Method
Speedup
τ𝜏\tauitalic_τ
Speedup
τ𝜏\tauitalic_τ
Speedup
τ𝜏\tauitalic_τ
Speedup
τ𝜏\tauitalic_τ
Speedup
τ𝜏\tauitalic_τ
Speedup
τ𝜏\tauitalic_τ

Calls
Tokens
Calls
Tokens
Calls
Tokens
Calls
Tokens
Calls
Tokens
Calls
Tokens

V 13B
EAGLE-3
4.01×\times×

6.55
4.78×\times×

7.74
3.95×\times×

6.40
3.99×\times×

6.50
3.60×\times×

6.55
4.07×\times×

6.75

6267
327516
3424
171451
2433
124068
2064
108486
1983
101577
3234
166620

HeteroSpec
4.21×\times×
6.97
5.24×\times×
8.68
4.12×\times×
6.62
4.07×\times×
6.61
3.75×\times×
6.89
4.28×\times×
7.15

5925
281774
3091
128797
2352
113267
2045
104910
1890
89999
3061
143749

L31 8B
EAGLE-3
3.83×\times×

6.16
4.47×\times×

6.83
3.92×\times×

6.34
4.19×\times×

6.83
3.18×\times×

5.39
3.92×\times×

6.31

8704
542387
4124
248626
2477
151276
2573
162604
3973
243080
4370
269595

HeteroSpec
4.01×\times×
6.47
4.75×\times×
7.36
4.11×\times×
6.53
4.39×\times×
7.04
3.37×\times×
5.45
4.13×\times×
6.57

8319
485731
3845
198391
2380
141658
2502
153617
3897
233771
4189
242634

L33 70B
EAGLE-3
4.24×\times×

5.66
5.10×\times×

6.69
4.70×\times×

6.23
4.80×\times×

6.58
3.61×\times×

5.01
4.49×\times×

6.03

10017
494910
4643
223015
2367
115761
2892
146358
4392
214179
4862
238845

HeteroSpec
4.35×\times×
5.80
5.36×\times×
7.04
4.81×\times×
6.39
4.88×\times×
6.76
3.66×\times×
5.03
4.61×\times×
6.20

9747
447511
4449
168115
2320
103464
2838
134851
4362
205676
4743
211923

DSL 8B
EAGLE-3
3.67×\times×

5.85
4.12×\times×

6.57
4.67×\times×

7.09
3.47×\times×

5.64
3.09×\times×

5.02
3.80×\times×

6.03

7013
428458
6353
388987
4814
295708
7084
433001
8118
498255
6676
408882

HeteroSpec
3.84×\times×
6.08
4.35×\times×
6.78
4.96×\times×
7.75
3.71×\times×
5.75
3.24×\times×
5.05
4.02×\times×
6.28

6828
400157
6168
358446
4455
230522
6924
415689
8076
486366
6490
378236

Table 1 presents the comparative performance of our method and EAGLE-3 across several metrics, including speedup ratio, average acceptance length,total validation calls, and total verification tokens. Experimental results demonstrate that our approach consistently outperforms EAGLE-3 on all evaluated datasets and large language models (LLMs), yielding notable performance advantages. Compared to traditional autoregressive decoding, our method achieves an average speedup of 4.26× and attains an average acceptance length of 6.55.

In the code generation task (HumanEval), HeteroSpec exhibits the most significant performance improvement. This is primarily due to the high predictability of code structures, which are largely templated. In most cases, EAGLE-3 exhibits high confidence and is assigned to low-entropy bins. By incorporating the dynamic extended drafting strategy, HeteroSpec enables the acceptance of longer token sequences, resulting in an average reduction of 6.35% in the number of verifications required by the target model. Moreover, the adoption of top-N𝑁Nitalic_N pruning reduces the total number of verification tokens by 23.65% without causing noticeable accuracy loss. This optimization is device-dependent and proves especially advantageous on consumer-grade GPUs with limited computational capability.

For summarization task (CNN/DM), the increased diversity and unpredictability of outputs reduce the match between the draft model and the target model, leading to shorter average accepted lengths for EAGLE-3, with more frequent assignments to high-entropy bins. Consequently, HeteroSpec’s performance gains in these tasks are comparatively modest, and its behavior partially converges towards EAGLE-3. Overall, these experimental results demonstrate that HeteroSpec exhibits strong adaptability and significant effectiveness across different models and tasks, substantially reducing validation overhead for the target model.

It should be noted that HeteroSpec is orthogonal to all existing draft model optimization techniques. As the capabilities of draft models continue to improve in the future, HeteroSpec can realize even greater performance gains with no additional cost. EAGLE-3 further enhances its mathematical reasoning ability by additional training of the DeepSeekR1-Distill-LLaMA 8B model on the OpenThoughts114k-math dataset. Therefore, on the GSM8K task, HeteroSpec produces even more pronounced improvements.

5.2 Ablation Study

Table 2:
Ablation study on HumanEval dataset across three models of different sizes. DED (Dynamic Extended Drafting), TNP (Top-N𝑁Nitalic_N Pruning), and DGO (Dynamic Graph Optimization) are incrementally integrated; each row represents the ablation result after adding the corresponding module to the previous configuration.

LLaMA3.1-8B
Vicuna-13B
LLaMA3.3-70B

Method
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens

EAGLE-3
4.47×\times×

6.83
4124
248626
4.78×\times×

7.74
3424
171451
5.10×\times×

6.69
4643
223015

+DED
4.63×\times×

7.44
3817
230336
5.18×\times×

8.78
3070
153517
5.36×\times×

7.18
4364
209338

+TNP
4.59×\times×

7.36
3845
198391
5.16×\times×

8.68
3091
128797
5.32×\times×

7.04
4449
168115

+DGO
4.75×\times×

7.36
3845
198391
5.24×\times×

8.68
3091
128797
5.36×\times×

7.04
4449
168115

To systematically assess each optimization, we conducted hierarchical ablation experiments on three model sizes. Building upon EAGLE-3, Dynamic Extended Drafting, Top-N𝑁Nitalic_N Pruning, and Dynamic Graph Optimization were integrated stepwise. As shown in Table 2, Dynamic Extended Drafting provides the most significant performance improvement for HeteroSpec. Its underlying mechanism is that, when the cumulative meta-path Top-K𝐾Kitalic_K entropy is low, the probability that an initially drafted token is accepted at all layers increases significantly. Under such circumstances, even if subsequent generated tokens do or do not fall into low-entropy bins, their verification cost can be amortized alongside that of the initially drafted tokens. Due to the prevalence of simple patterns in natural language, subsequent tokens are still highly likely to remain in low-entropy bins. This mechanism effectively increases the average accepted length, reducing the total number of verifications by an average of 8.65% and thereby significantly alleviating the verification overhead bottleneck.

Top-N𝑁Nitalic_N Pruning strategy achieves a 19.94% reduction in verification tokens, at the cost of only a 0.11 decrease in average accepted length. This demonstrates that the strategy can select, at key decision points, the most likely-to-be-accepted critical paths dynamically and at minimal cost. Notably, although there is a slight 0.67% decrease in the speedup metric, further analysis reveals that in single-batch A800 GPU test scenarios, the computational load is far from saturating the GPU’s parallel computing threshold. Consequently, the temporal advantage of this optimization has not been fully manifested. However, in the discussion of complex multi-batch scenarios (see Section 5.4), this strategy reveals greater potential for acceleration. Dynamic Graph Optimization compensates for any potential performance loss caused by changes in the computational graph structure introduced by the previous two optimizations. These three optimization strategies collectively form a complementary and synergistic effect, leading to substantial acceleration gains.

5.3 Hyperparameter Study

Table 3: Hyperparameter study (α𝛼\alphaitalic_α) on HumanEval dataset across three models of different sizes.

LLaMA3.1-8B
Vicuna-13B
LLaMA3.3-70B

α𝛼\alphaitalic_α
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens
Speedup
τ𝜏\tauitalic_τ
Calls
Tokens

EAGLE-3
4.47×\times×

6.83
4124
248626
4.78×\times×

7.74
3424
171451
5.10×\times×

6.69
4643
223015

⌈d⁢e⁢p⁢t⁢h/2⌉−1
𝑑𝑒𝑝𝑡ℎ21\left\lceil depth/2\right\rceil-1⌈ italic_d italic_e italic_p italic_t italic_h / 2 ⌉ - 1
4.72×\times×

7.25
3905
200285
5.19×\times×

8.44
3159
131017
5.35×\times×

7.03
4474
169390

⌈d⁢e⁢p⁢t⁢h/2⌉
𝑑𝑒𝑝𝑡ℎ2\left\lceil depth/2\right\rceil⌈ italic_d italic_e italic_p italic_t italic_h / 2 ⌉
4.75×\times×

7.36
3845
198391
5.24×\times×

8.68
3091
128797
5.36×\times×

7.04
4449
168115

⌈d⁢e⁢p⁢t⁢h/2⌉+1

𝑑𝑒𝑝𝑡ℎ21\left\lceil depth/2\right\rceil+1⌈ italic_d italic_e italic_p italic_t italic_h / 2 ⌉ + 1
4.67×\times×

7.46
3805
197230
5.29×\times×

8.86
3006
126372
5.29×\times×

7.02
4479
169398

To further investigate the effect of the hyperparameter α𝛼\alphaitalic_α in Dynamic Extended Drafting, we conduct experiments with varying α𝛼\alphaitalic_α values, as summarized in Table 3. We find that the speedup is maximized when α𝛼\alphaitalic_α is set to ⌈d⁢e⁢p⁢t⁢h/2⌉
𝑑𝑒𝑝𝑡ℎ2\left\lceil depth/2\right\rceil⌈ italic_d italic_e italic_p italic_t italic_h / 2 ⌉, while both increasing or decreasing α𝛼\alphaitalic_α leads to reduced acceleration. In essence, this hyperparameter balances the additional drafting overhead against the reduction in verification cost. If α𝛼\alphaitalic_α is too low, the draft model underutilizes its capacity in low-entropy regions, missing opportunities to accept longer token sequences. Conversely, if α𝛼\alphaitalic_α is too high, the extra drafting overhead outweighs the savings in verification, leading to degraded performance. These results indicate that the optimal α𝛼\alphaitalic_α increases with the capability of the draft model; As the draft model improves, a larger α𝛼\alphaitalic_α can yield greater speedups.

5.4 More Discussion on HeteroSpec

The HeteroSpec method demonstrates strong potential for large-scale large-scale language model (LLM) service systems using speculative decoding liu2024optimizingspeculativedecodingserving ; sadhukhan2025magicdecbreakinglatencythroughputtradeoff ; li2025adaserveslocustomizedllmserving . Different applications impose diverse, scenario-specific service-level objectives (SLOs) on LLM inference latency. For instance, chatbots can tolerate response latencies of 200∼500similar-to200500200\sim 500200 ∼ 500 ms, while web search and autonomous driving require much stricter constraints, with the latter demanding decisions within 20∼100similar-to2010020\sim 10020 ∼ 100 ms BRYSBAERT2019104047 ; zhong2024distservedisaggregatingprefilldecoding ; lin2024parrotefficientservingllmbased . SLO-customized LLM service systems are thus designed to dynamically select tokens to meet these individualized latency requirements while optimizing overall throughput.

Formally, such problems are modeled by introducing a hardware budget, defined as the maximum number of tokens processed per forward pass li2025adaserveslocustomizedllmserving . In each decoding iteration, given a hardware budget and a batch of requests, the SLO-customized system aims to (1) satisfy diverse SLO requirements—typically measured by TPOT (Time Per Output Token)—and (2) maximize the number of tokens accepted during verification. HeteroSpec leverages Dynamic Extended Drafting to significantly increase token acceptance for low-entropy requests, while Top-N𝑁Nitalic_N Pruning reduces the budget needed for these requests without sacrificing acceptance rates. The saved budget can then be reallocated to requests with stricter SLOs, improving overall SLO satisfaction. As an orthogonal strategy to existing schedulers, HeteroSpec can significantly enhance throughput in SLO-customized systems. Future work will focus on integrating this approach into inference service systems.

6 Related Work

Speculative decoding chen2023acceleratinglargelanguagemodel ; leviathan2023fastinferencetransformersspeculative adopts a "drafting with a lightweight model and verification with the original model" paradigm for lossless acceleration. SpecInfer Miao_2024 introduced a tree-based attention mechanism to enable more efficient parallel verification. REST fu2024breaksequentialdependencyllm and LLMA yang2023inferencereferencelosslessacceleration employ retrieval-based methods for drafting. GLIDE du2024glidecapelowhasslemethod and MoA zimmer2025mixtureattentionsspeculativedecoding reuse the KV cache of the target model. Medusa cai2024medusasimplellminference , Hydra ankner2024hydrasequentiallydependentdraftheads , EAGLE li2025eaglespeculativesamplingrequires , and EAGLE-3 li2025eagle3scalinginferenceacceleration reuse the feature representations of the target model. Other approaches zhang2023draft ; yi2024generation ; elhoushi2024layer ; liu2024kangaroo ; sun2024triforce ; svirschevski2024specexec reuse partial weights of the target model. These approaches primarily focus on training more powerful draft models, which is orthogonal to our method. As the draft model’s performance improves, our approach can yield even greater gains.

Another line of research explores dynamic draft structures. EAGLE-2 li2024eagle2fasterinferencelanguage uses a dynamic drafting tree with joint probability as confidence. BiLD kim2023speculativedecodingbiglittle , Kangaroo liu2024kangaroo , DDD brown2024dynamicdepthdecodingfaster , and SVIP zhang2024draftmodelknowsstop introduce metrics to determine whether to continue drafting. SpecDec++ huang2024specdecboostingspeculativedecoding and AdaEAGLE zhang2024adaeagleoptimizingspeculativedecoding train additional modules to control early stopping or predict the draft length. C2T huo2025c2tclassifierbasedtreeconstruction proposes a tree-based method using small classifiers to correct the bias of joint probability as a confidence measure. However, these methods fail to fully exploit the heterogeneous linguistic nature to optimize the heavy verification overhead, resulting in limitations in performance, timeliness, applicability, and simplicity. In contrast, our method adaptively and dynamically optimizes computational resource allocation, achieving superior performance compared to existing SOTA methods.

7 Conclusion

Based on the challenges posed by the heterogeneous statistical properties of natural language, this work presents HeteroSpec, a plug-and-play, heterogeneity-adaptive speculative decoding framework that dynamically allocates computational resources based on linguistic context complexity. By introducing a cumulative meta-path Top-K𝐾Kitalic_K entropy metric and data-driven dynamic resource allocation, HeteroSpec efficiently identifies and exploits simple, high-confidence paths to minimize verification overhead without sacrificing accuracy. Experiments show that HeteroSpec achieves greater speedups than the state-of-the-art EAGLE-3, demonstrating the value of leveraging linguistic heterogeneity for efficient LLM inference. Limitations and future work are discussed in Appendix A.1.

References

[1]

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774, 2023.

[2]

Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon.

Hydra: Sequentially-dependent draft heads for medusa decoding, 2024.

[3]

L. Breiman, Jerome H. Friedman, Richard A. Olshen, and C. J. Stone.

Classification and regression trees.

1984.

[4]

Oscar Brown, Zhengjie Wang, Andrea Do, Nikhil Mathew, and Cheng Yu.

Dynamic depth decoding: Faster speculative decoding for llms, 2024.

[5]

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

Language models are few-shot learners.

2020.

[6]

Marc Brysbaert.

How many words do we read per minute? a review and meta-analysis of reading rate.

Journal of Memory and Language, 109:104047, 2019.

[7]

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao.

Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024.

[8]

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper.

Accelerating large language model decoding with speculative sampling, 2023.

[9]

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.

Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374, 2021.

[10]

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing.

Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.

[11]

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.

Training verifiers to solve math word problems.

arXiv preprint arXiv:2110.14168, 2021.

[12]

Daya Guo DeepSeek-AI, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948, 2025.

[13]

Cunxiao Du, Jing Jiang, Xu Yuanchen, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, and Yang You.

Glide with a cape: A low-hassle method to accelerate speculative decoding, 2024.

[14]

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al.

The Llama 3 herd of models.

arXiv preprint arXiv:2407.21783, 2024.

[15]

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al.

Layer skip: Enabling early exit inference and self-speculative decoding.

arXiv preprint arXiv:2404.16710, 2024.

[16]

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang.

Break the sequential dependency of llm inference using lookahead decoding, 2024.

[17]

Kaixuan Huang, Xudong Guo, and Mengdi Wang.

Specdec++: Boosting speculative decoding via adaptive candidate lengths, 2024.

[18]

Feiye Huo, Jianchao Tan, Kefeng Zhang, Xunliang Cai, and Shengli Sun.

C2t: A classifier-based tree construction method in speculative decoding, 2025.

[19]

Jungo Kasai, Nikolaos Pappas, Hao Peng, James Cross, and Noah A. Smith.

Deep encoder, shallow decoder: Reevaluating non-autoregressive machine translation, 2021.

[20]

Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer.

Speculative decoding with big little decoder, 2023.

[21]

Yaniv Leviathan, Matan Kalman, and Yossi Matias.

Fast inference from transformers via speculative decoding, 2023.

[22]

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang.

Eagle-2: Faster inference of language models with dynamic draft trees, 2024.

[23]

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang.

Eagle-3: Scaling up inference acceleration of large language models via training-time test, 2025.

[24]

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang.

Eagle: Speculative sampling requires rethinking feature uncertainty, 2025.

[25]

Zikun Li, Zhuofu Chen, Remi Delacourt, Gabriele Oliaro, Zeyu Wang, Qinghan Chen, Shuhuai Lin, April Yang, Zhihao Zhang, Zhuoming Chen, Sean Lai, Xupeng Miao, and Zhihao Jia.

Adaserve: Slo-customized llm serving with fine-grained speculative decoding, 2025.

[26]

Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, and Lili Qiu.

Parrot: Efficient serving of llm-based applications with semantic variable, 2024.

[27]

Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Kai Han, and Yunhe Wang.

Kangaroo: Lossless self-speculative decoding via double early exiting.

arXiv preprint arXiv:2404.18911, 2024.

[28]

Xiaoxuan Liu, Cade Daniel, Langxiang Hu, Woosuk Kwon, Zhuohan Li, Xiangxi Mo, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang.

Optimizing speculative decoding for serving large language models using goodput, 2024.

[29]

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia.

Specinfer: Accelerating large language model serving with tree-based speculative inference and verification.

In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS ’24, page 932–949. ACM, April 2024.

[30]

Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al.

Abstractive text summarization using sequence-to-sequence rnns and beyond.

arXiv preprint arXiv:1602.06023, 2016.

[31]

Ranajoy Sadhukhan, Jian Chen, Zhuoming Chen, Vashisth Tiwari, Ruihang Lai, Jinyuan Shi, Ian En-Hsu Yen, Avner May, Tianqi Chen, and Beidi Chen.

Magicdec: Breaking the latency-throughput tradeoff for long context generation with speculative decoding, 2025.

[32]

Noam Shazeer.

Fast transformer decoding: One write-head is all you need, 2019.

[33]

Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, and Beidi Chen.

Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decoding.

arXiv preprint arXiv:2404.11912, 2024.

[34]

Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin.

Specexec: Massively parallel speculative decoding for interactive llm inference on consumer devices.

arXiv preprint arXiv:2406.02532, 2024.

[35]

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto.

Alpaca: A strong, replicable instruction-following model.

Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023.

[36]

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample.

Llama: Open and efficient foundation language models, 2023.

[37]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

Attention is all you need, 2023.

[38]

Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, and Furu Wei.

Inference with reference: Lossless acceleration of large language models, 2023.

[39]

Hanling Yi, Feng Lin, Hongbin Li, Peiyang Ning, Xiaotian Yu, and Rong Xiao.

Generation meets verification: Accelerating large language model inference with smart parallel auto-correct decoding.

arXiv preprint arXiv:2402.11809, 2024.

[40]

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra.

Draft & verify: Lossless large language model acceleration via self-speculative decoding.

arXiv preprint arXiv:2309.08168, 2023.

[41]

Situo Zhang, Hankun Wang, Da Ma, Zichen Zhu, Lu Chen, Kunyao Lan, and Kai Yu.

Adaeagle: Optimizing speculative decoding via explicit modeling of adaptive draft structures, 2024.

[42]

Ziyin Zhang, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Rui Wang, and Zhaopeng Tu.

Draft model knows when to stop: A self-verification length policy for speculative decoding, 2024.

[43]

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Haotong Zhang, Joseph E. Gonzalez, and Ion Stoica.

Judging llm-as-a-judge with mt-bench and chatbot arena.

ArXiv, abs/2306.05685, 2023.

[44]

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang.

Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving, 2024.

[45]

Matthieu Zimmer, Milan Gritta, Gerasimos Lampouras, Haitham Bou Ammar, and Jun Wang.

Mixture of attentions for speculative decoding, 2025.

[46]

George Kingsley Zipf.

Human Behavior and the Principle of Least Effort.

Addison-Wesley, Reading, MA, 1949.

Appendix A Appendix

A.1 Limitations and Future Work

A limitation of HeteroSpec is its current implementation within the state-of-the-art EAGLE framework, necessitating validation for generalizability to other speculative decoding methods. However, the core principle of dynamic resource allocation guided by linguistic heterogeneity is fundamentally orthogonal and broadly applicable.

Future work includes extending HeteroSpec to support additional orthogonal speculative decoding strategies, such as asynchronous draft-and-target, for enhanced computational efficiency through overlapping execution. We will also explore its application to ultra-long sequence generation, addressing pronounced resource and verification overheads to further demonstrate universality and scalability. Another key direction is integrating HeteroSpec into SLO-aware inference service systems. We envision it as an auxiliary module alongside existing scheduling and batching strategies. By leveraging its orthogonality, HeteroSpec can enhance system performance, provide flexible resource management, and maximize overall throughput and SLO satisfaction in real-world LLM service deployments.

Generated on Mon May 19 15:01:00 2025 by LaTeXML