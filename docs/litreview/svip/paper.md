Title: Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation

URL Source: https://arxiv.org/html/2411.18462

Published Time: Tue, 26 Aug 2025 00:46:12 GMT

Markdown Content:
Ziyin Zhang 1,2, Jiahao Xu 2 Tian Liang 2 Xingyu Chen 1,2 1 1 footnotemark: 1 Zhiwei He 1,2 1 1 footnotemark: 1

Rui Wang 1 Zhaopeng Tu 2 2 2 footnotemark: 2

1 Shanghai Jiao Tong University 2 Tencent 

1{daenerystargaryen,galaxychen,zwhe.cs,wangrui12}@sjtu.edu.cn 

2{jettexu,ttianliang,zptu}@tencent.com

###### Abstract

Conventional speculative decoding (SD) methods utilize a predefined length policy for proposing drafts, which implies the premise that the target model smoothly accepts the proposed draft tokens. However, reality deviates from this assumption: the oracle draft length varies significantly, and the fixed-length policy hardly satisfies such a requirement. Moreover, such discrepancy is further exacerbated in scenarios involving complex reasoning and long-form generation, particularly under test-time scaling for reasoning-specialized models. Through both theoretical and empirical estimation, we establish that the discrepancy between the draft and target models can be approximated by the draft model’s prediction entropy: a high entropy indicates a low acceptance rate of draft tokens, and vice versa. Based on this insight, we propose SVIP: S elf-V er i fication Length P olicy for Long-Context Speculative Decoding, which is a training-free dynamic length policy for speculative decoding systems that adaptively determines the lengths of draft sequences by referring to the draft entropy. Experimental results on mainstream SD benchmarks as well as reasoning-heavy benchmarks demonstrate the superior performance of SVIP, achieving up to 17% speedup on MT-Bench at 8K context compared with fixed draft lengths, and 22% speedup for QwQ in long-form reasoning.

Draft Model Knows When to Stop: 

Self-Verification Speculative Decoding for Long-Form Generation

![Image 1: Refer to caption](https://arxiv.org/html/2411.18462v2/x1.png)

Figure 1: The variance of oracle draft length drastically increases with context length. MTB (MT-Bench): a conventional benchmark for SD systems. AIME: an extremely difficult mathematical testing set for advanced reasoning models.

1 Introduction
--------------

Speculative decoding(Leviathan et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib11); Chen et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib5)) is a novel technique that markedly enhances the generation wall-time of large language models (LLMs). This approach employs a small and efficient draft model to draft sequences, while concurrently utilizing a larger and more powerful expert model to verify the drafts. By avoiding the autoregressive generation of each token through the target LLM, speculative decoding achieves improved efficiency while preserving the quality of the output. This technique is particularly beneficial in the context of inference-time scaling, where LLMs generally generate long-form text.

The majority of research on speculative decoding focuses on improving the acceptance rate of the draft sequences(Sun et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib19); Li et al., [2024b](https://arxiv.org/html/2411.18462v2#bib.bib13); Elhoushi et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib8); Du et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib7); Li et al., [2024a](https://arxiv.org/html/2411.18462v2#bib.bib12); Lu et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib15)). However, they limit their settings to a fixed draft length (e.g. less than 5 tokens), which we find is sub-optimal in the scenario of long-form text generation.

![Image 2: Refer to caption](https://arxiv.org/html/2411.18462v2/x2.png)

Figure 2: Overview of SVIP: the draft model proceeds the generation process (marked by green) until it encounters a token for which it has low confidence (marked by red), signaled by high entropy, at which point the draft model would cease generation and send the draft spans to the target model for verification.

Specifically, as discussed in Section[2.1](https://arxiv.org/html/2411.18462v2#S2.SS1 "2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we investigate the token rejection phenomenon in speculative decoding (SD) systems. Our findings reveal that the oracle draft length varies considerably for long-form generation, as shown in [fig.˜1](https://arxiv.org/html/2411.18462v2#S0.F1 "In Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). Furthermore, heuristic methods, such as predicting the draft length, are impractically challenging because our further investigation reveals that token rejection occurs unexpectedly. Moreover, our closer investigation indicates a strong correlation between token rejection and the draft model’s prediction entropy at that moment.

Inspired by such a correlation, we analyze the acceptance rate and propose our SVIP: S elf-V er i fication Length P olicy in Section[2.2](https://arxiv.org/html/2411.18462v2#S2.SS2.SSS0.Px3 "Detecting Rejection with Draft Entropy ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). Specifically, we derive a lower bound for the acceptance rate based on the entropy information from the draft model. Notably, SVIP not only approximates this lower bound but also dynamically adjusts the length of draft sequences by determining whether to continue drafting or initiate verification after each token generation, as shown in Figure[2](https://arxiv.org/html/2411.18462v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). By optimizing draft sequence lengths, SVIP enhances SD systems’ efficiency. Importantly, our method is entirely training-free and thus can be seamlessly integrated with any SD decoding algorithm, making it broadly applicable and efficient.

With extensive experiments across multiple model sizes and evaluation benchmarks, we demonstrate the superior performance of SVIP in long-context generation. Compared with fixed-length draft policies, it yields up to 17% speedup on MT-Bench(Zheng et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib27)) and 22% on AIME. As a training-free length policy, SVIP is also extremely flexible and compatible with state-of-the-art speculative decoding systems such as EAGLE-2(Li et al., [2024a](https://arxiv.org/html/2411.18462v2#bib.bib12)), achieving an additional 13% speed improvement.

In summary, our contributions are threefold:

1.   1.We provide an in-depth analysis of the disagreement between draft model and target model in speculative decoding systems, highlighting the underperformance of fixed-length draft length policies. 
2.   2.Based on this analysis, we derive a low bound of speculative decoding systems, where the acceptance rate of the draft model could be modeled by its entropy only. We further develop SVIP, an entropy-based dynamic draft length policy for speculative decoding systems, which is extremely flexible and can be adapted to any auto-regressive draft model. 
3.   3.Experimental results demonstrate the superior performance of SVIP over baseline draft length policies on both conventional long-form generation and reasoning-heavy benchmarks. 

2 Draft Model Knows When to Stop
--------------------------------

In this section, we first examine the behavior of draft models at the rejection phenomenon, and analyze the oracle lengths for SD systems. Then, we theoretically derive SVIP, which approximates the draft token acceptance rate using the draft model’s own prediction entropy.

### 2.1 Investigation of Rejection

Speculative decoding enhances the efficiency of large language model (LLM) inference by assuming that draft tokens are _accepted_ by the LLM, thus avoiding autoregressive generation. Should the target model exhibit a tendency to reject tokens, the overall performance of the system may experience considerable degradation. Consequently, empirically investigating the rejection phenomenon is our primary interest.

Specifically, we analyze the distribution characteristics of rejected tokens across two scenarios:

*   •AIME (2022-2024): a challenging math reasoning dataset, using greedy decoding with QwQ-32B-Preview(Team, [2024](https://arxiv.org/html/2411.18462v2#bib.bib20)) and a 1.5B draft model; 
*   •MT-Bench: a conversational and instruction-following benchmark, using sampling decoding with the Qwen2.5 family(Yang et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib23)). 

![Image 3: Refer to caption](https://arxiv.org/html/2411.18462v2/x3.png)

(a) KL Div. (↓\downarrow)

![Image 4: Refer to caption](https://arxiv.org/html/2411.18462v2/x4.png)

(b) Sorted vocab. distribution

Figure 3: Agreement scores and sorted vocabulary log probability at the rejection phenomenon. The x-axis of the [Figure˜3(a)](https://arxiv.org/html/2411.18462v2#S2.F3.sf1 "In Figure 3 ‣ 2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") represents rejected tokens and the four tokens before them.

#### Rejection occurs out of the blue

A natural question is: How does the token rejection occur? Is there any symptom before it happens? We investigate the rejected tokens with the following metrics:

*   •KL divergence for vocabulary distribution difference. 
*   •Vocabulary distribution of accepted and rejected tokens. 

We quantify the occurrence of the token rejection phenomenon along with its corresponding prefix tokens, as illustrated in [Figures˜3(a)](https://arxiv.org/html/2411.18462v2#S2.F3.sf1 "In Figure 3 ‣ 2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") and[3(b)](https://arxiv.org/html/2411.18462v2#S2.F3.sf2 "Figure 3(b) ‣ Figure 3 ‣ 2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). It shows that KL metrics experienced a sudden and substantial surge in the position of the rejected token, and the output vocabulary distribution at rejected tokens differs significantly from previously correctly drafted tokens. Such a phenomenon is expected: KL divergence indicates a system discrepancy with rejection, where the ground-truth tokens are inherently difficult for the draft model to predict. We also quantify the entropy of the draft model at the acceptance and rejection positions in Table[1](https://arxiv.org/html/2411.18462v2#S2.T1 "Table 1 ‣ Rejection occurs out of the blue ‣ 2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). It shows that the draft models suffer from a severely high entropy at rejection, indicating extreme difficulty in modeling the corresponding tokens.

Table 1: Vocabulary entropy of the draft model at accepted and rejected draft tokens.

### 2.2 Our Method: SVIP

Since the rejection phenomenon occurs suddenly and the draft model suffers from high entropy at the rejection position, can we detect the rejection with KL divergence or draft model’s entropy? To achieve this, we seek a theoretical understanding of the rejection phenomenon.

#### Lower Bound of Acceptance

Since rejection denotes the sudden decrease of acceptance rate, we investigate the theoretic acceptance rate of the SD systems. Specifically, given a target model p p, a draft model q q, an input sequence x<t x_{<t}, and a draft token x t x_{t}, it’s easy to derive that x t x_{t}’s acceptance probability is min⁡(1,p​(x t)q​(x t))\min\left(1,\;\frac{p(x_{t})}{q(x_{t})}\right) (see Appendix[A](https://arxiv.org/html/2411.18462v2#A1 "Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")). Let β\beta denote the expected acceptance probability over the distribution of x t x_{t}, and it follows that:

β\displaystyle\beta=∑x q​(x)⋅min⁡(1,p​(x)q​(x))\displaystyle=\sum_{x}q(x)\cdot\min\left(1,\frac{p(x)}{q(x)}\right)
=∑x min⁡(p​(x),q​(x)),\displaystyle=\sum_{x}\min\left(p(x),q(x)\right),(1)

where p p and q q denote the target and draft model respectively. Chen et al. ([2023](https://arxiv.org/html/2411.18462v2#bib.bib5)) has proven that β\beta is related to the total variational distance (TVD) between p p and q q. Start from this, we utilize Pinsker’s inequality in [Equation˜3](https://arxiv.org/html/2411.18462v2#S2.E3 "In Lower Bound of Acceptance ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") and yield the following bound in [Equation˜4](https://arxiv.org/html/2411.18462v2#S2.E4 "In Lower Bound of Acceptance ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"):

β\displaystyle\beta=1−TVD​(p,q)\displaystyle=1-\mathrm{TVD}(p,q)(2)
⩾1−1 2 𝕂 𝕃(q||p)\displaystyle\geqslant 1-\sqrt{\frac{1}{2}\mathbb{KL}(q||p)}(3)
=1−1 2​H q,p−1 2​H q\displaystyle=1-\sqrt{\frac{1}{2}H_{q,p}-\frac{1}{2}H_{q}}(4)

where H q,p H_{q,p} is the cross entropy between q q and p p, and H q H_{q} is the entropy of q q. We denote the above bound as the oracle bound. Utilizing this bound for acceptance prediction is infeasible since it requires instantaneous access to the target model for cross entropy H q,p H_{q,p}, which is infeasible during the drafting phase.

#### Approximating the Oracle Bound with Draft Distribution

Can we approximate the cross-entropy H q,p H_{q,p} between the draft model’s distribution q q and the target distribution p p using only q q? We propose using the draft model’s entropy H q H_{q} as a proxy, approximating H q,p H_{q,p} as γ​H q\gamma H_{q}, where γ=H q,p/H q\gamma=H_{q,p}/H_{q} is a random variable capturing the ratio between H q,p H_{q,p} and H q H_{q}. This leads to a bound on the acceptance rate β\beta:

β≥1−1 2​(γ−1)​H q.\beta\geq 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}.(5)

To make this bound practical, we approximate γ\gamma with a constant c c, yielding the approximation bound:

β≥1−1 2​(γ−1)​H q≈1−c​H q.\beta\geq 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}\approx 1-\sqrt{cH_{q}}.(6)

This bound holds when 1−1 2​(γ−1)​H q≥1−c​H q 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}\geq 1-\sqrt{cH_{q}}, ensuring the approximation is conservative. Thus, the approximation bound 1−c​H q 1-\sqrt{cH_{q}} lower-bounds the true acceptance rate β\beta.

#### Detecting Rejection with Draft Entropy

With [Equation˜6](https://arxiv.org/html/2411.18462v2#S2.E6 "In Approximating the Oracle Bound with Draft Distribution ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") providing a way to estimate the acceptance probability using only the draft model’s entropy, we introduce SVIP, which dynamically adapts the draft length. After generating each draft token, we compute the approximation bound and halt drafting if it falls below a threshold h^\hat{h}, i.e., if 1−c​H q<h^1-\sqrt{cH_{q}}<\hat{h}. Since c c and h^\hat{h} are constant hyperparameters, we simplify the criterion by defining a new threshold h=(1−h^)/c h=(1-\hat{h})/\sqrt{c}, absorbing c\sqrt{c} into h h. This reduces the stopping condition to H q>h\sqrt{H_{q}}>h. Formally, given a prefix x<t x_{<t} of t−1 t-1 tokens, the stopping criterion is:

H q​(x<t)>h,\sqrt{H_{q}(x_{<t})}>h,(7)

where H q​(x<t)H_{q}(x_{<t}) is the entropy of the draft distribution conditioned on x<t x_{<t}. This ensures drafting stops when the estimated acceptance probability is too low, optimizing efficiency while maintaining reliability. We formalize SVIP in Algorithm[1](https://arxiv.org/html/2411.18462v2#alg1 "Algorithm 1 ‣ Detecting Rejection with Draft Entropy ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). The details of the methods Verify and Correct are given in Appendix[A](https://arxiv.org/html/2411.18462v2#A1 "Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), for which different versions are available for sampling (Algorithm[2](https://arxiv.org/html/2411.18462v2#alg2 "Algorithm 2 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), [4](https://arxiv.org/html/2411.18462v2#alg4 "Algorithm 4 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")) and greedy decoding (Algorithm[3](https://arxiv.org/html/2411.18462v2#alg3 "Algorithm 3 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), [5](https://arxiv.org/html/2411.18462v2#alg5 "Algorithm 5 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")).

Algorithm 1 SVIP

1:target model

p p
, draft model

q q
, input sequence

x⩽t x_{\leqslant t}
, maximum length

T T
, threshold

h h

2:Initialize

n←t n\leftarrow t

3:while

n<T n<T
do

4:

j=0 j=0

5:while True do

6: Sample

x n+j∼q​(x|x<n+j)x_{n+j}\sim q(x|x_{<{n+j}})

7:

j←j+1 j\leftarrow j+1

8:if

H​(q x|x<n+j)>h\sqrt{H(q_{x|x_{<n+j}})}>h
then

9: Exit while loop

10:end if

11:end while

12:

γ←j\gamma\leftarrow j

13: Compute

p​(x|x<n+j),j=1,⋯,γ+1 p(x|x_{<{n+j}}),\;j=1,\cdots,\gamma+1
in parallel

14:

n~←n\tilde{n}\leftarrow n

15:for

j=1 j=1
to

γ\gamma
do

16:if Verify

(p x|x<n+j,q x|x<n+j,x n+j)\left(p_{x|x_{<n+j}},\,q_{x|x_{<n+j}},x_{n+j}\right)
then

17:

n~←n~+1\tilde{n}\leftarrow\tilde{n}+1

18:else

19:

x n+j←Correct​(p x|x<n+j,q x|x<n+j)x_{n+j}\leftarrow\text{Correct}\left(p_{x|x_{<n+j}},\,q_{x|x_{<n+j}}\right)

20: Exit for loop

21:end if

22:end for

23:if

n~==n+γ\tilde{n}==n+\gamma
then

24: Sample

x n+γ+1 x_{n+\gamma+1}
from

p​(x|x⩽n+γ)p(x|x_{\leqslant n+\gamma})

25:end if

26:

n←n~+1 n\leftarrow\tilde{n}+1

27:end while

28:

x⩽n x_{\leqslant n}

![Image 5: Refer to caption](https://arxiv.org/html/2411.18462v2/x5.png)

Figure 4: Comparison between the actual acceptance probability in [Equation˜1](https://arxiv.org/html/2411.18462v2#S2.E1 "In Lower Bound of Acceptance ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), the acceptance probability lower bound in [Equation˜4](https://arxiv.org/html/2411.18462v2#S2.E4 "In Lower Bound of Acceptance ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), and the estimated lower bound in [Equation˜6](https://arxiv.org/html/2411.18462v2#S2.E6 "In Approximating the Oracle Bound with Draft Distribution ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). Each position on the x-axis corresponds to a token, which has been sorted according to the actual acceptance probability.

![Image 6: Refer to caption](https://arxiv.org/html/2411.18462v2/x6.png)

Figure 5: The SD system speedup on MT-Bench using Qwen2.5-14B (top) and Qwen2.5-32B (bottom) as targets and three different smaller models as drafts.

#### Justifying the Approximation Bound

The tightness of the approximation depends on how well c c captures the behavior of the random variable γ\gamma. For the approximation bound to be conservative, it requires:

γ≤2​c+1.\gamma\leq 2c+1.(8)

Given the right-skewed nature of γ≥1\gamma\geq 1, we model γ=1+X\gamma=1+X, where X∼Gamma​(α,β)X\sim\text{Gamma}(\alpha,\beta). The probability that the approximation is valid is:

P​(γ≤2​c+1)=P​(X≤2​c)=γ​(α,β⋅2​c)Γ​(α),P(\gamma\leq 2c+1)=P(X\leq 2c)=\frac{\gamma(\alpha,\beta\cdot 2c)}{\Gamma(\alpha)},(9)

where γ​(α,z)=∫0 z t α−1​e−t​𝑑 t\gamma(\alpha,z)=\int_{0}^{z}t^{\alpha-1}e^{-t}\,dt is the lower incomplete gamma function, and Γ​(α)\Gamma(\alpha) is the gamma function. The choice of c c trades off reliability and tightness:

*   •If c c is small (e.g., 2​c<𝔼​[X]=α/β 2c<\mathbb{E}[X]=\alpha/\beta), the probability P​(X≤2​c)P(X\leq 2c) is low, risking an invalid bound. 
*   •If c c is large, P​(X≤2​c)→1 P(X\leq 2c)\to 1, but the bound β≥1−c​H q\beta\geq 1-\sqrt{cH_{q}} becomes looser. 

Optimal performance requires balancing the reliability of the approximation (high P​(X≤2​c)P(X\leq 2c)) with the tightness of the bound (small c c).

We analyze such a trade-off on Qwen2.5 on MT-Bench and QwQ-32B on AIME in [Figure˜4](https://arxiv.org/html/2411.18462v2#S2.F4 "In Detecting Rejection with Draft Entropy ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). It shows that for most cases our estimated approximation bound works well (has a higher acceptance probability than the oracle bound when hyperparameter c c (i.e. h h) is properly selected, set to 0.18 in the figure).

3 Experiments
-------------

Next, to verify the effectiveness of SVIP, we conduct experiments on both conventional long-form generation(Section[3.1](https://arxiv.org/html/2411.18462v2#S3.SS1 "3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")) and reasoning with test-time scaling(Section[3.2](https://arxiv.org/html/2411.18462v2#S3.SS2 "3.2 Long-form Reasoning ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")). Since SVIP is completely training-free, we also apply it to other speculative decoding methods and demonstrate its flexibility(Section[3.1](https://arxiv.org/html/2411.18462v2#S3.SS1.SSS0.Px4 "SVIP further boosts strong baseline ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")).

As baselines, we consider two widely adopted policies for draft length:

1.   1.Constant: a constant draft length (set to 5 unless otherwise stated), which is commonly used in the literature 
2.   2.Heuristic: the heuristics implemented in Hugging Face Transformers library(Wolf et al., [2019](https://arxiv.org/html/2411.18462v2#bib.bib21)), where the draft length for the next draft iteration is increased by 2 if all draft tokens in the current iteration are accepted, and otherwise decreased by 1. 

### 3.1 Results on Long-form Generation

#### Settings

We first validate the effectiveness of SVIP on the widely used MT-Bench(Zheng et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib27)) using different sizes of Qwen2.5(Yang et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib23)) as target and draft models. Unlike many existing works on speculative decoding(Chen et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib5); Du et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib7)) that limit their experiments to generating short sequences of 128 tokens, we conduct experiments on long-form generation with up to 8K context to investigate the applicability of speculative decoding in a broader scope. We set the sampling temperature to 1, as we found that when using greedy decoding in long-form generation, both the draft and the target models are prone to repeat themselves, resulting in very low information entropy and exaggerated speedup ratios(Ouyang et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib17)) (see Appendix[D](https://arxiv.org/html/2411.18462v2#A4 "Appendix D Additional Results on Long-form Generation ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") for details). The entropy threshold h h in SVIP is chosen from {0.2,0.3,0.4,0.5}\{0.2,0.3,0.4,0.5\} based on performance on 8 held out samples using the 14B model as target and the 0.5B model as draft, which is set to 0.3 and reused in all following experiments.

As evaluation metrics, we mainly report the average speedup over target-model-only autoregressive decoding, but also consider other auxiliary information including accepted draft lengths and draft token accept rate. Also, since the memory consumption of verifying n n draft tokens is quadratic in n n, we limit the maximum draft length to 40 in both heuristics and SVIP scenarios, beyond which we start to encounter out-of-memory issues.

#### SVIP outperforms all baselines

We validate our proposed method, SVIP, with two target models: Qwen2.5-14B and Qwen2.5-32B, utilizing draft models that vary in size from 0.5B to 3B. We show the performance of SVIP and baselines in Figure[5](https://arxiv.org/html/2411.18462v2#S2.F5 "Figure 5 ‣ Detecting Rejection with Draft Entropy ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). It shows that SVIP consistently outperforms constant and heuristics draft length by a large margin.

![Image 7: Refer to caption](https://arxiv.org/html/2411.18462v2/x7.png)

Figure 6: Analysis of Qwen2.5 14B/0.5B’s behaviours on MT-Bench. Compared with constant and heuristics length policies, SVIP generates shorter drafts with a significantly higher accept rate.

![Image 8: Refer to caption](https://arxiv.org/html/2411.18462v2/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2411.18462v2/x9.png)

Figure 7: Further analysis of Qwen2.5 14B/0.5B on MT-Bench: (a) SVIP outperforms all constant length policies ranging from 2-5; (b) delta draft length (defined as proposed draft length minus oracle draft length) show that constant and heuristics length policies tend to over-generate drafts, while SVIP models the oracle draft length almost perfectly.

To further investigate the origin of SVIP’s speedup, we analyze the proposed draft lengths and accepte rate of the different length policies in Figure[6](https://arxiv.org/html/2411.18462v2#S3.F6 "Figure 6 ‣ SVIP outperforms all baselines ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). We observe that by terminating the draft process early when the draft model entropy is high, SVIP leads to shorter draft lengths and a much higher acceptance rate. However, in Figure[7](https://arxiv.org/html/2411.18462v2#S3.F7 "Figure 7 ‣ SVIP outperforms all baselines ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") we also compare the performance of SVIP with shorter constant draft length policies, which suggest that simply using shorter constant draft length does not suffice, highlighting the importance of dynamically determining draft lengths. Figure[6](https://arxiv.org/html/2411.18462v2#S3.F6 "Figure 6 ‣ SVIP outperforms all baselines ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") also suggests that while the heuristics draft length policy tends to produce very long drafts at long context, its acceptance rate remains low, resulting in no effective speedup compared with fixed draft length.

#### SVIP drafts fit oracle length

In Figure[7](https://arxiv.org/html/2411.18462v2#S3.F7 "Figure 7 ‣ SVIP outperforms all baselines ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we revisit the concept of “oracle draft length” introduced in Section[2.1](https://arxiv.org/html/2411.18462v2#S2.SS1 "2.1 Investigation of Rejection ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), and plot the differences between actual draft lengths and oracle draft lengths. The results suggest that both constant and heuristics draft length policies tend to generate drafts that are too long, while SVIP models the oracle draft length almost perfectly, with an average delta below 0.5 tokens.

#### SVIP further boosts strong baseline

In the previous experiments, we evaluated SVIP on vanilla speculative decoding, where a standard pretrained Transformer decoder model with the same vocabulary as the target is used as the draft model. However, in the past years many works on speculative decoding have proposed other stronger or more efficient draft models(Cai et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib3); Du et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib7); Li et al., [2024b](https://arxiv.org/html/2411.18462v2#bib.bib13)). Since most of these works assume a constant draft length, SVIP is orthogonal to them and can be applied on top of them without any additional training.

Table 2: Speedup on MT-Bench on top of EAGLE-2, using Vicuna as base models.

Specifically, we also apply SVIP to EAGLE-2(Li et al., [2024a](https://arxiv.org/html/2411.18462v2#bib.bib12)), the current state-of-the-art (SOTA) speculative decoding system which utilizes the target model’s language modeling head on top of the draft model’s features to predict the next draft token, and dynamically constructs a draft tree at each draft position. Following Li et al. ([2024a](https://arxiv.org/html/2411.18462v2#bib.bib12)), we use Vicuna 7B, 13B(Chiang et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib6)) as the base models, and set the sampling temperature to 1. To the best of our knowledge, we are also the first to investigate EAGLE-2’s effectiveness in long-form generation. The results of EAGLE-2 on MT-Bench are given in Table[2](https://arxiv.org/html/2411.18462v2#S3.T2 "Table 2 ‣ SVIP further boosts strong baseline ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). Even on top of this SOTA speculative decoding system, SVIP yields consistent improvement, which is especially notable at longer context length, surpassing the vanilla EAGLE-2 by 14% speedup for Vicuna 7B and 7% for Vicuna 13B.

Table 3: Speedup of QwQ on MATH, AIME, and GPQA, along with their average generation length.

### 3.2 Long-form Reasoning

#### Settings

Recently, o1-style reasoning models have come into the spotlight of LLM research. Thus, we are especially interested in seeing the effectiveness of SVIP and other speculative decoding strategies on such models, which often have very long outputs. Consequently, we utilize QwQ-32B-Preview(Team, [2024](https://arxiv.org/html/2411.18462v2#bib.bib20)), the only applicable open-source reasoning model at the time of writing, which does not have off-the-shelf smaller variants, so we train our own draft model based on Qwen2.5 1.5B by distilling QwQ 32B on 1M mathematical Persona data(Chan et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib4))1 1 1 We have made this model publicly available and will provide links in camera-ready version.. Using this draft model, we conduct experiments on MATH(Hendrycks et al., [2021](https://arxiv.org/html/2411.18462v2#bib.bib9)), AIME 2 2 2[https://maa.org/maa-invitational-competitions/](https://maa.org/maa-invitational-competitions/), and GPQA(Rein et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib18)). We sample 200 questions from MATH ranging from level 1 to level 5, and use 73 questions released from 2022 to 2024 for AIME. For GPQA, we use the diamond test set.

#### SVIP achieves strong speedup in long-form reasoning

The overall results are given in Table[3](https://arxiv.org/html/2411.18462v2#S3.T3 "Table 3 ‣ SVIP further boosts strong baseline ‣ 3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). SVIP outperforms the two baselines by a large margin across different benchmarks and context lengths. Detailed analysis of the different length policies’ behaviours suggests similar results to the previous experiment: SVIP has an average proposal length similar to the constant draft length policy, but with a much higher draft token accept rate, leading to more effective speedup.

![Image 10: Refer to caption](https://arxiv.org/html/2411.18462v2/x10.png)

Figure 8: A case comparison of different length policies during one generation example. Drafts proposed by SVIP vary drastically in length, while constant and heuristics policies are insufficient to model such changes.

Table 4: The average draft model entropy and draft token acceptance rate of some representative tokens between QwQ-32B and 1.5B.

In Table[4](https://arxiv.org/html/2411.18462v2#S3.T4 "Table 4 ‣ SVIP achieves strong speedup in long-form reasoning ‣ 3.2 Long-form Reasoning ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we present some token cases with very low or very high acceptance rate. We find that completing subword units or equations is quite easy for the small draft model, while several keywords in QwQ’s reasoning patterns are much harder. The reverse correlation of draft entropy and acceptance rate on these tokens further validates the motivation of SVIP. In Figure[8](https://arxiv.org/html/2411.18462v2#S3.F8 "Figure 8 ‣ SVIP achieves strong speedup in long-form reasoning ‣ 3.2 Long-form Reasoning ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we plot the behaviors of different draft length policies in an example generation case. The drastic draft length oscillations in SVIP also highlight the importance of dynamic draft length policy.

4 Related Work
--------------

Since Leviathan et al. ([2023](https://arxiv.org/html/2411.18462v2#bib.bib11)) and Chen et al. ([2023](https://arxiv.org/html/2411.18462v2#bib.bib5)) introduced speculative decoding into large language models, numerous works have followed their tracks in pursuit of more efficient LLM inference. We broadly categorize these works into three types: better draft models, draft tree expansion, and draft length control, which are orthogonal to each other. A more comprehensive review of speculative decoding is provided by Xia et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib22)).

#### Better draft models.

As Xia et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib22)) suggest, draft models in speculative decoding can be either based on self-drafting or based on an independent draft model. For the first type, one may use a quantized(Zhao et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib26)), early-exiting(Elhoushi et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib8)), or forward-padded(Monea et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib16)) version of the target model to produce draft tokens, while the second type is represented by the vanilla speculative decoding(Leviathan et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib11)). Some works also take the best of both worlds and introduce extra layers on top of the target model’s hidden representations to construct draft models, represented by EAGLE(Li et al., [2024b](https://arxiv.org/html/2411.18462v2#bib.bib13)), GliDe(Du et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib7)), and Medusa(Cai et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib3)).

Unlike previous methods, SVIP has no requirement for draft models except for autoregression, and is a totally training-free adaptive-length policy, which could boost any draft model’s performance.

#### Draft tree expansion.

Given a draft model, one may verify multiple draft tokens for the same position in parallel to increase the probability of finding an accepted draft token, and we use “draft tree expansion” as an umbrella term for such techniques. Li et al. ([2024a](https://arxiv.org/html/2411.18462v2#bib.bib12)) introduce EAGLE-2, which reranks draft tokens in EAGLE’s draft tree to select tokens with the highest confidence for verification. Similarly, CaPE(Du et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib7)) improves GliDe by expanding the token set chosen for verification at each position based on top-1 confidence. Other works have also addressed the problem of multi-draft verification from a theoretic perspective(Sun et al., [2023](https://arxiv.org/html/2411.18462v2#bib.bib19); Yin et al., [2024](https://arxiv.org/html/2411.18462v2#bib.bib24)).

In contrast to previous methods which introduce tree expansion for proposing fixed fine-grained n-gram draft, our method SVIP involves no fixed tree expansion rules, and is mainly about a more dynamic and flexible draft length policy.

#### Draft length control.

Works in this category are few, but most relevant to ours. Liu et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib14)) introduce PEARL, which lets the target model perform verification in parallel to draft generation, stopping the draft process when a mismatch is found. Huang et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib10)) propose SpecDec++, which trains an acceptance prediction head on top of the draft model to predict the acceptance probability of the current draft token, stopping the draft round when the predicted acceptance probability falls below a constant threshold. Brown et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib2)) propose Dynamic Depth Decoding (DDD) on top of EAGLE-2, which uses the sum of all tokens’ confidences in one level of its draft tree as an indicator to predict whether or not to continue draft generation. Concurrent with our work, Zhang et al. ([2024](https://arxiv.org/html/2411.18462v2#bib.bib25)) propose AdaEAGLE, which utilizes an MLP on top of EAGLE to predict the next round’s draft length.

In contrast to prior length-control strategies that necessitate the training of a length-prediction module, SVIP stands out with its training-free nature. This unique characteristic endows it with remarkable flexibility, allowing it to be seamlessly integrated and applied to any autoregressive draft model.

5 Conclusion
------------

We propose SVIP, a flexible, training-free, and plug-and-play dynamic draft length policy for speculative decoding systems. Based on a theoretical lower bound of acceptance probability and its empirical approximation, SVIP determines whether to continue draft generation or to quit drafting based on the draft model’s entropy after the generation of each draft token. With extensive experiments spanning various base models, draft methods, test domains, and generation length, we validated the effectiveness of SVIP, sparking new insights on speculative decoding and more efficient large language models. For future work, we aim to investigate tighter bounds on the acceptance rate to improve the accuracy of acceptance probability estimates, thereby enabling more efficient draft length adaptation. Additionally, our current analysis may not fully capture context-dependent patterns. A more nuanced investigation into these patterns could further enhance performance.

Limitations
-----------

While SVIP advances speculative decoding through adaptive draft length control, it has several limitations that offer avenues for future work. The acceptance rate bound in [Equation˜6](https://arxiv.org/html/2411.18462v2#S2.E6 "In Approximating the Oracle Bound with Draft Distribution ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") could be overly conservative. Developing a tighter bound would enhance the accuracy of acceptance probability estimates, enabling more effective draft length adaptation. Further, our analysis assumes they follow a simplified distribution discrepancy of SD systems. This may not fully capture the nuanced factors contributing to their occurrence, such as context-dependent patterns or model-specific biases. Context-dependent length proxy for SD systems could be the potential research direction.

References
----------

*   Bretagnolle and Huber (1978) Jean Bretagnolle and Catherine Huber. 1978. [Estimation des densités : risque minimax](http://eudml.org/doc/113157). _Séminaire de probabilités de Strasbourg_, 12:342–363. 
*   Brown et al. (2024) Oscar Brown, Zhengjie Wang, Andrea Do, Nikhil Mathew, and Cheng Yu. 2024. [Dynamic depth decoding: Faster speculative decoding for llms](https://doi.org/10.48550/ARXIV.2409.00142). _CoRR_, abs/2409.00142. 
*   Cai et al. (2024) Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. [Medusa: Simple LLM inference acceleration framework with multiple decoding heads](https://openreview.net/forum?id=PEpbUobfJv). In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_. OpenReview.net. 
*   Chan et al. (2024) Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024. [Scaling synthetic data creation with 1,000,000,000 personas](https://doi.org/10.48550/ARXIV.2406.20094). _CoRR_, abs/2406.20094. 
*   Chen et al. (2023) Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. [Accelerating large language model decoding with speculative sampling](https://doi.org/10.48550/ARXIV.2302.01318). _CoRR_, abs/2302.01318. 
*   Chiang et al. (2023) Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. [Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality](https://lmsys.org/blog/2023-03-30-vicuna/). 
*   Du et al. (2024) Cunxiao Du, Jing Jiang, Yuanchen Xu, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, and Yang You. 2024. [Glide with a cape: A low-hassle method to accelerate speculative decoding](https://openreview.net/forum?id=mk8oRhox2l). In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_. OpenReview.net. 
*   Elhoushi et al. (2024) Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, and Carole-Jean Wu. 2024. [Layerskip: Enabling early exit inference and self-speculative decoding](https://doi.org/10.18653/V1/2024.ACL-LONG.681). In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024_, pages 12622–12642. Association for Computational Linguistics. 
*   Hendrycks et al. (2021) Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. [Measuring mathematical problem solving with the MATH dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html). In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_. 
*   Huang et al. (2024) Kaixuan Huang, Xudong Guo, and Mengdi Wang. 2024. [Specdec++: Boosting speculative decoding via adaptive candidate lengths](https://doi.org/10.48550/ARXIV.2405.19715). _CoRR_, abs/2405.19715. 
*   Leviathan et al. (2023) Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. [Fast inference from transformers via speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html). In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_, volume 202 of _Proceedings of Machine Learning Research_, pages 19274–19286. PMLR. 
*   Li et al. (2024a) Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. [EAGLE-2: faster inference of language models with dynamic draft trees](https://doi.org/10.48550/ARXIV.2406.16858). _CoRR_, abs/2406.16858. 
*   Li et al. (2024b) Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. [EAGLE: speculative sampling requires rethinking feature uncertainty](https://openreview.net/forum?id=1NdN7eXyb4). In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_. OpenReview.net. 
*   Liu et al. (2024) Tianyu Liu, Yun Li, Qitan Lv, Kai Liu, Jianchen Zhu, and Winston Hu. 2024. [Parallel speculative decoding with adaptive draft length](https://doi.org/10.48550/ARXIV.2408.11850). _CoRR_, abs/2408.11850. 
*   Lu et al. (2024) Xiaofan Lu, Yixiao Zeng, Feiyang Ma, Zixu Yu, and Marco Levorato. 2024. [Improving multi-candidate speculative decoding](https://doi.org/10.48550/ARXIV.2409.10644). _CoRR_, abs/2409.10644. 
*   Monea et al. (2023) Giovanni Monea, Armand Joulin, and Edouard Grave. 2023. [Pass: Parallel speculative sampling](https://doi.org/10.48550/ARXIV.2311.13581). _CoRR_, abs/2311.13581. 
*   Ouyang et al. (2024) Siru Ouyang, Shuohang Wang, Minhao Jiang, Ming Zhong, Donghan Yu, Jiawei Han, and Yelong Shen. 2024. [Temperature-centric investigation of speculative decoding with knowledge distillation](https://aclanthology.org/2024.findings-emnlp.767). In _Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024_, pages 13125–13137. Association for Computational Linguistics. 
*   Rein et al. (2023) David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. [GPQA: A graduate-level google-proof q&a benchmark](https://doi.org/10.48550/ARXIV.2311.12022). _CoRR_, abs/2311.12022. 
*   Sun et al. (2023) Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix X. Yu. 2023. [Spectr: Fast speculative decoding via optimal transport](http://papers.nips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html). In _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_. 
*   Team (2024) Qwen Team. 2024. [Qwq: Reflect deeply on the boundaries of the unknown](https://qwenlm.github.io/blog/qwq-32b-preview/). 
*   Wolf et al. (2019) Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. 2019. [Huggingface’s transformers: State-of-the-art natural language processing](https://arxiv.org/abs/1910.03771). _CoRR_, abs/1910.03771. 
*   Xia et al. (2024) Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. 2024. [Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding](https://doi.org/10.18653/V1/2024.FINDINGS-ACL.456). In _Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024_, pages 7655–7671. Association for Computational Linguistics. 
*   Yang et al. (2024) An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and 43 others. 2024. [Qwen2 technical report](https://doi.org/10.48550/ARXIV.2407.10671). _CoRR_, abs/2407.10671. 
*   Yin et al. (2024) Ming Yin, Minshuo Chen, Kaixuan Huang, and Mengdi Wang. 2024. [A theoretical perspective for speculative decoding algorithm](https://doi.org/10.48550/ARXIV.2411.00841). _CoRR_, arXiv:2411.00841. 
*   Zhang et al. (2024) Situo Zhang, Hankun Wang, Da Ma, Zichen Zhu, Lu Chen, Kunyao Lan, and Kai Yu. 2024. [Adaeagle: Optimizing speculative decoding via explicit modeling of adaptive draft structures](https://doi.org/10.48550/ARXIV.2412.18910). _CoRR_, abs/2412.18910. 
*   Zhao et al. (2024) Juntao Zhao, Wenhao Lu, Sheng Wang, Lingpeng Kong, and Chuan Wu. 2024. [Qspec: Speculative decoding with complementary quantization schemes](https://doi.org/10.48550/ARXIV.2410.11305). _CoRR_, abs/2410.11305. 
*   Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. [Judging llm-as-a-judge with mt-bench and chatbot arena](http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html). In _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_. 

Appendix A The Complete Speculative Decoding Algorithms
-------------------------------------------------------

In Algorithm[2](https://arxiv.org/html/2411.18462v2#alg2 "Algorithm 2 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") to [6](https://arxiv.org/html/2411.18462v2#alg6 "Algorithm 6 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we present the complete algorithms of the vanilla speculative decoding in both the greedy decoding and the sampling scenarios. For the sampling scenario, the Verify and Correct methods in Algorithm[6](https://arxiv.org/html/2411.18462v2#alg6 "Algorithm 6 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") resolve to Algorithm[2](https://arxiv.org/html/2411.18462v2#alg2 "Algorithm 2 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") and [4](https://arxiv.org/html/2411.18462v2#alg4 "Algorithm 4 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). For greedy decoding, they resolve to Algorithm[3](https://arxiv.org/html/2411.18462v2#alg3 "Algorithm 3 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation") and [5](https://arxiv.org/html/2411.18462v2#alg5 "Algorithm 5 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation").

Algorithm 2 Verify (Sampling)

1:target distribution

p​(x)p(x)
, draft distribution

q​(x)q(x)
, draft token

x t x_{t}

2:

a​c​c​e​p​t←accept\leftarrow
False

3:

r∼U​[0,1]r\sim U[0,1]

4:if

r<p​(x t)q​(x t)r<\frac{p(x_{t})}{q(x_{t})}
then

5:

a​c​c​e​p​t←accept\leftarrow
True

6:end if

7:

a​c​c​e​p​t accept

Algorithm 3 Verify (Greedy)

1:target distribution

p​(x)p(x)
, draft distribution

q​(x)q(x)
, draft token

x t x_{t}

2:

a​c​c​e​p​t←accept\leftarrow
False

3:if

arg max p(x)==x t\arg\max p(x)==x_{t}
then

4:

a​c​c​e​p​t←accept\leftarrow
True

5:end if

6:

a​c​c​e​p​t accept

Algorithm 4 Correct (Sampling)

1:target distribution

p​(x)p(x)
, draft distribution

q​(x)q(x)

2:Sample

x^∼max⁡(q​(x)−p​(x), 0)∑i max⁡(q​(x i)−p​(x i), 0)\hat{x}\sim\frac{\max(q(x)-p(x),\,0)}{\sum_{i}\max(q(x^{i})-p(x^{i}),\,0)}

3:

x^\hat{x}

Algorithm 5 Correct (Greedy)

1:target distribution

p​(x)p(x)
, draft distribution

q​(x)q(x)

2:

arg⁡max⁡p​(x)\arg\max p(x)

Algorithm 6 Speculative Decoding

1:target model

p p
, draft model

q q
, input sequence

x⩽t x_{\leqslant t}
, maximum length

T T
, draft length

γ\gamma

2:Initialize

n←t n\leftarrow t

3:while

n<T n<T
do

4:for

j=1 j=1
to

γ\gamma
do

5: Sample

x n+j∼q​(x|x<n+j)x_{n+j}\sim q(x|x_{<{n+j}})

6:end for

7: Compute

p​(x|x<n+j),j=1,⋯,γ+1 p(x|x_{<{n+j}}),\;j=1,\cdots,\gamma+1
in parallel

8:

n~←n\tilde{n}\leftarrow n

9:for

j=1 j=1
to

γ\gamma
do

10:if Verify

(p​(x|x<n+j),q​(x|x<n+j),x n+j)\left(p(x|x_{<n+j}),\,q(x|x_{<n+j}),x_{n+j}\right)
then

11:

n~←n~+1\tilde{n}\leftarrow\tilde{n}+1

12:else

13:

x n+j←Correct​(p​(x|x<n+j),q​(x|x<n+j))x_{n+j}\leftarrow\text{Correct}\left(p(x|x_{<n+j}),\,q(x|x_{<n+j})\right)

14: Exit for loop

15:end if

16:end for

17:if

n~==n+γ\tilde{n}==n+\gamma
then

18:

x n+γ+1∼p​(x|x⩽n+γ)x_{n+\gamma+1}\sim p(x|x_{\leqslant n+\gamma})

19:end if

20:

n←n~+1 n\leftarrow\tilde{n}+1

21:end while

22:

x⩽n x_{\leqslant n}

We note that from Algorithm[2](https://arxiv.org/html/2411.18462v2#alg2 "Algorithm 2 ‣ Appendix A The Complete Speculative Decoding Algorithms ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), it’s straightforward that the acceptance rate of a draft token x t x_{t} in the sampling scenario is by definition min⁡(1,p​(x t)q​(x t))\min(1,\frac{p(x_{t})}{q(x_{t})}).

Appendix B Alternatives for Acceptance Rate Lower Bound Computation
-------------------------------------------------------------------

In Section[2](https://arxiv.org/html/2411.18462v2#S2 "2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we used Pinsker’s inequality to compute a lower bound for the expected acceptance probability:

β\displaystyle\beta=∑x min⁡(p​(x),q​(x))\displaystyle=\sum_{x}\min\left(p(x),q(x)\right)(10)
⩾1−1 2 𝕂 𝕃(q||p).\displaystyle\geqslant 1-\sqrt{\frac{1}{2}\mathbb{KL}(q||p)}.(11)

Another way to compute the lower bound of acceptance probability can be derived from Bretagnolle-Huber inequality(Bretagnolle and Huber, [1978](https://arxiv.org/html/2411.18462v2#bib.bib1)):

β\displaystyle\beta⩾1−1−e−𝕂 𝕃(q||p).\displaystyle\geqslant 1-\sqrt{1-e^{-\mathbb{KL}(q||p)}}.(12)

Compared with the Pinsker’s bound, it’s trivial to see that this bound is guaranteed to be always larger than 0. However, in practice we find that the Pinsker’s bound is about 11% tighter.

Appendix C γ\gamma Approximation
--------------------------------

### C.1 Approximation Bound

Following Eq.([4](https://arxiv.org/html/2411.18462v2#S2.E4 "Equation 4 ‣ Lower Bound of Acceptance ‣ 2.2 Our Method: SVIP ‣ 2 Draft Model Knows When to Stop ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation")), the acceptance rate β\beta satisfies:

β≥1−1 2 𝕂 𝕃(q||p)=1−1 2​H q,p−1 2​H q\beta\geq 1-\sqrt{\frac{1}{2}\mathbb{KL}(q||p)}=1-\sqrt{\frac{1}{2}H_{q,p}-\frac{1}{2}H_{q}}

We denote the above bound as the actual bound. While this bound is theoretically sound, it relies on the exact access to the H q,p H_{q,p} the cross entropy between target and the draft models, which is inaccessible during the SD drafting phase. To address this, approximate H q,p H_{q,p} with γ​H q\gamma H_{q}, where γ\gamma is a random variable to describe the ratio between H q,p H_{q,p} and H q H_{q}, i.e. γ=H q,p/H q\gamma=H_{q,p}/H_{q}, we could rewrite the bound:

𝕂 𝕃(q||p)=H q,p−H q=(γ−1)H q\displaystyle\mathbb{KL}(q||p)=H_{q,p}-H_{q}=(\gamma-1)H_{q}
β≥1−1 2​(γ−1)​H q\displaystyle\beta\geq 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}

To make this bound more practical, we approximate it using a constant c c, obtaining

β≥1−1 2​(γ−1)​H q≈1−c​H q\beta\geq 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}\approx 1-\sqrt{cH_{q}}

We denote the above bound as the approximation bound. Since γ\gamma is a random variable, the tightness and reliability of this approximation depend on how well c c aligns with γ\gamma’s behavior. Specifically, we need our approximation bound is smaller than the actual bound:

β≥1−1 2​(γ−1)​H q≥1−c​H q\beta\geq 1-\sqrt{\frac{1}{2}(\gamma-1)H_{q}}\geq 1-\sqrt{cH_{q}}

Simplify the right side inequality:

γ≤2​c+1\displaystyle\gamma\leq 2c+1(13)

### C.2 Theoretical Analysis

Now, from the γ\gamma’s distribution in Figure 5, let’s analyze the probability that the lower bounds hold by modeling γ\gamma’s distribution.

#### Gaussian Distribution

Let’s assume γ∼N​(μ,σ 2)\gamma\sim N(\mu,\sigma^{2}), and the probability that the bound holds is:

P​(γ≤2​c+1)=Φ​(2​c+1−μ σ)P(\gamma\leq 2c+1)=\Phi(\frac{2c+1-\mu}{\sigma})(14)

where Φ\Phi is the standard normal CDF. It demonstrates that:

*   •If c c is small (e.g. 2​c+1≤μ 2c+1\leq\mu), the probability that the bound holds is low. 
*   •If c c is large (e.g. 2​c+1≥μ 2c+1\geq\mu), the bound holds with high probability; however the bound itself becomes loose. 

#### Gamma Distribution

Given the right-skewed nature of γ≥1\gamma\geq 1, we model as a shifted Gamma distribution: γ=1+X\gamma=1+X, where X∼Gamma​(α,β)X\sim\text{Gamma}(\alpha,\beta). The conditions for the bound to hold is :

P​(γ≤2​c+1)=P​(X≤2​c)=γ​(α,β⋅2​c)Γ​(α)P(\gamma\leq 2c+1)=P(X\leq 2c)=\frac{\gamma(\alpha,\beta\cdot 2c)}{\Gamma(\alpha)}(15)

where γ​(α,z)=∫0 z t α−1​e−t​𝑑 t\gamma(\alpha,z)=\int_{0}^{z}t^{\alpha-1}e^{-t}dt is the lower incomplete gamma function, and Γ​(α)\Gamma(\alpha) is the gamma function. This probability depends on c c, α\alpha, and β\beta.

*   •If c c is small (e.g. 2​c<𝔼​[x]=α/β 2c<\mathbb{E}[x]=\alpha/\beta), so the probability that our approximation further lowers the actual bound is low. 
*   •It c c is large, the probability approaches 1, but the bound β≥1−c​H q\beta\geq 1-\sqrt{cH_{q}} is looser. 

Appendix D Additional Results on Long-form Generation
-----------------------------------------------------

In Table[5](https://arxiv.org/html/2411.18462v2#A4.T5 "Table 5 ‣ Appendix D Additional Results on Long-form Generation ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"), we present the results of greedy decoding using Qwen2.5 14B as target and 0.5B as draft, and find that the speedup ratio of greedy decoding is much higher compared with the sampling experiments in Section[3.1](https://arxiv.org/html/2411.18462v2#S3.SS1 "3.1 Results on Long-form Generation ‣ 3 Experiments ‣ Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation"). Further investigation suggests that this is a result of repetition hallucination in both target and draft models during long-form greedy generation.

Table 5: Results of greedy decoding on MT-Bench. Greedy decoding leads to repetition hallucinations in both target and draft models in long-form generation, resulting in exaggerated speedup ratio.

