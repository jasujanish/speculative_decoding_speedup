Title: BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms

URL Source: https://arxiv.org/html/2505.15141

Markdown Content:
Back to arXiv

This is experimental HTML to improve accessibility. We invite you to report rendering errors. 
Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off.
Learn more about this project and help improve conversions.

Why HTML?
Report Issue
Back to Abstract
Download PDF
 Abstract
1Introduction
2Preliminaries
3Bandits for Adaptive Speculative Decoding
4Modeling Tokens Stochastically
5Modeling Tokens Adversarially
6Experiments
7Conclusions and Discussions
Acknowledgements:
License: CC BY 4.0
arXiv:2505.15141v2 [cs.LG] 20 Nov 2025
BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms
Yunlong Hou
Fengzhuo Zhang
Cunxiao Du
Xuan Zhang
Jiachun Pan
Tianyu Pang
Chao Du
Vincent Y. F. Tan
Zhuoran Yang
Abstract

Speculative decoding has emerged as a popular method to accelerate the inference of Large Language Models (LLMs) while retaining their superior text generation performance. Previous methods either adopt a fixed speculative decoding configuration regardless of the prefix tokens or train draft models in an offline or online manner to align them with the context. This paper proposes a training-free online learning framework to adaptively choose the configuration of the hyperparameters for speculative decoding as text is being generated. We first formulate this hyperparameter selection problem as a Multi-Armed Bandit problem and provide a general speculative decoding framework BanditSpec. Furthermore, two bandit-based hyperparameter selection algorithms, UCBSpec and EXP3Spec, are designed and analyzed in terms of a novel quantity, the stopping time regret. We upper bound this regret under both stochastic and adversarial reward settings. By deriving an information-theoretic impossibility result, it is shown that the regret performance of UCBSpec is optimal up to universal constants. Finally, extensive empirical experiments with LLaMA3 and Qwen2 demonstrate that our algorithms are effective compared to existing methods, and the throughput is close to the oracle best hyperparameter in simulated real-life LLM serving scenarios with diverse input prompts.

Machine Learning, ICML
1Introduction
Figure 1:Given the prefix tokens and the candidate hyperparameter configurations (e.g., models), which configuration should be selected to decode the next tokens? We formulate this problem as a bandit problem and propose a general framework BanditSpec.

A Large Language Model (LLM) is trained to predict the probability of the next token conditioned on all previous tokens (Brown2020LanguageMA; Touvron2023Llama2O). This autoregressive decoding approach involves multiple forward passes, with each pass generating one token sequentially. Consequently, this process can lead to significant latency during inference.

Speculative decoding was introduced by leviathan2023fast; chen2023accelerating to accelerate the inference of LLMs. The standard speculative decoding framework has been extended with improved performance since then. A thorough overview is presented at Appendix A. While the existing speculative decoding methods are diverse, most previous works adopt a fixed one across tasks, severely limiting their potential. For instance, when dealing with code debugging or grammar-checking tasks, the generated tokens are expected to resemble most of the input tokens. Therefore, retrieval-based speculative decoding techniques are preferred (hu2024sam). In contrast, for story generation tasks, we expect the generated tokens to be more creative. Thus, a draft model with a high-temperature parameter is preferred over retrieval-based methods. The potential of these speculative decoding methods can only be exploited when the configuration of hyperparameters is well-aligned to the given task. There are existing works that attempt to achieve this goal, e.g., zhou2024distillspec distills the draft model during inference. Furthermore, even when the choice of the draft model is optimized, the associated hyperparameters can still be refined. For instance, liu2024optimizing and huang2024specdec++ aim to optimize the speculation length in a training and training-free manner. Based on these observations, we ask the questions (see Figure 1) : given prefix prompts and candidate configurations of hyperparameters, is there a theoretically sound framework to model and solve the hyperparameters selection problem? Is there any training-free method that can adaptively choose the hyperparameters such that the latency of speculative decoding can be minimized?

In this paper, we answer these questions affirmatively. We adopt a bandit framework to leverage its adaptivity in unknown environments to achieve this goal. Our contributions can be summarized as follows.

∙
 We formulate the hyperparameter selection problem in speculative decoding as a bandit problem and propose a general speculative decoding framework 
BanditSpec
​
(
𝙰𝙻𝙶
)
 (see Algorithm 3), where the hyperparameter selection algorithm 
𝙰𝙻𝙶
 selects the hyperparameters to be deployed in each round of speculative decoding. The objective is to minimize the stopping time regret, which measures the latency of 
𝙰𝙻𝙶
 compared to that of the best hyperparameter.

∙
 Under mild stochastic and adversarial reward assumptions, we devise two hyperparameter selection algorithms, UCBSpec and EXP3Spec, respectively. By deriving upper bounds on the stopping time regret, we prove that the inference latency between the proposed algorithms and the best hyperparameter under a given initial prompt vanishes asymptotically. In addition, we show, via deriving an information-theoretic impossibility result, that the regret performance of UCBSpec is optimal up to constants.

∙
 Extensive empirical experiments with LLaMA3 and Qwen2 are conducted to demonstrate the efficacy of the proposed framework. When the batch size is 
1
, the adaptive selection of models via UCBSpec and EXP3Spec can greatly improve the latency, exhibiting competitive performance against the existing methods. Under simulated real-life scenarios where LLMs are implemented for diverse prompts simultaneously, the adaptive selection of speculation length via UCBSpec achieves comparable throughput with the oracle best.

2Preliminaries

LLM Decoding We denote an LLM as 
𝑃
:
𝒳
∗
→
𝚫
𝒳
, where 
𝒳
 and 
𝒳
∗
 are the space of tokens and the space of all token sequences, respectively. Most LLMs predict this conditional probability via predicting the logits of the next token. Concretely, the LLMs predict 
log
⁡
𝑃
​
(
𝑥
𝑡
|
𝑥
1
:
𝑡
−
1
)
, where 
𝑥
𝑡
∈
𝒳
 and 
𝑥
1
:
𝑡
−
1
∈
𝒳
∗
 are respectively the 
𝑡
-th token and first 
𝑡
−
1
 tokens. In the inference stage, LLMs use an additional temperature parameter 
𝛾
>
0
 to predict the next token’s probability as 
softmax
(
𝛾
−
1
log
𝑃
(
⋅
|
𝑥
1
:
𝑡
−
1
)
)
, where softmax is the softmax operator. The results in our work hold for any 
𝛾
>
0
, and we just denote 
softmax
​
(
𝛾
−
1
​
log
⁡
𝑃
)
 as 
𝑃
 for ease of notation. When 
𝛾
>
0
, we sample the next token from the predicted distribution, which is called sampling (sampling decoding). When 
𝛾
↓
0
, the next token will be the token that corresponds to the highest logit value; this is called greedy decoding. We note that the greedy decoding is deterministic, i.e., the token is sampled from a degenerate distribution. These two families of decoding methods can be unified as Algorithm 1, where LLMs autoregressively generate tokens until the 
EOS
 token.

Inputs: initial prompt 
pt
0
=
pt
∈
𝒳
∗
, target model 
𝑃
.
Procedures:

Algorithm 1 Canonical Decoding
1: Set 
𝑡
=
0
.
2: while 
𝑡
≠
0
 and 
𝑥
𝑡
≠
EOS
 do
3:  
𝑡
=
𝑡
+
1
.
4:  
𝑥
𝑡
∼
𝑃
(
⋅
∣
pt
𝑡
−
1
)
.
5:  
pt
𝑡
=
concat
​
(
pt
𝑡
−
1
,
𝑥
𝑡
)
.
6: end while
7: return 
𝑡
,
pt
𝑡

Speculative Decoding As shown in Algorithm 1, the autoregressive decoding feature requires multiple forward inferences of LLMs 
𝑃
 sequentially. To reduce the number of forward inferences, leviathan2023fast; chen2023accelerating proposed the vanilla speculative decoding algorithm, which implements a draft model 
𝑄
 to generate draft tokens and let the target model 
𝑃
 verify them in parallel. For completeness, we present and describe the vanilla speculative decoding algorithm in Appendix C.1. This vanilla speculative decoding is then extended by some existing works, e.g., miao2023SpecInferAG; cai2024medusa organises the draft tokens as a tree, which improves the number of accepted tokens. The speculative decoding algorithm contains many hyperparameters, e.g., the draft model 
𝑄
, and the tree structure in miao2023SpecInferAG; cai2024medusa. Most existing works keep these hyperparameters fixed for all the tasks. Some other works optimise the draft model in an online or offline manner (liu2023OnlineSD) and the size of the tree (chen2024SequoiaSR), which are designed for specific considerations. In contrast, our work aims to derive a unified online hyperparameter selection algorithm that can be applied for any type of hyperparameters.

Multi-Armed Bandits The Multi-Armed Bandit (MAB) is a fundamental online decision-making problem (see Algorithm 7 for its dynamics). In its classical stochastic form, an agent chooses from 
𝐾
 arms, each of which delivers a reward sampled i.i.d. from an unknown but fixed distribution when pulled (lattimore2020bandit). The goal is to select arms over 
𝑇
 rounds to maximise cumulative rewards. Two primary classes of algorithms—UCB-type (auer2002finite) and sampling-based methods (russo2017ATO)—have been developed and proven optimal in this setting. In the adversarial formulation, there are no assumptions on the reward distributions; rewards can evolve arbitrarily over time and may be correlated across arms (auer2002nonstochastic). Several algorithms, such as EXP3 and EXP4 (auer2002nonstochastic), are known to achieve optimal performance under these conditions. In this work, we frame the hyperparameter selection problem as an MAB problem and develop algorithms tailored to both stochastic and adversarial settings.

Notations: Let 
[
𝑁
]
:=
{
1
,
⋯
,
𝑁
}
. For a finite set 
𝒳
, we denote the set of distributions supported on it as 
𝚫
𝒳
=
{
𝑃
:
𝒳
→
[
0
,
1
]
|
∑
𝑥
∈
𝒳
𝑃
​
(
𝑥
)
=
1
,
𝑃
​
(
𝑥
)
≥
0
​
 for all 
​
𝑥
∈
𝒳
}
. The space of all finite-length sequences whose components belong to 
𝒳
 is denoted as 
𝒳
∗
, and we use 
𝑥
1
:
𝐿
∈
𝒳
𝐿
⊆
𝒳
∗
 to denote a length-
𝐿
 sequence. The Kullback–Leibler (
KL
) divergence between two distributions 
𝑃
 and 
𝑄
 is denoted as 
KL
​
(
𝑃
,
𝑄
)
.

3Bandits for Adaptive Speculative Decoding

In the section, we formally formulate the hyperparameter selection problem in speculative decoding using the parlance of multi-armed bandits. The goal of this online decision-making process is to decode as soon as possible, i.e., minimizing the latency of the LLM decoding. Different from the classical multi-armed bandit problem, this problem involves two stochastic processes that march at various paces. In fact, as described in Appendix C.1, each (vanilla) speculative decoding subroutine produces several tokens, where the number of accepted tokens itself is also a random variable. Thus, the selection of hyperparameters of each speculative decoding subroutine and the token generation processes are evolving at different paces.

Inputs: 
pt
∈
𝒳
∗
, target model 
𝑃
, the hyperparameters 
𝑆
, maximum speculation length 
𝐿
.
Procedures:

Algorithm 2 Speculative Decoding Subroutine (SpecDecSub)
1: Call a standard speculative decoding algorithm with 
(
pt
,
𝑃
,
𝑆
,
𝐿
)
.
2: return the accepted and bonus tokens 
𝑥
1
:
𝜏
, where 
𝜏
≥
1
.

To put the problem in a mathematically sound way, we first specify a general speculative decoding subroutine (SpecDecSub) in Algorithm 2. The input of this subroutine is a prompt 
pt
∈
𝒳
∗
, a target model 
𝑃
, a specification of hyperparameters 
𝑆
, and the maximum speculation length 
𝐿
, and the output is the accepted token sequence 
𝑥
1
:
𝜏
∈
𝒳
∗
. We provide two examples of the hyperparameter sets here. (1) If we adopt the vanilla speculative decoding (Algorithm 6) as Line 1, 
𝑆
 can be different draft models 
𝑄
:
𝒳
∗
→
𝚫
𝒳
, and 
𝒮
 is the set of all the provided draft models. We would like to choose a draft model according to its training context, e.g. math, creative writing, to decode the current prefix. Then the problem we consider is how to adaptively select a proper draft model for speculative decoding via bandit algorithms. (2) If we adopt Medusa (cai2024medusa) as Line 1, 
𝑆
 can be different tree structures, and 
𝒮
 is the set of plausible tree structures. In this problem, we would like to adaptively adjust the speculation tree structure according to the context.

Inputs: arm selection algorithm 
𝙰𝙻𝙶
, initial prompt 
pt
0
=
pt
∈
𝒳
∗
, bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
.
Procedures:

Algorithm 3 Speculative Decoding with Bandits (BanditSpec)
1: 
𝑡
=
0
,
ℋ
0
=
∅
,
𝐼
0
=
1
,
𝑥
𝐼
0
,
0
=
∅
.
2: while 
EOS
∉
𝑥
𝐼
𝑡
,
𝑡
 do
3:  
𝑡
=
𝑡
+
1
.
4:  Select a hyperparameter index 
𝐼
𝑡
=
𝙰𝙻𝙶
​
(
ℋ
𝑡
−
1
)
.
5:  
𝑥
𝐼
𝑡
,
𝑡
=
SpecDecSub
​
(
pt
𝑡
−
1
,
𝑃
,
𝑆
𝐼
𝑡
,
𝐿
)
.
6:  
pt
𝑡
=
concat
​
(
pt
𝑡
−
1
,
𝑥
𝐼
𝑡
,
𝑡
)
.
7:  
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝐼
𝑡
,
𝑥
𝐼
𝑡
,
𝑡
)
)
.
8: end while
9: return 
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
=
𝑡
,
pt
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
=
pt
𝑡
.

With the help of SpecDecSub, the speculative decoding with a bandit framework, BanditSpec, can be specified in Algorithm 3 and as illustrated in Figure 1. The bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
 consists of three components: the target model 
𝑃
, the set of 
𝐾
 candidate hyperparameter specifications 
𝒮
, and the maximum speculation length 
𝐿
. Each hyperparameter specification 
𝑆
𝑖
∈
𝒮
 corresponds to an arm in the bandit problem. Given a prompt 
pt
 and an arm selection algorithm 
𝙰𝙻𝙶
, a hyperparameter specification is chosen according to the history 
ℋ
𝑡
−
1
 in Line 4. Then SpecDecSub is invoked with selected hyperparameters 
𝑆
𝐼
𝑡
 as input. The output of SpecDecSub, 
𝑥
𝐼
𝑡
,
𝑡
,1 is then adopted to update the prompt (Line 6) and the history information (Line 7). The whole process stops when the 
EOS
 token appears in the prompt. We denote the number of calls to SpecDecSub (the stopping time) and the generated token sequence as 
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
 and 
pt
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
, respectively. To minimize the decoding latency, we aim to design 
𝙰𝙻𝙶
 to minimize 
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
. Since the position of the 
EOS
 token itself is a random variable, we would like to minimize the expectation of 
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
. The performance of 
𝙰𝙻𝙶
 is measured via the stopping time regret

	
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
	
:=
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
∣
pt
,
𝜈
]
		
(1)

		
−
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
𝑖
∗
​
(
pt
,
𝜈
)
,
pt
,
𝜈
)
∣
pt
,
𝜈
]
,
		
(2)

where 
𝙰𝙻𝙶
𝑖
 is the arm selection algorithm which adopts 
𝑆
𝑖
 in all rounds, i.e., 
𝑖
=
𝙰𝙻𝙶
𝑖
​
(
ℋ
𝑡
)
 for all 
ℋ
𝑡
 and 
𝑡
, and 
𝑖
∗
​
(
pt
,
𝜈
)
=
argmin
𝑖
∈
[
𝐾
]
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
𝑖
,
pt
,
𝜈
)
∣
pt
,
𝜈
]
 denotes the index of the best hyperparameter for prompt 
pt
 under bandit configuration 
𝜈
.

For ease of notation, when 
𝜈
 and 
pt
 are clear from the context, 
ST
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
, 
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
, and 
𝑖
∗
​
(
pt
,
𝜈
)
 will be abbreviated as 
ST
​
(
𝙰𝙻𝙶
)
,
Reg
​
(
𝙰𝙻𝙶
)
, and 
𝑖
∗
, respectively. We will use 
BanditSpec
​
(
𝙰𝙻𝙶
)
 to specify the choice of 
𝙰𝙻𝙶
 in Algorithm 3. For simplicity, we regard the bonus token as the last accepted token. Thus, the length of the accepted tokens 
𝑥
𝐼
𝑡
,
𝑡
 at each round, 
𝑦
𝐼
𝑡
,
𝑡
, is between 
[
1
,
𝐿
+
1
]
.

Before going to the algorithm design and the theoretical analysis, we would like to specify some important properties that are shared by any arm selection algorithm 
𝙰𝙻𝙶
 and clarify the intuitions about our theoretical analysis. We denote the stopping time of the canonical decoding (Algorithm 1) and the generated sequence as 
𝜏
c
 and 
pt
𝜏
c
, respectively.

Proposition 3.1.

For any arm selection algorithm 
𝙰𝙻𝙶
 that selects an arm according to the history, the generated prompt 
pt
ST
​
(
𝙰𝙻𝙶
)
 is equal to 
pt
𝜏
c
 in distribution, i.e.,

	
pt
ST
​
(
𝙰𝙻𝙶
)
​
=
𝑑
​
pt
𝜏
c
,
 and 
​
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
​
=
𝑑
​
len
​
(
pt
𝜏
c
)
.
		
(3)

The stopping time 
ST
​
(
𝙰𝙻𝙶
)
 can be bounded as

	
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
𝐿
+
1
≤
ST
​
(
𝙰𝙻𝙶
)
≤
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
,
𝑎
.
𝑠
.
		
(4)

The proof of Proposition 3.1 is provided in Appendix D.1. This proposition states that the distribution of the generated prompt is the same as that of the prompt generated by Algorithm 1. The stopping time 
ST
​
(
𝙰𝙻𝙶
)
 is equal to the length of the generated prompt up to a constant. To facilitate our theoretical understanding, we pose the following question.

Question: Whether it is possible to devise an arm selection algorithm 
𝙰𝙻𝙶
 to achieve sublinear regret in terms of the length of the generated token sequence, i.e., is 
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
=
𝑜
​
(
𝔼
​
[
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
]
)
?

Interpretation of the Desired Result. Given a prompt 
pt
 and bandit configuration 
𝜈
, BanditSpec adaptively selects the hyperparameter via 
𝙰𝙻𝙶
 and learns the context. The stopping time regret (1) measures how the stopping time of 
BanditSpec
​
(
𝙰𝙻𝙶
)
 compares to that of the (agnostic) best one 
BanditSpec
​
(
𝙰𝙻𝙶
𝑖
∗
)
. By minimizing 
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
, we want to devise an 
𝙰𝙻𝙶
 to (approximately) match the performance of 
𝙰𝙻𝙶
𝑖
∗
. In particular, if 
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
=
𝑜
​
(
𝔼
​
[
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
]
)
, it implies that 
BanditSpec
​
(
𝙰𝙻𝙶
)
 requires the same number of speculative decoding rounds as 
BanditSpec
​
(
𝙰𝙻𝙶
𝑖
∗
)
 asymptotically even though the information about 
𝑆
𝑖
∗
 is not revealed at the beginning. In other words, 
BanditSpec
​
(
𝙰𝙻𝙶
)
 learns the identity of 
𝑆
𝑖
∗
 quickly and the price for this learning process can be amortized over time. Additionally, when 
BanditSpec
​
(
𝙰𝙻𝙶
)
 is deployed over diverse prompt inputs, we expect a significant acceleration of token generation compared to any fixed single speculative decoding method.

Why do we consider stochastic and adversarial settings? To derive efficient algorithms and meaningful theoretical analysis, it is necessary to make certain plausible assumptions of the problem. For the BanditSpec problem, we need to model the stochasticity of the number of accepted tokens for each hyperparameter specification. We highlight that in real-world applications, they are far from identically and independently distributed. The stochastic case (Section 4) models it as random variables and only assumes that each hyperparameter will have a stationary mean acceptance length (Assumption 4.1) without the independence assumption. The adversarial case (Section 5) removes this stationarity assumption and does not make any distributional assumption of the number of accepted tokens for each hyperparameter. We highlight that there is no explicit adversary in the speculative decoding, but we model the stochasticity of the number of accepted tokens as the randomness from an (imaginary) adversary.

4Modeling Tokens Stochastically
Figure 2:Illustration of our bandit model for choosing configurations to decode the next token, where UCB and EXP3 refer to UCBSpec and EXP3Spec, respectively.

In this section, we model the length of the accepted tokens as random variables.

Assumption 4.1 (Stationary Mean Values).

There exist 
𝐾
 values 
{
𝜇
𝑖
}
𝑖
∈
[
𝐾
]
⊂
[
1
,
𝐿
+
1
]
, such that conditioned on the history 
ℋ
𝑡
−
1
 and the chosen arm 
𝐼
𝑡
 at time 
𝑡
, the expected number of the accepted tokens 
𝔼
​
[
𝑌
𝐼
𝑡
,
𝑡
∣
ℋ
𝑡
−
1
,
𝐼
𝑡
]
=
𝜇
𝐼
𝑡
.

This assumption assumes that the conditional expectation of the number of accepted tokens for each hyperparameter is equal to a fixed number conditioned on the previous tokens. We emphasize that this assumption does not require independence between the number of accepted tokens across implementations of SpecDecSub, which would be unrealistic in real-world applications. More discussions are provided in Appendix B.1.

4.1Upper Bounds for the Stochastic Case

Algorithm Design We design a UCB-type arm selection algorithm UCBSpec, as shown in Algorithm 4. To avoid additional terms, we call the aggregated algorithm, 
BanditSpec
​
(
UCBSpec
)
, as UCBSpec. The full version of UCBSpec is detailed in Algorithm 8.

This aggregated algorithm, UCBSpec, is adapted from the classical UCB-1 algorithm in auer2002finite. The main differences are the confidence radius design 
cr
𝑖
,
𝑡
 and the stopping rule. We highlight that the form of 
cr
𝑖
,
𝑡
 is designed to fit the weak assumption of the number of accepted tokens. In fact, the proof of the regret of UCB-1 assumes that the values of each arm are generated before the pull of arms (auer2002finite; lattimore2020bandit), which bifurcates from practical LLM inference scenarios. In contrast, we remove this strong restriction. The stopping rule of UCBSpec makes the analysis of our algorithm rather different from that of UCB-1. The stochasticity of the total number of arm pulls requires a novel regret decomposition analysis that is not presented in previous works.

Algorithm 4 UCBSpec

Inputs: number of hyperparameter specifications 
𝐾
, history 
ℋ
𝑡
=
(
(
𝐼
𝑠
,
𝑋
𝐼
𝑠
,
𝑠
)
)
𝑠
=
1
𝑡
, confidence parameter 
𝛿
.
Procedures:

1: if 
𝑡
≤
𝐾
−
1
 then return 
𝐼
𝑡
+
1
=
𝑡
+
1
.
2: Compute the lengths 
𝑌
𝐼
𝑠
,
𝑠
=
len
​
(
𝑋
𝐼
𝑠
,
𝑠
)
 for all 
𝑠
∈
[
𝑡
]
.
3: Set the statistics 
{
𝜇
^
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
,
{
UCB
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
, where
	
𝑛
𝑖
,
𝑡
=
∑
𝑠
=
1
𝑡
𝟙
​
{
𝐼
𝑠
=
𝑖
}
,
𝜇
^
𝑖
,
𝑡
=
∑
𝑠
=
1
𝑡
𝑌
𝑖
,
𝑠
​
𝟙
​
{
𝐼
𝑠
=
𝑖
}
𝑛
𝑖
,
𝑡
,
		
(5)

	
cr
𝑖
,
𝑡
=
𝐿
2
​
1
+
𝑛
𝑖
,
𝑡
𝑛
𝑖
,
𝑡
2
​
(
1
+
2
​
log
⁡
𝐾
​
𝑡
2
​
(
1
+
𝑛
𝑖
,
𝑡
)
1
2
𝛿
)
,
		
(6)

	
UCB
𝑖
,
𝑡
=
𝜇
^
𝑖
,
𝑡
+
cr
𝑖
,
𝑡
.
		
(7)
4: return index 
𝐼
𝑡
+
1
=
argmax
𝑖
∈
[
𝐾
]
UCB
𝑖
,
𝑡
.

Theoretical Analysis We first state an assumption.

Assumption 4.2 (Finite Generation Length).

Given any prompt 
pt
∈
𝒳
∗
, the expected length of the output sequence of the canonical decoding algorithm (Algorithm 1) is finite, i.e., 
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
<
∞
.

This assumption states that the expected length of the generated prompt is finite. In real-world applications, the length of the generated prompt is always finite due to the limits of computation and storage.

To state our main result, we denote the suboptimality gap between the best arm 
𝑖
∗
:=
argmax
𝑖
∈
[
𝐾
]
𝜇
𝑖
 and arm 
𝑖
 as 
Δ
𝑖
:=
𝜇
𝑖
∗
−
𝜇
𝑖
. Define the hardness parameter 
H
​
(
pt
,
𝜈
)
:=
∑
𝑖
≠
𝑖
∗
1
/
(
𝜇
𝑖
∗
​
Δ
𝑖
)
, which captures the difficulty of acceleration given the initial prompt 
pt
 and bandit configuration 
𝜈
.

Theorem 4.3 (Upper Bound).

Under Assumptions 4.1 and 4.2, given any prompt 
pt
∈
𝒳
∗
 and bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
, the expected stopping time regret of Algorithm 3 with 
𝙰𝙻𝙶
=
Algorithm 4 (UCBSpec) is upper bounded as

	
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
=
𝑂
​
(
H
​
(
pt
,
𝜈
)
⋅
𝐿
2
​
log
⁡
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
)
.
		
(8)

Theorem 4.3 answers the proposed question in Section 3 in the affirmative under Assumptions 4.1 and 4.2. To interpret the results of the theorem, for each hyperparameter 
𝑆
𝑖
, it requires 
𝑛
𝑖
=
𝑂
(
𝐿
2
log
𝔼
[
len
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
]
/
Δ
𝑖
2
)
)
 pulls to identify that 
𝑆
𝑖
 is suboptimal under the current prompt 
pt
 and bandit configuration 
𝜈
, resulting in 
𝑛
𝑖
​
Δ
𝑖
 token loss compared to the case where 
𝑆
𝑖
∗
 had been adopted. Additionally, this loss could be compensated by 
𝑛
𝑖
​
Δ
𝑖
/
𝜇
𝑖
∗
 pulls of 
𝑆
𝑖
∗
, which constitutes the final stopping time regret bound. The proof is postponed to Appendix D.2 with more discussions in Appendix B.2.

4.2Lower Bound for the Stochastic Case

We further provide an information-theoretic lower bound of the regret under the greedy decoding strategy to indicate how the upper bound is in Theorem 4.3. More details and the proof of Theorem 4.4 are deferred to Appendix D.4.

Theorem 4.4 (Lower Bound).

Given any sequence of initial prompts 
(
pt
𝑚
)
𝑚
=
1
∞
⊂
𝒳
init
∗
 with 
len
​
(
pt
𝜏
c
𝑚
)
→
∞
,
𝑚
→
∞
 and a bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
, under Assumption D.4, the greedy decoding strategy and the dynamics represented in Algorithm 3, for any non-anticipatory and consistent arm selection algorithm 
𝙰𝙻𝙶
, the expected regret satisfies

	
lim inf
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
1
kl
𝑖
,
		
(9)

where 
kl
𝑖
:=
inf
𝑆
∈
𝒮
{
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
)
:
𝔼
𝑋
∼
𝑃
𝑆
​
[
𝑋
]
>
𝜇
𝑖
∗
}
.

To provide a more concrete example of the lower bound, consider the truncated geometric distribution (TGD) on 
[
1
,
𝐿
+
1
]
 with parameter 
𝑝
∈
(
0
,
1
)
, i.e.,

	
𝑃
𝑆
(
𝑥
)
=
{
	
𝑝
𝑥
−
1
​
(
1
−
𝑝
)
,
		
𝑥
=
1
,
2
,
…
,
𝐿
,

	
𝑝
𝐿
,
		
𝑥
=
𝐿
+
1
.
		
(10)

This TGD was considered in the seminal works on speculative decoding (leviathan2023fast; chen2023accelerating).

Proposition 4.5 (Tightness Result).

Let 
𝒮
TGD
=
{
𝑆
:
𝑃
𝑆
​
 satisfies (
10
)
}
. Let 
{
𝑆
𝑖
}
𝑖
=
1
𝐾
⊂
𝒮
TGD
 and 
𝑆
𝑖
 satisfies (10) with 
𝑝
𝑖
 (Line 5 in Algorithm 2), then

	
lim inf
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
𝑚
,
𝜈
)
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
H
​
(
pt
,
𝜈
)
⋅
𝑝
𝑖
∗
​
(
1
−
𝑝
𝑖
∗
𝐿
)
(
1
−
𝑝
𝑖
∗
)
.
		
(11)

Therefore, the upper and lower bound match up absolute constants and a 
𝐿
2
​
(
1
−
𝑝
𝑖
∗
)
𝑝
𝑖
∗
​
(
1
−
𝑝
𝑖
∗
𝐿
)
 factor. In particular, if 
𝑝
𝑖
∗
∈
(
2
−
1
/
𝐿
,
1
)
, they match up to absolute constants and 
𝐿
.

The proof is deferred to Appendix E.3. Proposition 4.5 indicates UCBSpec is optimal up to constants and 
𝐿
 when considering the TGD. In other words, the additional speculative decoding rounds of UCBSpec not only achieves 
𝑂
​
(
log
⁡
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
)
 compared to 
𝙰𝙻𝙶
𝑖
∗
, but is also among the best possible for any arm selection algorithm (up to constants).

For the tightness of our algorithm, according to Note 15.3 in lattimore2020bandit, 
kl
𝑖
=
𝑂
​
(
Δ
𝑖
2
)
 when 
Δ
𝑖
 is small. This indicates the dominating terms in the upper bound in Theorem 4.3 match the lower bound in Theorem 4.4 up to (possibly instance-dependent) constants. Furthermore, because the truncated geometric distribution is closer to a sub-exponential family distribution, especially when 
𝐿
 is large, bandit algorithms built upon UCB1 (auer2002finite) are generally loose in some factors. In order to close the gap between the upper and lower bounds completely, KL-UCB (garivier2011KLUCB) can possibly be adapted to this problem out of theoretical interest. However, on the practical side, KL-UCB demands solving an optimization problem at each round, which can be time-consuming during implementations. Thus, it does not perfectly align with our ultimate goal of LLM inference acceleration.

5Modeling Tokens Adversarially

In this section, we weaken Assumption 4.1 in Section 4 and consider a more general case. Specifically, we make the following assumption on the number of accepted tokens.

Assumption 5.1 (Adversarial Mean Values).

Let the number of accepted tokens generated by hyperparameter 
𝑆
𝑖
 at time step 
𝑡
 be 
𝑦
𝑖
,
𝑡
=
len
​
(
𝑋
𝑖
,
𝑡
)
. We assume 
{
𝑦
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
,
𝑡
∈
ℕ
 is fixed by the environment before the algorithm starts.

The bandit problem with this assumption is often referred to as the oblivious adversarial bandits in the online learning works (auer2002nonstochastic; lattimore2020bandit). It admits more general and practical setups compared to the stochastic MAB. We find the greedy decoding strategy aligns more closely to this setup in the sense that the generated tokens by the models are (potentially) fixed given the initial prompt. Hence, we present our result under the greedy decoding strategy in this section.2 Given a prompt 
pt
∈
𝒳
∗
 and a bandits configuration 
𝜈
, the stopping time regret (1) of an arm selection algorithm 
𝙰𝙻𝙶
 becomes

	
Reg
​
(
𝙰𝙻𝙶
)
:=
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
min
𝑖
∈
[
𝐾
]
⁡
ST
​
(
𝙰𝙻𝙶
𝑖
)
		
(12)

It is worth pointing out that under the greedy decoding strategy, the stopping time of any proposed algorithm can still be random due to the internal randomness embedded in the algorithm. For instance, the choice of hyperparameter 
𝑆
𝐼
𝑡
 in Line 5 in Algorithm 5.

Algorithm 5 EXP3Spec

Inputs: number of hyperparameter specifications 
𝐾
, history 
ℋ
𝑡
=
(
(
𝐼
𝑠
,
𝑋
𝐼
𝑠
,
𝑠
)
)
𝑠
=
1
𝑡
.
Procedures:

1: Compute the lengths 
𝑌
𝐼
𝑠
,
𝑠
=
len
​
(
𝑋
𝐼
𝑠
,
𝑠
)
 for all 
𝑠
∈
[
𝑡
]
.
2: Set the statistics for all 
𝑖
∈
[
𝐾
]
	
𝑍
^
𝑖
,
𝑡
=
𝟙
​
{
𝑖
=
𝐼
𝑡
}
⋅
𝐿
+
1
−
𝑌
𝑖
,
𝑡
𝐿
⋅
𝑝
𝑡
,
𝑖
.
		
(13)
3: Set learning rate 
𝜂
𝑡
=
log
⁡
𝐾
/
(
𝑡
⋅
𝐾
)
.
4: Set probability vector 
𝑝
𝑡
∈
𝚫
[
𝐾
]
 with for all 
𝑖
∈
[
𝐾
]
	
𝑝
𝑡
,
𝑖
=
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑖
,
𝑠
)
∑
𝑗
=
1
𝐾
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑗
,
𝑠
)
.
		
(14)
5: return hyperparameter index 
𝐼
𝑡
+
1
∼
𝑝
𝑡
.

Algorithm Design We present our arm selection algorithm in Algorithm 5, which is an abridged version of the full version 
BanditSpec
​
(
EXP3Spec
)
 delineated in Algorithm 9. This algorithm modifies the anytime EXP3 algorithm (lattimore2020bandit) to suit the speculative decoding application. In terms of the algorithm design, the main difference lies in the change of the stopping rule. We highlight that while the stopping time of the algorithm is random, the anytime feature of Algorithm 5 does not require any information about the time horizon. This is achieved by the vanishing sequence of learning rates 
{
𝜂
𝑡
}
𝑡
∈
ℕ
 which can be elegantly adapted to the unknown stopping time. With regard to the analysis, previous works (auer2002nonstochastic; bubeck2012regret; lattimore2020bandit) only consider the gap between the cumulated rewards over the same fixed horizon 
𝑇
, i.e., 
max
𝑖
∈
[
𝐾
]
⁡
𝔼
​
[
∑
𝑡
∈
[
𝑇
]
𝑦
𝑖
,
𝑡
−
𝑦
𝐼
𝑡
,
𝑡
]
. In contrast, we need to upper bound the stopping time regret in (12) where the baseline 
𝙰𝙻𝙶
𝑖
 and any proposed algorithm 
𝙰𝙻𝙶
 have different termination times in general. Thus, the analysis is much more involved.

Theoretical Analysis To ease the analysis, we make an assumption on the stopping time of Algorithm 5.

Assumption 5.2 (Stopping Time assumption).

Given a prompt 
pt
∈
𝒳
∗
 and configuration 
𝜈
, let 
𝑖
∗
:=
argmin
𝑖
∈
[
𝐾
]
ST
​
(
𝙰𝙻𝙶
𝑖
)
. We assume that 
ST
​
(
𝙰𝙻𝙶
)
>
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
 almost surely.

In speculative decoding, when the initial prompt is given, there generally exists a hyperparameter that has the highest acceptance rate in most rounds compared to the rest of the hyperparameters. As bandit algorithms will explore those suboptimal hyperparameters, the termination time falls behind that of the optimal hyperparameter. Therefore, Assumption 5.2 is satisfied in practical applications.

Theorem 5.3.

Under Assumptions 4.2,  5.1 and 5.2, given any prompt 
pt
∈
𝒳
∗
 and bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
, the expected stopping time regret of Algorithm 3 with 
𝙰𝙻𝙶
=
Algorithm 5 (EXP3Spec),

	
Reg
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
≤
2
𝐿
⋅
min
{
len
​
(
pt
𝜏
c
)
​
𝐾
​
log
⁡
𝐾
,
		
(15)

	
2
𝐿
𝐾
log
𝐾
+
min
𝑖
∈
[
𝐾
]
⁡
ST
​
(
𝙰𝙻𝙶
𝑖
)
​
𝐾
​
log
⁡
𝐾
}
.
		
(16)

Theorem 5.3 also provides an affirmative answer to the question posed in Section 3. The first term in the minimum provides a worst-case guarantee. Even if all hyperparameters in 
𝒮
 are not good or 
𝐾
 is large, EXP3Spec will stop at no more than 
𝑂
​
(
len
​
(
pt
𝜏
𝑐
)
)
 time steps after 
𝑆
𝑖
∗
 terminates. The second term is an instance-dependent bound in terms of hyperparameters 
𝒮
. Specifically, when the best hyperparameter 
𝑆
𝑖
∗
 has small stopping time, EXP3Spec will scale as 
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
+
𝑂
​
(
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
)
. This upper bound suggests that the number of speculative decoding rounds of EXP3Spec is almost the same as that of the best hyperparameter configuration 
𝙰𝙻𝙶
𝑖
∗
.

6Experiments

In this section, we conduct two sets of experiments to demonstrate the efficacy of the proposed bandit framework BanditSpec, along with UCBSpec and EXP3Spec. In the first experiment, the candidate hyperparameters are different draft models. In the second experiment, the candidate hyperparameters are different speculation lengths, where real-life LLM serving scenarios are simulated with diverse input prompts. Additional experimental results on memory utilization and additional experiments on larger models and different hardware are provided in Appendix G. The code is accessible via https://github.com/sail-sg/BanditSpec.

6.1Experiment with Draft Models

Experimental Setups We adopt the open-sourced LLaMA3-8B-Instruct (grattafiori2024llama3herdmodels) and Qwen2-7B-Instruct (yang2024qwen2technicalreport) as the target models. The commonly-used existing speculative decoding methods PLD (saxena2023prompt), Rest (he2024rest), Suffix Tree (oliaro2024suffixdecoding; hu2024sam) and Eagle-2 (li2024eagle2) are adopted as the baselines. Among these baselines, PLD, Rest, and Suffix Tree represent the non-parametric (or model-free) speculative decoding methods, whereas Eagle-2 represents the speculative decoding methods that utilize smaller draft models. Each of these methods corresponds to an arm in our problem.

The experiments are carried out on Spec Bench (xia2024unlocking), Alpaca (Taori2023alpaca), Code Editor (guo2024codeeditorbench) and Debug Bench (tian2024debugbench). Among these benchmarks, Spec Bench and Alpaca encompass multiple topics, while Code Editor and Debug Bench focus on coding tasks, a representative scenario for specialized models.

We record the number of accepted tokens for each speculative decoding step, as well as the wall-time for generating each complete response. The Mean Accepted Tokens (MAT) and the throughput (Tokens/s) is computed. These two metrics are widely adopted in the speculative decoding community and are positively correlated (xia2024unlocking). In particular, Tokens/s measures the actual latency during decoding. The experiments are conducted on a single A100 and set the batch size to 
1
.

Table 1:Empirical Comparison between the proposed algorithms and the existing works, measured by Mean Accepted Tokens (MAT) (
↑
) and Tokens/s (
↑
). The best result is highlighted in bold, while the second best result is underlined. The proposed algorithms demonstrate unequivocal superior performance compared with the existing methods.
Methods	Spec Bench	Alpaca	Code Editor	Debug Bench
MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)
LLaMA3-8B-Instruct			
Vanilla	1.00	35.73	1.00	35.92	1.00	36.32	1.00	36.89
PLD	1.46	43.96	1.53	53.06	2.13	82.61	1.67	82.76
Rest	1.29	40.67	1.48	52.40	1.33	51.32	1.29	48.49
Suffix Tree	1.83	55.10	1.71	64.02	2.30	90.21	2.13	77.56
Eagle-2	3.94	98.15	4.04	110.00	4.79	128.76	4.78	119.12
EXP3Spec	3.65	102.10	4.23	120.38	4.36	137.29	4.50	132.25
UCBSpec	3.98	105.72	4.35	125.78	4.83	138.27	4.60	135.34
Qwen2-7B-Instruct			
Vanilla	1.00	38.71	1.00	39.32	1.00	39.30	1.00	39.57
PLD	1.55	52.44	1.42	58.41	1.89	64.56	2.15	70.49
Rest	1.31	46.42	1.47	59.01	1.31	53.79	1.22	50.51
Suffix Tree	1.96	68.42	1.46	62.60	2.18	85.75	2.49	101.47
Eagle-2	3.64	97.82	3.61	104.43	4.88	138.58	4.79	126.01
EXP3Spec	3.76	107.36	3.83	113.90	4.90	160.41	4.86	151.73
UCBSpec	4.13	112.33	3.93	114.20	4.92	161.35	5.10	151.37

Experimental Results We report the results of our experiments in Table 1. The proposed adaptive speculative decoding framework BanditSpec exhibits superior performance compared to existing methods in the datasets we consider. In particular, the best performance measured by Token/s is always achieved by the proposed framework. We note that although the non-parametric methods are worse than Eagle-2 in average, they are effective on a portion of prompts. Our proposed methods, UCBSpec and EXP3Spec, automatically adapt to different prompts, i.e., suffering from a small stopping time regret on each prompt. Thus, they achieve better performance than all the methods that only use a fixed model. On Debug Bench, UCBSpec can even achieve improvements of 
13
%
 for LLaMA3 and 
19
%
 for Qwen2. Moreover, as UCBSpec demonstrates better performance under almost all benchmarks with both target models, this suggests that speculative decoding in real-life environments tends to be closer to the stochastic (Assumption 4.1) compared to the adversarial reward case (Assumption 5.1).

Remark 6.1.

The adversarial setting can be regarded as a means of comparison to the stationary setting. Prior to this work, it was a priori unclear how to use MAB to improve speculative decoding. Should one employ a stochastic, adversarial or even more generalized model? We consider a range of such MAB models and do a comparison among them to provide the community with a guide on which MAB model is best suited to the speculative decoding problem. As the empirical performance of UCBSpec is better than EXP3Spec (Table 1), it implies that real-life scenario tends to be benign and may be more aligned with the stationary mean assumption.

6.2Experiment with Speculation Lengths
(a)Target model: LLaMA3
(b)Target model: Qwen2
Figure 3:We compare throughtput improvements with different speculative decoding lengths 
𝛾
∈
[
4
]
 and the canonical decoding (
𝛾
=
0
). The performance of UCBSpec approaches that of the best hyperparameter across all samples for both target models LLaMA3 and Qwen2. The sample indices are sorted according to the best arm improvement for a clear demonstration.

Experimental Setups In addition to improving the latency when batch size is 
1
, our proposed algorithms also improve the throughput in real LLM serving scenarios with different batch sizes. In practical serving environments, speculative decoding does not always yield performance gains due to variations in batch size and acceptance rate. As the batch size increases, the system rapidly becomes compute-bound, while a lower acceptance rate can lead to wasted computation resources of the GPU. Additionally, the execution time of the draft model contributes to an overall decrease in throughput. Given these confoundingly interrelated factors, along with latent variables such as the acceptance rate (which is unknown before verification and depends on the input prompts), we adopt a bandit-based approach to model the current throughput as the reward. Specifically, we employ UCBSpec to dynamically adjust the hyperparameter 
𝛾
, the speculation length, to maximize the throughput, i.e., the number of generated tokens per second. We set the maximum speculation length 
𝐿
 as 
4
, and 
𝛾
 takes values in 
{
0
,
…
,
4
}
 where 
𝛾
=
0
 corresponds to the canonical decoding (Algorithm 1). As the first experiment suggests UCBSpec is more in line with the real-life speculative decoding environment than EXP3Spec, we only evaluate UCBSpec in this experiment. The experiments are conducted on a single A100.

Specifically, we use LLaMA3-8B-Instruct and Qwen2-7B-Instruct as the target models and adopt Eagle-1 (li2024eagle), the current state-of-the-art model, as the draft model. We do not use Eagle-2 (li2024eagle2) because it does not support batch inference. For evaluation, we adopt Alpaca (Taori2023alpaca) as the test set, as it covers various topics, thereby simulating a realistic setting with diverse acceptance rates. To approximate real-world conditions, we randomly sample prompts from the test set to form a batch for inference, with batch sizes ranging from 
1
 to 
50
. As our evaluation metric, we measure the throughput improvement relative to the canonical decoding (non-speculative) baseline. Our result is averaged over 
16
 independent runs to smooth the hardware-dependent factors.

Experimental Results The results are presented in Figure 3, where we reorder the 
500
 sample indices in ascending order of the performance of the best hyperparameter (blue line) for easy comparison. Otherwise, the lines in this figure will not be largely monotonic. Here the “worst” and “best” lines are calculated among results of 
𝛾
∈
{
1
,
⋯
,
4
}
 in hindsight. Thus, we call the “best” line as the oracle best. Firstly, since the optimal hyperparameter 
𝛾
 varies with different input prompts for either target model, fixing a single hyperparameter is suboptimal, e.g., in Figure 3 (b), the best hyperparameter changes from 
𝛾
=
1
 (light green) to 
𝛾
=
2
 (green) at a sample index around 
80
; and the original Eagle-1 (li2024eagle) (
𝛾
=
4
 in purple) is even inferior to the canonical decoding (
𝛾
=
0
 in grey) for sample indices less than 
80
. This necessitates the use of adaptive hyperparameter selection. Next, UCBSpec demonstrates competitive throughput performance, outperforming the second-best hyperparameter in most cases and closely approaching the (varying) oracle best across experiments. These benefits are obtained thanks to the adaptivity of BanditSpec.

7Conclusions and Discussions

In this work, we propose a MAB framework together with two hyperparameter selection algorithms that adaptively choose appropriate hyperparameters to accelerate LLM inference under realistic assumptions. Both theoretical guarantees and extensive experiments are provided to demonstrate that adaptive speculative decoding via bandit algorithms can boost the performance of existing methods in a training-free manner. For future work, we would like to point out some directions, improving the performance of the current algorithms.

Therefore, another direction is to design hyperparameter selection algorithms that can achieve the (near) optimal balance between these two goals based on practical needs.

Structured Bandits Our current framework is based on the standard 
𝐾
-armed bandit model. However, broader classes of bandit problems with additional structures—such as linear bandits (Abbasi2011improved) and Lipschitz bandits (magureanu2014lipschitz)—can also be considered. This aligns more closely with practical scenarios, where the number of hyperparameters can be large, and the value of 
𝐾
 may be very high when modeling the problem as a 
𝐾
-armed MAB. By leveraging such structures in MABs, we can expect to identify better hyperparameters more efficiently, thereby further accelerating the optimization process.

Robust bandits and Non-stationary bandits As indicated by the experimental result, the real-life speculative decoding environment is closer to the stochastic reward case (Assumption 4.1) than the adversarial reward case (Assumption 5.1). Therefore, one direction for future work is to consider the settings “in between”, e.g., robust bandits in the presence of adversarial corruptions (Ding2022Robust; zhong2021probabilistic), or non-stationary bandits (cao2019nearly; Besbes2014Stochastic; hou2024almost) where the mean number of accepted tokens can vary across time. These settings are more benign than the adversarial reward assumption and can be exploited to accelerate the inference.

Contextual Bandits Another direction is to explore contextual bandits, where the environment reveals additional information that can be leveraged to reduce the learning burden (luo2018contextual; kato2021role).

Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Acknowledgements:

This work is supported by funding from the Singapore Ministry of Education Academic Research Fund (AcRF) Tier 1 grants under grant numbers A-8002934-00-00 and A-8000980-00-00. This research is also supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-PhD-2023-08-044T-J), and is part of the programme DesCartes which is supported by the National Research Foundation, Prime Minister’s Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme.

Appendix ARelated Works

Speculative Decoding Speculative decoding is proposed in leviathan2023fast; chen2023accelerating, where the draft model only generates a single chain of draft tokens. Then a line of works extends the chain structure to the tree structure (miao2023SpecInferAG; cai2024medusa; du2024glide; li2024eagle; huaccelerated). In these works, the draft tokens are organized as a connected tree. To further improve the number of accepted tokens, previous works propose to generate tokens in a batch manner, i.e., the draft tokens are organized as multiple disconnected parts. SpecTr (sun2024spectr) views this problem from the optimal transport perspective and derives the algorithm that is optimal up to a multiplicative constant. khisti2024multi derives the canonical form of this problem and designs the relaxed optimization algorithms. All these algorithms verify the draft tokens in a token-by-token manner. sun2024block proposes to verify all the draft tokens as a whole block, which further boosts the acceleration ratio. sun2024triforce proposes to fit the speculative deciding into a hierarchical structure, where multiple draft models with various sizes are generating and verifying tokens. The smaller model will generate more tokens. This fine-grained behavior improves the overall performance of the system. liu2023OnlineSD design algorithms to update the draft model parameters in an online manner, which makes the draft model adaptive to the current context. liu2024optimizing and huang2024specdec++ aim to optimize the speculative length in a training and training-free manner (more discussions on SpecDec++ (huang2024specdec++) are provided in Appendix B.4). chen2024SequoiaSR optimizes the hyperparameters related to the hardware by dynamic programming in an offline manner. We also note that there is a series of non-parametric speculative decoding algorithms (hu2024sam; oliaro2024suffixdecoding), i.e., the draft model itself does not require any training procedures. yin2024theoretical derives the theoretical analysis of the speculative decoding.

Multi-Armed Bandit The multi-armed bandit problem is a fundamental topic in decision theory and reinforcement learning, with various algorithms developed to address the exploration-exploitation trade-off. The standard stochastic 
𝐾
-armed bandit problem was first introduced by robbins1952some and then studied by lai1985asymptotically. There has been a major theoretical advancement with the introduction of Upper Confidence Bound (UCB) algorithms (auer2002finite). Various algorithms have been proposed to achieve improved theoretical guarantees and practical performance since then (garivier2011KLUCB; bubeck2012regret). Beyond UCB-type algorithms, sampling-based algorithms, such as Thompson Sampling (agrawal2012analysis; agrawal2017near; russo2017ATO) and sampling via bootstrap (kveton2019garbage; wan2023multiplier), have also exhibited strong empirical performance with provable regret bounds. Furthermore, the problem has been extended to the adversarial settings where the rewards are no longer stochastic (auer2002nonstochastic; bubeck2012regret). We refer to lattimore2020bandit for a comprehensive introduction to the Multi-Armed Bandit problem.

Appendix BAdditional Discussions and Remarks
B.1Discussion on the Assumptions

On the theoretical side, the stationary mean assumption (Assumption 4.1) is strictly weaker than the i.i.d. assumption. In particular, the number of accepted tokens can depend on the generated tokens. Therefore, this assumption is aligned with real-world scenarios in which different decoding steps are correlated. Furthermore, the basic Multi-Armed Bandits (MAB) model can be generalized to contextual bandits and non-stationary bandits. The proposed BanditSpec framework provides a basic template to apply these more general MAB setups to speculative decoding. Our formulations under the stationary/adversarial mean assumptions are just basic setups and we leave the more general/elaborate setups as future research.

On the experimental side, our experimental results (Table 1) indicate, the performance of UCBSpec significantly outperforms one of the best speculative decoding methods, Eagle-2 (Li et al., 2024). This corroborates the stationary mean assumption in our formulation.

B.2Discussion on UCBSpec

We comment that UCBSpec is among the simplest UCB-type algorithms in the sense that only the empirical means and UCB’s need to be maintained, and the hyperparameter to be selected can be directly determined via the UCB’s.

In contrast, Thompson Sampling (agrawal2012analysis) and KL-UCB (garivier2011KLUCB) generally achieve better empirical regret bounds than UCB1 (auer2002finite). However, Thompson Sampling requires maintaining the posterior distribution and sampling from it to select the arm to pull; whereas KL-UCB involves solving an optimization problem for arm selection. These steps add additional complexity to the algorithms.

Given our goal of accelerating LLM inference, the simplicity of UCB1 is more in line with this objective. Therefore, we propose UCBSpec, which redesigns the confidence radius and the stopping rule of UCB1 to adapt specifically to the speculative decoding application.

B.3Discussion on Adaptive Adversary

We consider the oblivious adversary in this paper where the number of accepted tokens generated by all hyperparameters at all time steps, i.e., 
{
𝑦
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
,
𝑡
∈
ℕ
, is fixed before the decoding process starts. One may be interested in considering the adapative adversary, where the environment (adversary) can choose the number of accepted tokens generated by 
𝑆
𝐼
𝑡
 based on (part of) the history information 
ℋ
𝑡
−
1
 and 
𝐼
𝑡
 (arora2012online). This adversary is more malicious than the oblivious one and the regret is expected to be even larger than the current one in Theorem 5.3. As our empirical experiments suggest that the practical scenario aligns more closely with the stochastic reward assumption (Assumption 4.1) and deviates from the oblivious adversarial reward assumption (Assumption 5.1), we believe it is unnecessary to consider the adaptive adversarial reward.

B.4Discussion on SpecDec++ (huang2024specdec++)

We compare the proposed methods with an existing adaptive speculative decoding method, SpecDec++ (huang2024specdec++), in this section.

SpecDec++ (huang2024specdec++) is adaptive in choosing the speculation length, achieving good performance compared to the vanilla speculative decoding method (leviathan2023fast; chen2023accelerating). It trains an acceptance probability prediction head and stops drafting new tokens when the predicted rejection probability reaches a certain threshold.

We compare it with the proposed methods as follows:

• 

Firstly, we highlight that our proposed method is training-free which can be deployed easily along with existing off-the-shelf methods. In contrast, SpecDec++ focuses on training of an acceptance prediction head. Currently, SpecDec++ is only available when using LLaMA-2-Chat-7B as the draft model and LLaMA-2-Chat-70B as the target model (bfloat 16).

• 

Secondly, the proposed BanditSpec framework considers the more general hyperparameter selection problem that goes beyond merely the speculation length. Therefore, it is ”orthogonal” to SpecDec++ in the sense that any methods with (or without) SpecDec++ can also be candidates for the hyperparameter in our framework, e.g., {Eagle-2, LLaMA-2-Chat-7B} with SpecDec++ can also be regarded as arms (if they are available).

Appendix CAdditional Details

In this section, we provide more details that complement the main paper.

C.1Vanilla Speculative Decoding

For completeness, we present and describe the vanilla speculative decoding algorithm (leviathan2023fast; chen2023accelerating) in Algorithm 6 in this section.

We introduce some notations first. For any nonnegative function 
𝑓
:
𝒳
→
ℝ
+
 with 
∑
𝑥
∈
𝒳
𝑓
​
(
𝑥
)
>
0
, we define the distribution induced by it as 
Norm
​
(
𝑓
​
(
⋅
)
)
=
𝑓
​
(
⋅
)
/
∑
𝑥
′
∈
𝒳
𝑓
​
(
𝑥
′
)
. The positive part of a function 
𝑓
 is denoted as 
[
𝑓
​
(
⋅
)
]
+
=
max
⁡
{
0
,
𝑓
​
(
⋅
)
}
.

Speculative decoding implements a draft model 
𝑄
 to generate draft tokens and let the target model 
𝑃
 verify them in parallel. In practice, the draft model is much smaller than the target model. Thus, the draft token generation (Line 1) can be achieved in a short time. Then we let the target model only forward inference once with these draft tokens as inputs (Line 2). The verification procedures (Lines 4 to 9) are designed to guarantee that the output tokens 
𝑥
1
:
𝜏
+
1
 are distributed as the target model 
𝑃
. Here, the additional 
(
𝜏
+
1
)
-st accepted token is also called the bonus token.

Algorithm 6 Vanilla Speculative Decoding

Inputs: base model 
𝑃
, draft model 
𝑄
, prefix 
pt
, maximum speculation length 
𝐿

Procedures:

1: Generate 
𝐿
 draft tokens 
𝑥
~
1
:
𝐿
 via 
𝑥
~
𝑖
∼
𝑄
(
⋅
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
 for 
𝑖
∈
[
𝐿
]
.
2: Set 
𝜏
=
0
, and calculate the values of 
𝑃
​
(
𝑥
~
𝑖
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
 for 
𝑖
∈
[
𝐿
]
 in parallel.
3: for 
𝑘
=
1
,
…
,
𝐿
 do
4:  Sample 
𝑟
𝑖
∼
Unif
​
(
[
0
,
1
]
)
.
5:  if 
𝑟
𝑖
≤
min
⁡
{
1
,
𝑃
​
(
𝑥
~
𝑖
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
/
𝑄
​
(
𝑥
~
𝑖
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
}
 then
6:   Set 
𝜏
=
𝑖
 and 
𝑥
𝑖
=
𝑥
~
𝑖
.
7:  else
8:   Sample 
𝑥
𝑖
∼
Norm
(
[
𝑃
(
⋅
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
−
𝑄
(
⋅
|
pt
,
𝑥
~
1
:
𝑖
−
1
)
]
+
)
.
9:   break.
10:  end if
11: end for
12: if 
𝜏
=
𝐿
 then sample 
𝑥
𝐿
+
1
∼
𝑃
(
⋅
|
pt
,
𝑥
1
:
𝐿
)
.
13: return 
𝑥
1
:
𝜏
+
1
.
C.2Dynamics of MAB

We provide a description of the dynamics of MAB in Algorithm 7.

Algorithm 7 Dynamics of MAB

Inputs: 
𝐾
 arms, time horizon 
𝑇
.

1: 
ℋ
0
=
∅
.
2: for 
𝑡
=
1
,
2
,
…
,
𝑇
 do
3:  Agent adopts an algorithm to select 
𝐼
𝑡
 based on 
ℋ
𝑡
−
1
.
4:  Environment reveals the reward 
𝑋
𝐼
𝑡
,
𝑡
 to the agent.
5:  
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝐼
𝑡
,
𝑋
𝐼
𝑡
,
𝑡
)
)
.
6: end for

The goal is to minimize the cumulative regret

	
max
𝑖
∈
[
𝐾
]
⁡
𝔼
​
[
∑
𝑡
=
1
𝑇
𝑋
𝑖
,
𝑡
]
−
𝔼
​
[
∑
𝑡
=
1
𝑇
𝑋
𝐼
𝑡
,
𝑡
]
,
		
(17)

where the expectation is taken w.r.t. the randomness in the rewards (for the stochastic setup) and the possible internal randomness in the arm selection algorithm.

C.3Full Description of BanditSpec with UCBSpec

We provide the full description of 
BanditSpec
​
(
𝙰𝙻𝙶
)
 with 
𝙰𝙻𝙶
=
UCBSpec
 in Algorithm 8.

Algorithm 8 
BanditSpec
​
(
UCBSpec
)
 (Full version of UCBSpec)

Inputs: initial prompt 
pt
0
=
pt
∈
𝒳
∗
, bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
.
Procedures:

1: 
𝑡
=
0
,
ℋ
0
=
∅
,
𝐼
0
=
1
,
𝑥
𝐼
0
,
0
=
∅
.
2: while 
EOS
∉
𝑥
𝐼
𝑡
,
𝑡
 do
3:  
𝑡
=
𝑡
+
1
4:  if 
𝑡
≤
𝐾
 then
5:   
𝐼
𝑡
=
𝑡
. (Round-Robin)
6:  else
7:   Select index 
𝐼
𝑡
=
argmax
𝑖
∈
[
𝐾
]
𝑈
​
𝐶
​
𝐵
𝑖
,
𝑡
−
1
.
8:  end if
9:  Observe 
𝑋
𝐼
𝑡
,
𝑡
=
𝑥
𝐼
𝑡
,
𝑡
=
SpecDecSub
​
(
pt
𝑡
−
1
,
𝑃
,
𝑆
𝐼
𝑡
,
𝐿
)
and
𝑌
𝐼
𝑡
,
𝑡
=
𝑦
𝐼
𝑡
,
𝑡
=
len
​
(
𝑋
𝐼
𝑡
,
𝑡
)
.
10:  
pt
𝑡
=
concat
​
(
pt
𝑡
−
1
,
𝑋
𝐼
𝑡
,
𝑡
)
, 
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝐼
𝑡
,
𝑋
𝐼
𝑡
,
𝑡
)
)
.
11:  Update the statistics 
{
𝜇
^
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
,
{
cr
𝑖
,
𝑡
}
𝑖
∈
[
𝐾
]
, where
	
𝑛
𝑖
,
𝑡
	
=
∑
𝑠
=
1
𝑡
𝟙
​
{
𝐼
𝑠
=
𝑖
}
,
𝜇
^
𝑖
,
𝑡
=
∑
𝑠
=
1
𝑡
𝑌
𝑖
,
𝑠
​
𝟙
​
{
𝐼
𝑠
=
𝑖
}
𝑛
𝑖
,
𝑡
,
		
(18)

	
cr
𝑖
,
𝑡
	
=
𝐿
2
​
1
+
𝑛
𝑖
,
𝑡
𝑛
𝑖
,
𝑡
2
​
(
1
+
2
​
log
⁡
𝐾
​
𝑡
2
​
(
1
+
𝑛
𝑖
,
𝑡
)
1
2
𝛿
)
,
		
(19)

	
UCB
𝑖
,
𝑡
	
=
𝜇
^
𝑖
,
𝑡
+
cr
𝑖
,
𝑡
.
		
(20)
12: end while
13: return 
𝑡
,
pt
𝑡
C.4Full Description of BanditSpec with EXP3Spec

We provide the full description of 
BanditSpec
​
(
𝙰𝙻𝙶
)
 with 
𝙰𝙻𝙶
=
EXP3Spec
 in Algorithm 9.

Algorithm 9 
BanditSpec
​
(
EXP3Spec
)
 (Full version of EXP3Spec)

Inputs: initial prompt 
pt
0
=
pt
∈
𝒳
∗
, bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
, learning rates 
𝜂
𝑡
=
log
⁡
𝐾
𝑡
⋅
𝐾
,
𝑡
∈
ℕ
.
Procedures:

1: 
𝑡
=
0
,
ℋ
0
=
∅
,
𝐼
0
=
1
,
𝑥
𝐼
0
,
0
=
∅
.
2: while 
EOS
∉
𝑥
𝐼
𝑡
,
𝑡
 do
3:  
𝑡
=
𝑡
+
1
4:  Set probability vector 
𝑝
𝑡
∈
𝚫
[
𝐾
]
 with
	
𝑝
𝑡
,
𝑖
=
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑖
,
𝑠
)
∑
𝑗
=
1
𝐾
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑗
,
𝑠
)
,
∀
𝑖
∈
[
𝐾
]
.
		
(21)
5:  Select a hyperparameter index 
𝐼
𝑡
∼
𝑝
𝑡
.
6:  Observe 
𝑋
𝐼
𝑡
,
𝑡
=
𝑥
𝐼
𝑡
,
𝑡
=
SpecDecSub
​
(
pt
𝑡
−
1
,
𝑃
,
𝑆
𝐼
𝑡
,
𝐿
)
 and 
𝑦
𝐼
𝑡
,
𝑡
=
len
​
(
𝑋
𝐼
𝑡
,
𝑡
)
.
7:  
pt
𝑡
=
concat
​
(
pt
𝑡
−
1
,
𝑋
𝐼
𝑡
,
𝑡
)
,
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝐼
𝑡
,
𝑋
𝐼
𝑡
,
𝑡
)
)
.
8:  Set the statistics
	
𝑍
^
𝑖
,
𝑡
=
𝟙
​
{
𝑖
=
𝐼
𝑡
}
⋅
𝐿
+
1
−
𝑦
𝑖
,
𝑡
𝐿
⋅
𝑝
𝑡
,
𝑖
,
∀
𝑖
∈
[
𝐾
]
.
		
(22)
9: end while
10: return 
𝑡
,
pt
𝑡
Appendix DProofs of Main Results
D.1Proof of Proposition 3.1

To prove Proposition 3.1, we note that we only need to prove

	
pt
ST
​
(
𝙰𝙻𝙶
)
​
=
𝑑
​
pt
𝜏
c
,
 and 
​
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
𝐿
+
1
≤
ST
​
(
𝙰𝙻𝙶
)
≤
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
,
𝑎
.
𝑠
.
	

The other results are implied by these two results. For equality, we note that this is already proved by Theorem 1 in yin2024theoretical, where the equality holds for any specification of the hyperparameters. For the inequality, we note that each implementation of SpecDecSub generates at least one token and at most 
𝐿
+
1
 tokens. Thus, the inequality holds almost surely.

D.2Proof of Theorem 4.3

See 4.3

Proof of Theorem 4.3.

Our proof of Theorem 4.3 consists of three steps.

• 

Reward and Stopping time decomposition.

• 

Construction of the high probability event.

• 

Concluding the proof.

As we mentioned in Section 4, the main difference lies at the two aspects:
Firstly, the stopping time is now a random variable, which depends on the generated tokens. This causes trouble when we decompose the reward/regret, as both the rewards andthe time horizon depend on the history. We tackle this problem in Step 1 by making use of the martingale structure of the rewards sequence.
Secondly, we consider the problem under Assumption 4.1, where the number of accepted tokens can be dependent. This is practical as LLM generates tokens in an autoregressive manner. In contrast, under the commonly seen assumption for the 
𝐾
-armed MAB, the rewards are i.i.d. and can be regarded as if they had been sampled before the algorithm starts (see Chapter 4 in lattimore2020bandit). Thus, Chernoff-Hoeffding bound (Lemma F.2) can be directly applied, which cannot be used under Assumption 4.1. We solve this problem in Step 2, by adopting the so-called self-normalized confidence bounds (Abbasi2011improved).

Step 1: Reward and Stopping time decomposition.

By the property of speculative decoding in (3) and (30), for any algorithm 
𝙰𝙻𝙶
, 3

	
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
=
𝔼
​
[
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
]
		
(23)

We wish to decompose the expected total token sequence in terms of each hyperparameter 
𝑖
∈
[
𝐾
]
 in the first step, i.e.,

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
]
=
∑
𝑖
=
1
𝐾
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
.
		
(24)

The standard regret analysis adopts Wald’s equation to decompose the expected cumulative regret, or equivalently, the stopping time. However, as both the stopping time 
ST
​
(
𝙰𝙻𝙶
)
 and 
𝑌
𝐼
𝑡
,
𝑡
 depend on the history under our problem setup, Wald’s equation fails. We propose a new and general approach to decomposing the reward.

∙
 Step 1.1: We first prove that 
𝑀
𝑛
:=
∑
𝑡
=
1
𝑛
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
,
𝑛
=
0
,
1
,
2
,
…
 is a martingale with respect to 
{
ℱ
𝑛
}
𝑛
=
0
∞
, where 
𝑀
0
:=
0
,
ℱ
𝑛
:=
𝜎
​
(
ℋ
𝑛
)
.

By the definition of martingale, we only need to show (1) 
𝔼
​
[
|
𝑀
𝑛
|
]
<
∞
, and (2) 
𝔼
​
[
𝑀
𝑛
+
1
|
ℱ
𝑛
]
=
𝑀
𝑛
.

(1) 
𝔼
​
[
|
𝑀
𝑛
|
]
<
∞
: As the number of the accepted tokens at each round is bounded as 
𝑌
𝐼
𝑡
,
𝑡
∈
[
1
,
𝐿
+
1
]
 almost surely and 
𝜇
𝐼
𝑡
∈
[
1
,
𝐿
+
1
]
, we have 
|
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
|
≤
𝐿
. Then the triangular inequality 
|
𝑀
𝑛
|
≤
∑
𝑡
=
1
𝑛
|
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
|
≤
𝐿
⋅
𝑛
<
∞
 indicates that

	
𝔼
​
[
|
𝑀
𝑛
|
]
<
∞
.
		
(25)

(2) 
𝔼
​
[
𝑀
𝑛
+
1
|
ℱ
𝑛
]
=
𝑀
𝑛
: The conditional expectation of 
𝑀
𝑛
+
1
 can be calculated via tower property as

	
𝔼
​
[
𝑀
𝑛
+
1
∣
ℱ
𝑛
]
=
𝑀
𝑛
+
𝔼
​
[
𝔼
​
[
𝑌
𝐼
𝑛
+
1
,
𝑛
+
1
−
𝜇
𝐼
𝑛
+
1
∣
ℋ
𝑛
,
𝐼
𝑛
+
1
]
∣
ℋ
𝑛
]
=
𝑀
𝑛
,
		
(26)

where the last equality results from Assumption 4.1.

Based on (25) and (26), 
𝑀
𝑛
,
𝑛
=
0
,
1
,
2
,
…
 is a martingale with respect to 
{
ℱ
𝑛
}
𝑛
=
0
∞
.

∙
 Step 1.2: We then prove 
𝔼
​
[
𝑀
ST
​
(
𝙰𝙻𝙶
)
]
=
0
 via Doob’s optional stopping lemma (Lemma F.1).

We have already showed that 
∑
𝑡
=
1
𝑛
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
,
𝑛
=
1
,
2
,
…
 is a martingale with respect to 
{
ℋ
𝑛
}
𝑛
=
0
∞
. In order to apply Lemma F.1, we firstly verify the prerequisites listed in Lemma F.1 condition (b): (1) 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
, and (2) there exists 
𝑐
∈
ℝ
, such that 
𝔼
​
[
|
𝑀
𝑡
−
𝑀
𝑡
−
1
|
∣
ℱ
𝑡
−
1
]
≤
𝑐
 almost surely for 
𝑡
≤
ST
​
(
𝙰𝙻𝙶
)
.

(1) 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
: According to the property of speculative decoding (4) and Assumption 4.2, we have that 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
≤
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
<
∞
.

(2) 
𝔼
​
[
|
𝑀
𝑡
−
𝑀
𝑡
−
1
|
∣
ℱ
𝑡
−
1
]
≤
𝑐
: As 
𝑀
𝑡
−
𝑀
𝑡
−
1
=
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
 and 
|
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
|
≤
𝐿
 almost surely, it holds that 
𝔼
​
[
|
𝑀
𝑡
−
𝑀
𝑡
−
1
|
∣
ℱ
𝑡
−
1
]
≤
𝐿
. Taking 
𝑐
=
𝐿
 finishes the verification.

Therefore, condition (b) in Lemma F.1 is satisfied and we obtain

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
]
=
0
.
		
(27)

Step 1.3: We show that 
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
]
=
∑
𝑖
=
1
𝐾
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
.

We firstly note that

	
𝔼
​
[
|
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
|
]
≤
(
𝐿
+
1
)
​
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
 and 
𝔼
​
[
|
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜇
𝐼
𝑡
|
]
≤
(
𝐿
+
1
)
​
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
,
		
(28)

so the expectations of 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
 and 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜇
𝐼
𝑡
 exist and are finite (integrable).

Furthermore, we have

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜇
𝐼
𝑡
]
=
𝔼
​
[
∑
𝑖
=
1
𝐾
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
𝐼
𝑡
=
𝑖
}
⋅
𝜇
𝑖
]
=
∑
𝑖
=
1
𝐾
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
,
		
(29)

where 
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
𝐼
𝑡
=
𝑖
}
 by definition. Because 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
 and 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜇
𝐼
𝑡
 are integrable, (27) and (29) imply

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
]
+
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜇
𝐼
𝑡
]
=
∑
𝑖
=
1
𝐾
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
.
		
(30)

This equality decomposes the cumulative reward (and stopping time) in terms of the arms.

Step 2: Construction of the high probability event.

We then derive the concentration property for the number of accepted tokens. Define the good events:

	
ℰ
𝑡
:=
{
𝜇
^
𝑖
,
𝑡
∈
[
𝜇
𝑖
−
cr
𝑖
,
𝑡
,
𝜇
𝑖
+
cr
𝑖
,
𝑡
]
,
∀
𝑖
∈
[
𝐾
]
​
 at round 
​
𝑡
}
.
		
(31)

Since random variables supported on 
[
𝑎
,
𝑏
]
 is 
(
𝑏
−
𝑎
)
2
/
4
-sub-Gaussian and 
1
−
𝜇
𝐼
𝑡
≤
𝑌
𝐼
𝑡
,
𝑡
−
𝜇
𝐼
𝑡
≤
𝐿
+
1
−
𝜇
𝐼
𝑡
 under our problem setup. According to Lemma F.3, we obtain

	
ℙ
​
(
ℰ
𝑡
)
≥
1
−
𝛿
𝑡
2
 and 
∑
𝑡
=
𝐾
+
1
∞
ℙ
​
(
ℰ
𝑡
𝑐
)
≤
𝜋
2
​
𝛿
6
,
		
(32)

where 
𝛿
 is a confidence parameter that will be specified later. We remark that Lemma F.3 from Abbasi2011improved adopts a self-normalized concentration bound for the martingale sequence, which generalizes the standard i.i.d. reward assumption in the 
𝐾
-armed MAB problem.

As a result, we can now bound the number of times arm 
𝑖
 is pulled at any round 
𝑡
≥
𝐾
. Conditional on the good event 
ℰ
𝑡
, we have 
𝜇
^
𝑖
,
𝑡
∈
[
𝜇
𝑖
−
cr
𝑖
,
𝑡
,
𝜇
𝑖
+
cr
𝑖
,
𝑡
]
 and arm 
𝑖
 will not be pulled if 
cr
𝑖
,
𝑡
<
Δ
𝑖
/
2
. By adopting Lemma F.4, when arm 
𝑖
 is selected at time 
𝑡
+
1
, it must hold that

	
𝑛
𝑖
,
𝑡
≤
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
​
𝐾
⋅
𝑡
2
Δ
𝑖
​
𝛿
)
.
		
(33)

Step 3: Concluding the proof.

According to Step 1,

	
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
=
𝔼
​
[
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑌
𝐼
𝑡
,
𝑡
]
=
∑
𝑖
=
1
𝐾
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
.
		
(34)

Therefore,

	
Reg
​
(
𝙰𝙻𝙶
)
	
=
1
𝜇
𝑖
∗
​
(
∑
𝑖
∈
[
𝐾
]
𝜇
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
+
∑
𝑖
∈
[
𝐾
]
Δ
𝑖
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
−
𝜇
𝑖
∗
⋅
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
]
)
		
(35)

		
=
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
.
		
(36)

Under the UCBSpec algorithm, we have

	
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
		
(37)

	
=
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
​
[
∑
𝑡
=
𝐾
+
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
𝐼
𝑡
=
𝑖
,
𝑛
𝑖
,
𝑡
−
1
≤
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
​
𝐾
​
𝑡
2
Δ
𝑖
​
𝛿
)
}
]
		
(38)

	
+
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
​
[
∑
𝑡
=
𝐾
+
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
𝐼
𝑡
=
𝑖
,
𝑛
𝑖
,
𝑡
−
1
>
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
​
𝐾
​
(
𝑡
−
1
)
2
Δ
𝑖
​
𝛿
)
}
]
+
𝐾
		
(39)

	
≤
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
​
[
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
​
𝐾
⋅
(
ST
​
(
𝙰𝙻𝙶
)
)
2
Δ
𝑖
​
𝛿
)
]
+
𝔼
​
[
∑
𝑡
=
𝐾
+
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
ℰ
𝑡
𝑐
}
]
+
𝐾
		
(40)

	
≤
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
(
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
​
𝐾
⋅
(
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
)
2
Δ
𝑖
​
𝛿
)
)
+
∑
𝑡
=
𝐾
+
1
∞
ℙ
​
(
ℰ
𝑡
𝑐
)
+
𝐾
,
		
(41)

	
≤
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
(
4
+
2
​
𝐿
2
Δ
𝑖
2
​
(
1
+
2
​
log
⁡
𝐿
𝐾
⋅
(
𝔼
[
len
(
pt
𝜏
c
]
)
2
Δ
𝑖
​
𝛿
)
)
+
𝜋
2
​
𝛿
6
+
𝐾
,
		
(42)

where the first inequality results from (33), the second inequality utilizes Jensen’s inequality, and the last inequality adopts the property of speculative decoding (4) and the upper bound on the error probability of good event (32) in Step 2. Taking 
𝛿
=
1
/
2
 in the above bound concludes the proof of this theorem. ∎

D.3Proof of Theorem 5.3

Fix any 
𝑏
∈
[
𝐾
]
, the baseline algorithm is set to be 
𝙰𝙻𝙶
𝑏
, i.e., Algorithm 3 implements Line 4 of Algorithm 3 with hyperparameter 
𝑆
𝑏
 only. Let 
𝙰𝙻𝙶
=
EXP3Spec
. we assume the 
BanditSpec
​
(
𝙰𝙻𝙶
)
 repeats the while loop in Algorithm 3 until 
max
⁡
{
ST
​
(
𝙰𝙻𝙶
)
,
ST
​
(
𝙰𝙻𝙶
𝑏
)
}
. To avoid any confusion, we restate the algorithm for the purpose of analysis in Algorithm 10. Algorithm 10 takes 
𝙰𝙻𝙶
 and 
𝙰𝙻𝙶
𝑏
 as an input and stops until 
pt
ST
​
(
𝙰𝙻𝙶
)
 and 
pt
ST
​
(
𝙰𝙻𝙶
𝑏
)
 are generated. The two token sequences up to 
EOS
 token are output at the end of the algorithm.

Algorithm 10 
BanditSpec
​
(
EXP3Spec
)
 (For analysis purpose)

Inputs: initial prompt 
pt
0
=
pt
∈
𝒳
∗
, speculative decoding configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
, stopping time 
𝜏
=
∞
, baseline hyperparameter 
𝑆
𝑏
, initial prompt 
pt
0
𝑏
=
pt
, stopping time 
𝜏
𝑏
=
∞
.
Procedures:

1: 
𝑡
=
0
,
ℋ
0
=
∅
,
𝐼
0
=
1
,
𝑥
𝐼
0
,
0
=
∅
,
𝑥
𝑏
,
0
𝑏
=
∅
.
2: while 
𝜏
=
∞
 or 
𝜏
𝑏
=
∞
 do
3:  
𝑡
=
𝑡
+
1
4:  
/
⁣
/
 Procedures of the original EXP3Spec
5:  if 
𝜏
=
∞
 and 
EOS
∈
𝑥
𝐼
𝑡
−
1
,
𝑡
−
1
 then
6:   
𝜏
=
𝑡
−
1
 and 
pt
𝜏
=
pt
𝑡
−
1
.
7:  end if
8:  Set probability vector 
𝑝
𝑡
∈
𝚫
[
𝐾
]
 with
	
𝑝
𝑡
,
𝑖
=
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑖
,
𝑠
)
∑
𝑗
=
1
𝐾
exp
⁡
(
−
𝜂
𝑡
​
∑
𝑠
=
1
𝑡
−
1
𝑍
^
𝑗
,
𝑠
)
​
with learning rate
​
𝜂
𝑡
=
log
⁡
𝐾
𝑡
⋅
𝐾
,
∀
𝑖
∈
[
𝐾
]
.
		
(43)
9:  Select a hyperparameter index 
𝐼
𝑡
∼
𝑝
𝑡
.
10:  Observe 
𝑋
𝐼
𝑡
,
𝑡
=
𝑥
𝐼
𝑡
,
𝑡
=
SpecDecSub
​
(
pt
𝑡
−
1
,
𝑃
,
𝑆
𝐼
𝑡
,
𝐿
)
 and 
𝑦
𝐼
𝑡
,
𝑡
=
len
​
(
𝑋
𝐼
𝑡
,
𝑡
)
.
11:  
pt
𝑡
=
concat
​
(
pt
𝑡
−
1
,
𝑋
𝐼
𝑡
,
𝑡
)
,
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝐼
𝑡
,
𝑋
𝐼
𝑡
,
𝑡
)
)
.
12:  Set the statistics
	
𝑍
^
𝑖
,
𝑡
=
𝟙
​
{
𝑖
=
𝐼
𝑡
}
⋅
𝐿
+
1
−
𝑦
𝑖
,
𝑡
𝐿
⋅
𝑝
𝑡
,
𝑖
,
∀
𝑖
∈
[
𝐾
]
.
		
(44)
13:  
/
⁣
/
 Procedures of the baseline 
𝙰𝙻𝙶
𝑏
14:  if 
𝜏
𝑏
=
∞
 and 
EOS
∉
𝑥
𝐼
𝑡
−
1
,
𝑡
−
1
𝑏
 then
15:   
𝜏
𝑏
=
𝑡
−
1
 and 
pt
𝜏
𝑏
𝑏
=
pt
𝑡
−
1
𝑏
.
16:  end if
17:  Observe 
𝑋
𝑏
,
𝑡
𝑏
=
𝑥
𝐼
𝑡
,
𝑡
𝑏
=
SpecDecSub
​
(
pt
𝑡
−
1
𝑏
,
𝑃
,
𝑆
𝑏
,
𝐿
)
 and 
𝑦
𝑏
,
𝑡
=
len
​
(
𝑋
𝑏
,
𝑡
𝑏
)
.
18:  
pt
𝑡
𝑏
=
concat
​
(
pt
𝑡
−
1
𝑏
,
𝑋
𝑏
,
𝑡
𝑏
)
.
19: end while
20: return 
ST
​
(
𝙰𝙻𝙶
)
=
𝜏
,
pt
ST
​
(
𝙰𝙻𝙶
)
=
pt
𝜏
 and 
ST
​
(
𝙰𝙻𝙶
𝑏
)
=
𝜏
𝑏
,
pt
ST
​
(
𝙰𝙻𝙶
𝑏
)
=
pt
𝜏
𝑏
𝑏
.

See 5.3

Proof of Theorem 5.3.

For ease of presentation, we present Algorithm 10, where the while loop in BanditSpec is repeated until 
max
⁡
{
ST
​
(
𝙰𝙻𝙶
)
,
ST
​
(
𝙰𝙻𝙶
𝑖
)
}
.

Our analysis of Algorithm 10 is novel compared to the standard analysis of EXP3 algorithm (auer2002nonstochastic; lattimore2020bandit). It requires more technical manipulations due to the fact that the termination times of the baseline algorithm 
𝙰𝙻𝙶
𝑖
 and 
𝙰𝙻𝙶
 are different and random, and that our goal is to minimize the stopping time regret (12).

For theoretical analysis, we regard Algorithm 10 as an instantiation of the Follow-the-Regularized-Leader (FTRL) algorithm (gordon1999regret; lattimore2020bandit).

The proof is decomposed into 
5
 steps:

• 

Connection between FTRL and Algorithm 10: we firstly introduce FTRL and the problem where it is applicapable. The shared features and differences are highlighted.

• 

Transformation of the stopping time regret: the stopping time regret is related to the regret under FTRL framework. In this case, the minimized regret by FTRL can be translated to the stopping time regret.

• 

Regret decomposition: the FTRL regret is decomposed for easier processing.

• 

Upper bound each term in the decomposed regret: we upper bound each term in the decomposed regret. The main difficulty is to deal with the difference in the time scales 
ST
​
(
𝙰𝙻𝙶
)
 and 
ST
​
(
𝙰𝙻𝙶
𝑖
)
 and the randomness in 
ST
​
(
𝙰𝙻𝙶
)
. Specifically, because both the loss vectors 
{
𝑍
^
𝑡
}
𝑡
∈
ℕ
 and the stopping time 
ST
​
(
𝙰𝙻𝙶
)
 are random, taking expectation of the cumulative loss within 
[
ST
​
(
𝙰𝙻𝙶
𝑖
)
,
ST
​
(
𝙰𝙻𝙶
)
]
 is non-trivial. We devise Lemma D.1 and Lemma D.2 to deal with this problem.

• 

Conclusion of the stopping time regret: we aggregate the results in the previous steps and derive the final bound for the stopping time regret.

Step 1: Connection between FTRL and Algorithm 5.

We denote 
𝚫
[
𝐾
]
 as the 
𝐾
-dimensional probability simplex.

FTRL is often used in the Online Learning Optimization Problem (OLO). We firstly provide a brief introduction to OLO that operates on 
𝚫
[
𝐾
]
. Let 
ℓ
1
,
ℓ
2
,
…
∈
ℝ
𝐾
 be a sequence of unknown loss vectors. The dynamics of OLO problem is stated in Algorithm 11.

Algorithm 11 Dynamics of the OLO Problem with Full Information Feedback
1: 
ℋ
0
=
∅
.
2: for 
𝑡
=
1
,
2
,
…
,
𝑇
 do
3:  Selects 
𝑝
𝑡
∈
𝚫
[
𝐾
]
 based on 
ℋ
𝑡
−
1
.
4:  Observes the loss vector 
ℓ
𝑡
 and suffers loss 
𝑝
𝑡
⊤
​
ℓ
𝑡
.
5:  
ℋ
𝑡
=
concat
​
(
ℋ
𝑡
−
1
,
(
𝑝
𝑡
,
ℓ
𝑡
)
)
.
6: end for

Given a time horizon 
𝑇
∈
ℕ
, the agent (or algorithm) aims to minimize the (loss-based) regret

	
max
𝑝
∈
𝚫
[
𝐾
]
​
∑
𝑡
=
1
𝑇
(
𝑝
𝑡
−
𝑝
)
⊤
​
ℓ
𝑡
,
		
(45)

where 
𝑝
𝑡
 is the action taken by the agent at time step 
𝑡
, 
𝑝
 is some fixed baseline action in 
𝚫
[
𝐾
]
 and the maximum operator indicates the agent is competing with the best fixed baseline. The FTRL algorithm minimizes (45) by taking the action 
𝑝
𝑡
=
argmin
𝑝
∈
𝚫
[
𝐾
]
Φ
𝑡
​
(
𝑝
)
 at time step 
𝑡
, where 
Φ
𝑡
:
𝚫
[
𝐾
]
→
ℝ
 is defined as

	
Φ
𝑡
​
(
𝑥
)
:=
𝐹
𝑡
​
(
𝑥
)
+
∑
𝑠
=
1
𝑡
−
1
𝑥
⊤
​
ℓ
𝑠
		
(46)

and 
𝐹
𝑡
:
𝚫
[
𝐾
]
→
ℝ
,
∀
𝑡
∈
ℕ
,
 are some convex functions.

In the following, we illustrate the connection between FTRL and Algorithm 5 (or Algorithm 10). We let 
ℓ
𝑡
=
𝑍
^
𝑡
:=
(
𝑍
^
1
,
𝑡
​
…
,
𝑍
^
𝐾
,
𝑡
)
⊤
, 
𝐹
𝑡
​
(
𝑥
)
=
𝐹
​
(
𝑥
)
/
𝜂
𝑡
 with 
𝐹
​
(
𝑥
)
:
𝚫
[
𝐾
]
→
ℝ
 and 
𝐹
​
(
𝑥
)
:=
∑
𝑖
=
1
𝐾
(
𝑥
𝑖
​
log
⁡
𝑥
𝑖
−
𝑥
𝑖
)
+
log
⁡
𝐾
+
1
. Furthermore, some calculation indicates the action taken by FTRL is exactly 
𝑝
𝑡
 as in (43), i.e.,

	
𝑝
𝑡
=
argmin
𝑝
∈
𝚫
[
𝐾
]
Φ
𝑡
​
(
𝑝
)
.
		
(47)

Therefore, Algorithm 5 is indeed an instantiation of FTRL in terms of the algorithm design.

Under our problem setup, the difference lies in the target of the algorithm. Instead of minimizing the corresponding regret

	
max
𝑝
∈
𝚫
[
𝐾
]
⁡
𝔼
​
[
∑
𝑡
=
1
𝑇
(
𝑝
𝑡
−
𝑝
)
⊤
​
𝑍
^
𝑡
]
,
		
(48)

we aim at minimizing the (loss-based) regret defined on two different time scales

	
Reg
~
​
(
𝙰𝙻𝙶
)
:=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑝
𝑡
⊤
​
𝑍
^
𝑡
]
−
min
𝑖
∈
[
𝐾
]
⁡
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
,
		
(49)

where 
𝑒
𝑖
∈
ℝ
𝐾
 is an one-hot vector with the 
𝑖
𝑡
​
ℎ
 coordinate being 
1
, and the expectation is taken w.r.t. the internal randomness within 
𝙰𝙻𝙶
 and 
𝑍
^
𝑡
. We highlight again that 
ST
​
(
𝙰𝙻𝙶
)
 is random, whereas 
ST
​
(
𝙰𝙻𝙶
𝑖
)
 is fixed under the greedy decoding strategy.

Step 2: Transformation of the stopping time regret.

The stopping time regret (12) and 
Reg
~
​
(
𝙰𝙻𝙶
)
 in (49) may look different at first sight. We demonstrate that these two notions of regret can be transformed into one another up to some constant factors.

We firstly simplify 
Reg
~
​
(
𝙰𝙻𝙶
)
. Let 
𝑧
𝑖
,
𝑡
=
1
−
(
𝑦
𝑖
,
𝑡
−
1
)
/
𝐿
. According to the definition of 
𝑝
𝑡
 in (43) and 
𝑍
^
𝑡
 in (44),

	
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑝
𝑡
⊤
​
𝑍
^
𝑡
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝐿
+
1
−
𝑦
𝑖
,
𝑡
𝐿
=
𝐿
+
1
𝐿
⋅
ST
​
(
𝙰𝙻𝙶
)
−
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
𝐿
.
		
(50)

Furthermore, note that 
𝑍
^
𝑡
 is an unbiased estimator for 
𝑧
𝑡
 and 
ST
​
(
𝙰𝙻𝙶
𝑖
)
 is a fixed real number,

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑧
𝑖
,
𝑡
=
𝐿
+
1
𝐿
⋅
ST
​
(
𝙰𝙻𝙶
𝑖
)
−
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
𝑖
)
)
𝐿
.
		
(51)

Hence, 
Reg
~
​
(
𝙰𝙻𝙶
)
 is simplified as

	
Reg
~
​
(
𝙰𝙻𝙶
)
	
=
𝔼
​
[
𝐿
+
1
𝐿
⋅
ST
​
(
𝙰𝙻𝙶
)
−
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
)
)
𝐿
]
−
min
𝑖
∈
[
𝐾
]
⁡
(
𝐿
+
1
𝐿
⋅
ST
​
(
𝙰𝙻𝙶
𝑖
)
−
len
​
(
pt
ST
​
(
𝙰𝙻𝙶
𝑖
)
)
𝐿
)
		
(52)

		
=
𝐿
+
1
𝐿
⋅
(
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
min
𝑖
∈
[
𝐾
]
⁡
ST
​
(
𝙰𝙻𝙶
𝑖
)
)
		
(53)

where we adopt the property of speculative decoding (3) in the last equality. This indicates that

	
Reg
~
​
(
𝙰𝙻𝙶
)
=
𝐿
+
1
𝐿
⋅
Reg
​
(
𝙰𝙻𝙶
)
.
		
(54)

Step 3: Regret decomposition.

Based on the previous two steps, we now upper bound 
Reg
~
​
(
𝙰𝙻𝙶
)
. This notion of regret distinguishes itself from the standard regret analysis due to the difference in the time scales 
ST
​
(
𝙰𝙻𝙶
)
 and 
ST
​
(
𝙰𝙻𝙶
𝑖
)
.

For simplicity, we define the Bregman divergence induced by convex function 
𝑓
:
𝚫
[
𝐾
]
→
ℝ
+
 as 
D
𝑓
​
(
⋅
,
⋅
)
:
𝚫
[
𝐾
]
×
𝚫
[
𝐾
]
→
ℝ
 with 
D
𝑓
​
(
𝑎
,
𝑏
)
:=
𝑓
​
(
𝑎
)
−
𝑓
​
(
𝑏
)
−
(
𝑎
−
𝑏
)
⊤
​
∇
𝑓
​
(
𝑏
)
.
 Fix 
𝑖
∈
[
𝐾
]
, we now decompose this empirical regret w.r.t. 
𝑖
.

	
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑝
𝑡
⊤
​
𝑍
^
𝑡
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
.
		
(55)

	
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
(
𝑝
𝑡
−
𝑝
𝑡
+
1
)
⊤
​
𝑍
^
𝑡
)
+
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑝
𝑡
+
1
⊤
​
𝑍
^
𝑡
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
		
(56)

	
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
(
𝑝
𝑡
−
𝑝
𝑡
+
1
)
⊤
​
𝑍
^
𝑡
)
+
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
Φ
𝑡
+
1
​
(
𝑝
𝑡
+
1
)
−
𝐹
𝑡
+
1
​
(
𝑝
𝑡
+
1
)
−
Φ
𝑡
​
(
𝑝
𝑡
+
1
)
+
𝐹
𝑡
​
(
𝑝
𝑡
+
1
)
)
		
(57)

	
−
(
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
+
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
)
		
(58)

	
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
(
𝑝
𝑡
−
𝑝
𝑡
+
1
)
⊤
​
𝑍
^
𝑡
)
+
∑
𝑡
=
0
ST
​
(
𝙰𝙻𝙶
)
−
1
(
Φ
𝑡
+
1
​
(
𝑝
𝑡
+
1
)
−
Φ
𝑡
+
1
​
(
𝑝
𝑡
+
2
)
)
		
(59)

	
+
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
𝐹
𝑡
​
(
𝑝
𝑡
+
1
)
−
𝐹
𝑡
+
1
​
(
𝑝
𝑡
+
1
)
)
+
𝐹
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑒
𝑖
)
−
𝐹
1
​
(
𝑝
1
)
		
(60)

	
+
Φ
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑝
ST
​
(
𝙰𝙻𝙶
)
+
1
)
−
Φ
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑒
𝑖
)
+
(
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
)
		
(61)

	
≤
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
(
𝑝
𝑡
−
𝑝
𝑡
+
1
)
⊤
​
𝑍
^
𝑡
−
D
𝐹
​
(
𝑝
𝑡
+
1
,
𝑝
𝑡
)
𝜂
𝑡
)
⏟
(
□
)
+
𝐹
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑒
𝑖
)
−
𝐹
1
​
(
𝑝
1
)
⏟
(
⋄
)
		
(62)

	
+
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
𝐹
𝑡
​
(
𝑝
𝑡
+
1
)
−
𝐹
𝑡
+
1
​
(
𝑝
𝑡
+
1
)
)
⏟
(
†
)
+
Φ
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑝
ST
​
(
𝙰𝙻𝙶
)
+
1
)
−
Φ
ST
​
(
𝙰𝙻𝙶
𝑖
)
+
1
​
(
𝑒
𝑖
)
⏟
(
‡
)
		
(63)

	
+
(
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
)
⏟
(
¶
)
		
(64)

where the last inequality adopts the fact that 
D
Φ
𝑡
​
(
𝑎
,
𝑏
)
=
D
𝐹
𝑡
​
(
𝑎
,
𝑏
)
=
D
𝐹
​
(
𝑎
,
𝑏
)
/
𝜂
𝑡
 and the inequality

	
Φ
𝑡
​
(
𝑝
𝑡
)
−
Φ
𝑡
​
(
𝑝
𝑡
+
1
)
=
−
D
Φ
𝑡
​
(
𝑝
𝑡
+
1
,
𝑝
𝑡
)
−
(
𝑝
𝑡
+
1
−
𝑝
𝑡
)
⊤
​
∇
Φ
𝑡
​
(
𝑝
𝑡
)
≤
−
D
Φ
𝑡
​
(
𝑝
𝑡
+
1
,
𝑝
𝑡
)
.
		
(65)

Here 
(
𝑝
𝑡
+
1
−
𝑝
𝑡
)
⊤
​
∇
Φ
𝑡
​
(
𝑝
𝑡
)
≤
0
 results from the choice of 
𝑝
𝑡
=
argmin
𝑝
∈
𝚫
[
𝐾
]
Φ
𝑡
​
(
𝑝
)
 and the first-order optimization condition.

Step 4: Upper bound each term in the decomposed regret.

In this step, we upper bound each term in (64). We comment that 
(
□
)
 and 
(
¶
)
 require us to attend to the randomness in 
ST
​
(
𝙰𝙻𝙶
)
 and the different time indeces. This issue will not be encountered in the conventional scenario where 
ST
​
(
𝙰𝙻𝙶
)
=
ST
​
(
𝙰𝙻𝙶
𝑖
)
.

∙
 Upper bound 
(
□
)
: we will show that 
𝔼
​
[
(
□
)
]
≤
𝔼
​
[
𝐾
​
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
/
2
]
 almost surely.

Recall the definition of 
𝑍
^
𝑡
 in (13), 
𝑍
^
𝑡
 only has a positive value at the 
𝐼
𝑡
 coordinate. We divide the problem into two cases: (1) 
𝑝
𝑡
,
𝐼
𝑡
−
𝑝
𝑡
+
1
,
𝐼
𝑡
<
0
, and (2) 
𝑝
𝑡
,
𝐼
𝑡
−
𝑝
𝑡
+
1
,
𝐼
𝑡
≥
0
.

Case (1): 
𝑝
𝑡
,
𝐼
𝑡
−
𝑝
𝑡
+
1
,
𝐼
𝑡
<
0
. Because the Bregman divergence is always non-negative, hence,

	
(
□
)
≤
(
𝑝
𝑡
,
𝐼
𝑡
−
𝑝
𝑡
+
1
,
𝐼
𝑡
)
⋅
𝑍
^
𝐼
𝑡
,
𝑡
−
0
≤
0
≤
𝜂
𝑡
2
​
𝑝
𝑡
,
𝐼
𝑡
.
		
(66)

Case (2): 
𝑝
𝑡
,
𝐼
𝑡
−
𝑝
𝑡
+
1
,
𝐼
𝑡
≥
0
. Note that 
𝐹
​
(
𝑥
)
 is a Legendre function on 
𝚫
[
𝐾
]
. By invoking Lemma F.8,

	
(
𝑝
𝑡
−
𝑝
𝑡
+
1
)
⊤
​
𝑍
^
𝑡
−
D
𝐹
​
(
𝑝
𝑡
+
1
,
𝑝
𝑡
)
𝜂
𝑡
≤
𝜂
𝑡
2
​
‖
𝑍
^
𝑡
‖
𝐻
𝑡
−
1
2
,
		
(67)

where 
𝐻
𝑡
=
∇
2
𝐹
​
(
𝑞
𝑡
)
 and 
𝑞
𝑡
=
𝛼
⋅
𝑝
𝑡
+
(
1
−
𝛼
)
⋅
𝑝
𝑡
+
1
 for some 
𝛼
∈
[
0
,
1
]
. Furthermore, 
∇
2
𝐹
​
(
𝑞
𝑡
)
 is a 
𝐾
×
𝐾
 diagonal matrix with 
(
∇
2
𝐹
​
(
𝑞
𝑡
)
)
𝑖
,
𝑖
=
1
/
𝑞
𝑡
,
𝑖
,
∀
𝑖
∈
[
𝐾
]
. Therefore,

	
𝜂
𝑡
2
​
‖
𝑍
^
𝑡
‖
𝐻
𝑡
−
1
2
=
𝜂
𝑡
2
⋅
𝑧
𝐼
𝑡
,
𝑡
2
𝑝
𝑡
,
𝐼
𝑡
2
⋅
𝑞
𝑡
,
𝐼
𝑡
≤
𝜂
𝑡
2
⋅
1
𝑝
𝑡
,
𝐼
𝑡
2
⋅
𝑝
𝑡
,
𝐼
𝑡
=
𝜂
𝑡
2
​
𝑝
𝑡
,
𝐼
𝑡
.
		
(68)

To conclude the two cases, it holds almost surely that

	
(
□
)
≤
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
2
​
𝑝
𝑡
,
𝐼
𝑡
.
		
(69)

Lastly, by adopting Lemma D.1 which is proved in Appendix E.1, we obtain

	
𝔼
​
[
(
□
)
]
≤
𝐾
2
⋅
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
.
		
(70)
Lemma D.1.

Under Assumption 4.2, consider the learning rates 
𝜂
𝑡
 defined in Algorithm 5 (or Algorithm 10), it holds that

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
]
=
𝐾
⋅
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
.
		
(71)

∙
 Upper bound 
(
⋄
)
: Because 
𝐹
​
(
𝑥
)
 is non-negative on 
𝚫
[
𝐾
]
 and 
𝐹
​
(
𝑒
𝑖
)
=
log
⁡
𝐾
, hence,

	
(
⋄
)
≤
𝐹
​
(
𝑒
𝑖
)
𝜂
ST
​
(
𝙰𝙻𝙶
+
1
)
=
log
⁡
𝐾
𝜂
ST
​
(
𝙰𝙻𝙶
)
+
1
=
log
⁡
𝐾
𝜂
ST
​
(
𝙰𝙻𝙶
)
.
		
(72)

where we manually set 
𝜂
ST
​
(
𝙰𝙻𝙶
)
+
1
=
𝜂
ST
​
(
𝙰𝙻𝙶
)
.

∙
 Upper bound 
(
†
)
: Recall that 
𝐹
​
(
𝑥
)
=
∑
𝑖
=
1
𝐾
(
𝑥
𝑖
​
log
⁡
𝑥
𝑖
−
𝑥
𝑖
)
+
log
⁡
𝐾
+
1
, so 
𝐹
​
(
𝑥
)
 is non-negative for any 
𝑥
∈
𝚫
[
𝐾
]
. Additionally, 
𝜂
𝑡
=
log
⁡
𝐾
/
(
𝐾
⋅
𝑡
)
,
𝑡
∈
[
ST
​
(
𝙰𝙻𝙶
)
]
 is a decreasing sequence. Therefore,

	
(
†
)
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
𝐹
​
(
𝑝
𝑡
+
1
)
𝜂
𝑡
−
𝐹
​
(
𝑝
𝑡
+
1
)
𝜂
𝑡
+
1
)
≤
0
,
𝑎
.
𝑠
.
		
(73)

∙
 Upper bound 
(
‡
)
: Since 
𝑝
ST
​
(
𝙰𝙻𝙶
)
+
1
 is the minimizer of 
Φ
ST
​
(
𝙰𝙻𝙶
)
+
1
​
(
𝑝
)
, we have 
(
‡
)
≤
0
.

∙
 Upper bound 
(
¶
)
: Under Assumption 5.2, 
ST
​
(
𝙰𝙻𝙶
)
≥
ST
​
(
𝙰𝙻𝙶
1
)
 almost surely. We further prove that 
𝔼
​
[
(
¶
)
]
≤
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
)
 for 
𝑖
=
𝑖
∗
, which we summarize in the following lemma with proof postponed to Appendix E.1.

Lemma D.2.

Under Assumption 4.2 and 5.2, consider 
𝑍
^
𝑡
 defined in Algorithm 5, it holds that

	
𝔼
​
[
∑
𝑡
=
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
+
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
∗
⊤
​
𝑍
^
𝑡
]
≤
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
.
		
(74)

Step 5: Conclusion of the stopping time regret. Aggregating the upper bounds for each terms in (64) and taking expectation,

	
Reg
~
​
(
𝙰𝙻𝙶
)
≤
𝐾
2
​
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
+
𝔼
​
[
log
⁡
𝐾
𝜂
ST
​
(
𝙰𝙻𝙶
)
]
+
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
.
		
(75)

By substituting the learning rate values into the equation,

	
Reg
~
​
(
𝙰𝙻𝙶
)
	
≤
2
⋅
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
⋅
𝐾
​
log
⁡
𝐾
]
+
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
		
(76)

		
≤
2
⋅
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
⋅
𝐾
​
log
⁡
𝐾
+
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
		
(77)

		
≤
2
⋅
len
​
(
pt
𝜏
c
)
⋅
𝐾
​
log
⁡
𝐾
+
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
.
		
(78)

where the second inequality adopts Jensen’s inequality and the last equality holds due to 
ST
​
(
𝙰𝙻𝙶
)
≤
len
​
(
pt
𝜏
c
)
 almost surely as in (4). According to the regret transformation in (54),

	
Reg
​
(
𝙰𝙻𝙶
)
≤
2
​
𝐿
⋅
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
⋅
𝐾
​
log
⁡
𝐾
≤
2
​
𝐿
⋅
len
​
(
pt
𝜏
c
)
⋅
𝐾
​
log
⁡
𝐾
.
		
(79)

Furthermore, by solving the quadratic function in terms of 
Reg
​
(
𝙰𝙻𝙶
)
, i.e.,

	
Reg
​
(
𝙰𝙻𝙶
)
≤
2
​
𝐿
⋅
(
Reg
​
(
𝙰𝙻𝙶
)
+
ST
​
(
𝙰𝙻𝙶
𝑖
∗
)
)
⋅
𝐾
​
log
⁡
𝐾
,
		
(80)

we obtain

	
Reg
​
(
𝙰𝙻𝙶
)
≤
4
​
𝐿
2
⋅
𝐾
​
log
⁡
𝐾
+
2
​
𝐿
⋅
min
𝑖
∈
[
𝐾
]
⁡
ST
​
(
𝙰𝙻𝙶
𝑖
)
⋅
𝐾
​
log
⁡
𝐾
.
		
(81)

Aggregating (79) and (81) concludes the proof of this theorem. ∎

Remark D.3 (Sampling Decoding under Adversarial Mean Values).

Since the tokens can be regarded as fixed given the initial prompt and the hyperparameter configurations under the greedy decoding strategy, we consider the greedy decoding strategy under the adversarial mean values assumption (Assumption 5.1). If one wishes to consider the sampling decoding strategy, the proof of Theorem 5.3 can be adapted to it. Specifically, this switch of decoding strategy mainly influences (51), the proofs of Lemma D.1 and Lemma D.2. We can depend on Doob’s Optional Stopping Theorem (Lemma F.1) to solve this problem, just like what we have done to prove (100) and replacing the condition (1) therein by 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
≤
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
<
∞
. The rest of the proof can go through in a similar manner. In the end, we can arrive at a similar result, i.e.,

	
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
≤
2
​
𝐿
⋅
min
⁡
{
2
​
𝐿
​
𝐾
​
log
⁡
𝐾
+
min
𝑖
∈
[
𝐾
]
⁡
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
𝑖
)
]
​
𝐾
​
log
⁡
𝐾
,
𝔼
​
[
len
​
(
pt
𝜏
c
)
]
​
𝐾
​
log
⁡
𝐾
}
.
		
(82)
D.4Proof of Theorem 4.4

Under the greedy decoding strategy, the problem is alleviated in two aspects. Firstly, given the target model 
𝑃
 and the initial prompt 
pt
, the total length 
len
​
(
pt
𝜏
c
)
 is (potentially) determined. While the total length is determined, it is worth noting that the number of accepted tokens at each round (Line 5 in Algorithm 3) is still random. Additionally, under the dynamics presented in Algorithm 3 and given the history 
ℋ
𝑡
, there is a one-to-one mapping between the accepted tokens 
𝑋
𝐼
𝑡
,
𝑡
 and its length 
𝑌
𝐼
𝑡
,
𝑡
.

Since the lower bound is established in terms of a class of algorithms over a set of initial prompts, we adopt 
𝒳
𝑖
​
𝑛
​
𝑖
​
𝑡
​
𝑖
​
𝑎
​
𝑙
∗
 to denote the set of initial prompts and adopt 
𝒮
𝑎
​
𝑙
​
𝑙
 to denote the set of all hyperparameter specifications that can be selected to constitute 
𝒮
. To further ease the problem, we augment Assumption 4.1.

Assumption D.4.

We assume that

∙
 Given any bandit configuration 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
∈
[
𝐾
]
,
𝐿
)
 and initial prompt 
pt
∈
𝒳
𝑖
​
𝑛
​
𝑖
​
𝑡
​
𝑖
​
𝑎
​
𝑙
∗
, conditional on the history 
ℋ
𝑡
−
1
 and the selected arm 
𝐼
𝑡
 at round 
𝑡
, the distribution of the length of the accepted tokens 
ℙ
(
⋅
∣
pt
,
ℋ
𝑡
−
1
,
𝐼
𝑡
=
𝑖
)
=
𝑃
𝑆
𝑖
(
⋅
)
,
∀
𝑖
∈
[
𝐾
]
.

∙
 For any two hyperparameter specifications 
𝑆
,
𝑆
′
∈
𝒮
𝑎
​
𝑙
​
𝑙
, we have 
KL
​
(
𝑃
𝑆
,
𝑃
𝑆
′
)
<
∞
.

We consider the class of arm selection algorithms which are non-anticipatory and consistent.

Definition D.5 (Non-anticipartory Algorithm).

An arm selection algorithm 
𝙰𝙻𝙶
 is non-anticipatory if 
𝙰𝙻𝙶
(
⋅
∣
ℋ
𝑡
)
∈
𝜎
(
ℋ
𝑡
)
,
∀
𝑡
∈
ℕ
.

Definition D.6 (Consistent Algoirthm).

An arm selection algorithm 
𝙰𝙻𝙶
 is consistent over a class of bandit configurations 
Λ
 and prompt set 
𝒳
𝑖
​
𝑛
​
𝑖
​
𝑡
​
𝑖
​
𝑎
​
𝑙
∗
 if for all 
𝜈
∈
Λ
 and any sequence of initial prompts 
(
pt
𝑚
)
𝑚
=
1
∞
⊂
𝒳
𝑖
​
𝑛
​
𝑖
​
𝑡
​
𝑎
​
𝑙
∗
 with 
len
​
(
pt
𝜏
c
𝑚
)
→
∞
,
𝑚
→
∞
, and for all 
𝑎
∈
(
0
,
1
]
,

	
lim
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
𝑚
,
𝜈
)
len
​
(
pt
𝜏
c
𝑚
)
𝑎
=
0
.
		
(83)

See 4.4

Proof of Theorem 4.4.

The proof consists of three steps:

• 

Divergence decomposition: similar to the reward decomposition in the upper bound proof, the divergence decomposition cannot be done as the time horizon 
ST
​
(
𝙰𝙻𝙶
)
 is a random stopping time. We tackle this problem in the first step.

• 

Lower bound on 
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
: we adapt the standard trick to lower bound the expected number of times arm 
𝑖
 has been chosen.

• 

Conclusion of the proof.

Step 1: Divergence decomposition. The divergence decomposition suffers from the same issue as the reward decomposition, i.e., the stopping time 
ST
​
(
𝙰𝙻𝙶
)
 depends on the history. We adopt the same trick as in the reward decomposition step to overcome this issue and the result is summarized in Lemma D.7 whose proof is postponed to App. E.

Lemma D.7.

Under Assumption D.4, given two bandit configurations 
𝜈
=
(
𝑃
,
𝒮
=
{
𝑆
𝑖
}
𝑖
=
1
𝐾
,
𝐿
)
 and 
𝜈
′
=
(
𝑃
,
𝒮
′
=
{
𝑆
𝑖
′
}
𝑖
=
1
𝐾
,
𝐿
)
 which only differ in the hyperparameter specifications, for any 
pt
∈
𝒳
𝑖
​
𝑛
​
𝑖
​
𝑡
​
𝑖
​
𝑎
​
𝑙
∗
 and algorithm 
𝙰𝙻𝙶
,

	
KL
​
(
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
,
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
′
)
=
∑
𝑖
=
1
𝐾
𝔼
𝙰𝙻𝙶
,
pt
,
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
​
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
		
(84)

where 
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
 (resp. 
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
) is the probability measure induced by 
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
 (resp. 
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
) defined on the 
𝜎
-algebra 
{
𝜎
​
(
ℋ
𝑡
)
}
𝑡
=
1
∞
.

Step 2: Establishment for the lower bound of 
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
. Given algorithm 
𝙰𝙻𝙶
, bandit configuration 
𝜈
∈
Λ
, prompt 
pt
∈
(
pt
𝑚
)
𝑚
=
1
∞
 and any 
𝜀
>
0
, construct 
𝐾
−
1
 alternative bandit configurations 
𝜈
𝑖
=
(
𝑃
,
𝒮
𝑖
=
{
𝑆
𝑖
,
𝑗
}
𝑗
=
1
𝐾
,
𝐿
)
 for 
𝑖
≠
𝑖
∗
 with

	
𝑆
𝑖
,
𝑗
=
𝑆
𝑗
⋅
𝟙
​
{
𝑗
≠
𝑖
}
+
𝑆
𝑖
′
⋅
𝟙
​
{
𝑗
=
𝑖
}
,
		
(85)

where 
𝑆
𝑖
′
 in 
𝜈
𝑖
 satisfies that its mean 
𝜇
𝑆
𝑖
′
>
𝜇
𝑖
∗
 with 
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
≤
kl
𝑖
+
𝜀
. In other words, under bandit configuration 
𝜈
𝑖
, only 
𝑆
𝑖
 changes into 
𝑆
𝑖
′
 and arm 
𝑖
 becomes the best arm. As the bandit configurations only differ in the hyperparameter selection, we adopt the shorthand notation 
ℙ
𝜈
,
𝔼
𝜈
 and 
Reg
​
(
𝜈
)
 for 
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
,
𝔼
𝙰𝙻𝙶
,
pt
,
𝜈
 and 
Reg
​
(
𝙰𝙻𝙶
,
pt
,
𝜈
)
 respectively when there is no risk of confusion.

According to Lemma D.7, for any 
pt
∈
(
pt
𝑚
)
𝑚
=
1
∞
,

	
KL
​
(
ℙ
𝜈
,
ℙ
𝜈
𝑖
)
=
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
⋅
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
≤
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
⋅
(
kl
𝑖
+
𝜀
)
.
		
(86)

By Lemma F.6, with 
𝐴
𝑖
=
{
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
>
ST
​
(
𝙰𝙻𝙶
)
/
2
}
,

	
ℙ
𝜈
​
[
𝐴
𝑖
]
+
ℙ
𝜈
𝑖
​
[
𝐴
𝑖
𝑐
]
	
≥
1
2
​
exp
⁡
(
−
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
⋅
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
)
		
(87)

		
≥
1
2
​
exp
⁡
(
−
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
⋅
(
kl
𝑖
+
𝜀
)
)
		
(88)

Thus, by adopting (36), the expected reward under 
𝜈
 can be lower bounded as

	
Reg
​
(
𝜈
)
	
≥
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
≥
Δ
𝑖
𝜇
𝑖
∗
⋅
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
∣
𝐴
𝑖
]
⋅
ℙ
𝜈
​
[
𝐴
𝑖
]
		
(89)

		
≥
Δ
𝑖
𝜇
𝑖
∗
⋅
1
2
⋅
𝔼
𝜈
​
[
ST
​
(
𝙰𝙻𝙶
)
∣
𝐴
𝑖
]
⋅
ℙ
𝜈
​
[
𝐴
𝑖
]
≥
Δ
𝑖
𝜇
𝑖
∗
⋅
len
​
(
pt
𝜏
c
)
2
​
(
𝐿
+
1
)
⋅
ℙ
𝜈
​
[
𝐴
𝑖
]
.
		
(90)

Similarly, under the alternative bandit configuration 
𝜈
𝑖
,

	
Reg
​
(
𝜈
𝑖
)
≥
𝜇
𝑆
𝑖
′
−
𝜇
𝑖
∗
𝜇
𝑆
𝑖
′
⋅
𝔼
𝜈
𝑖
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
≥
𝜇
𝑆
𝑖
′
−
𝜇
𝑖
∗
𝜇
𝑆
𝑖
′
⋅
len
​
(
pt
𝜏
c
)
2
​
(
𝐿
+
1
)
⋅
ℙ
𝜈
𝑖
​
[
𝐴
𝑖
𝑐
]
.
		
(91)

Combining (87), (89) and (91),

	
Reg
​
(
𝜈
)
+
Reg
​
(
𝜈
𝑖
)
		
(92)

	
≥
min
⁡
{
Δ
𝑖
𝜇
𝑖
∗
,
𝜇
𝑆
𝑖
′
−
𝜇
𝑖
∗
𝜇
𝑆
𝑖
′
}
⋅
len
​
(
pt
𝜏
c
)
2
​
(
𝐿
+
1
)
​
(
ℙ
𝜈
​
[
𝐴
𝑖
]
+
ℙ
𝜈
𝑖
​
[
𝐴
𝑖
𝑐
]
)
		
(93)

	
≥
min
⁡
{
Δ
𝑖
𝜇
𝑖
∗
,
𝜇
𝑆
𝑖
′
−
𝜇
𝑖
∗
𝜇
𝑆
𝑖
′
}
⋅
len
​
(
pt
𝜏
c
)
4
​
(
𝐿
+
1
)
⋅
exp
⁡
(
−
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
⋅
(
kl
𝑖
+
𝜀
)
)
		
(94)

which holds for any 
pt
∈
(
pt
𝑚
)
𝑚
=
1
∞
. Rearranging the terms, we have

	
lim inf
𝑚
→
∞
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
		
(95)

	
≥
1
kl
𝑖
+
𝜀
+
lim inf
𝑚
→
∞
log
⁡
(
min
⁡
{
Δ
𝑖
𝜇
𝑖
∗
,
𝜇
𝑆
𝑖
′
−
𝜇
𝑖
∗
𝜇
𝑆
𝑖
′
}
)
−
log
⁡
(
4
​
(
𝐿
+
1
)
)
−
log
⁡
(
Reg
​
(
𝜈
)
+
Reg
​
(
𝜈
𝑖
)
)
(
kl
𝑖
+
𝜀
)
⋅
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
		
(96)

	
=
1
kl
𝑖
+
𝜀
		
(97)

where the last equality follows from the definition of a consistent algorithm. Because 
𝜀
>
0
 is arbitrarily chosen, by sending 
𝜀
→
0
, we obtain the lower bound for 
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]

		
lim inf
𝑚
→
∞
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
1
kl
𝑖
.
		
(98)

Step 3: Conclusion of the proof. Aggregating (36) and (98),

	
lim inf
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
𝑚
,
𝜈
)
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
∑
𝑖
≠
𝑖
∗
Δ
𝑖
𝜇
𝑖
∗
⋅
1
kl
𝑖
.
		
(99)

This concludes the proof. ∎

Appendix ESupporting Propositions
E.1Supporting Lemmas for Theorem 5.3

See D.1

Proof of Lemma D.1.

Similar to the proof of Lemma D.2, we adopt Doob’s Optional Stopping Theorem (Lemma F.1) to show

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
−
𝐾
⋅
𝜂
𝑡
]
=
0
.
		
(100)

According to condition 
(
𝑏
)
 in Lemma F.1, we only need to show that (1) 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
, and (2) there exists 
𝑐
∈
ℝ
 such that for all 
𝑡
<
ST
​
(
𝙰𝙻𝙶
)
, 
𝔼
​
[
|
𝜂
𝑡
/
𝑝
𝑡
,
𝐼
𝑡
−
𝐾
⋅
𝜂
𝑡
|
|
ℋ
𝑡
−
1
]
<
𝑐
.

Condition 
(
1
)
: Given Assumption 4.2, it holds that 
len
​
(
pt
𝜏
𝑐
)
<
∞
 under the greedy decoding strategy. Therefore, we have 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
≤
len
​
(
pt
𝜏
𝑐
)
<
∞
.

Condition 
(
2
)
: Note that

	
𝔼
​
[
|
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
−
𝐾
⋅
𝜂
𝑡
|
|
ℋ
𝑡
−
1
]
	
≤
𝔼
​
[
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
|
ℋ
𝑡
−
1
]
+
𝐾
⋅
𝜂
𝑡
≤
𝔼
​
[
∑
𝑖
=
1
𝐾
𝜂
𝑡
𝑝
𝑡
,
𝑖
​
𝟙
​
{
𝐼
𝑡
=
𝑖
}
|
ℋ
𝑡
−
1
]
+
𝐾
⋅
𝜂
𝑡
		
(101)

		
≤
2
​
𝐾
⋅
𝜂
𝑡
≤
2
​
𝐾
​
log
⁡
𝐾
.
		
(102)

Therefore, Condition 
(
2
)
 is satisfied and (100) is established.

In addition, by using Assumption 4.2,

	
𝔼
​
[
|
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝐾
⋅
𝜂
𝑡
|
]
≤
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
​
𝐾
​
log
⁡
𝐾
<
∞
.
		
(103)

So 
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
 exists. Lastly,

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
𝑝
𝑡
,
𝐼
𝑡
−
𝐾
⋅
𝜂
𝑡
]
+
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
		
(104)

which indicates 
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
/
𝑝
𝑡
,
𝐼
𝑡
]
 exists.

In conclusion, by adding 
𝐾
⋅
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝜂
𝑡
]
 on both sides of (100), the desired result is obtained. ∎

See D.2

Proof of Lemma D.2.

If the two expectations below exist and

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
,
		
(105)

then it holds that

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
−
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
−
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
𝑖
)
𝑧
𝑖
,
𝑡
		
(106)

	
≤
𝔼
​
[
∑
𝑡
=
ST
​
(
𝙰𝙻𝙶
𝑖
)
+
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
≤
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
−
ST
​
(
𝙰𝙻𝙶
𝑖
)
.
		
(107)

The desired result can be obtained.

Therefore, we aim to prove (105). Since both 
ST
​
(
𝙰𝙻𝙶
)
 and 
𝑍
^
𝑡
 are random, the obstacle is that we cannot directly take expectation of the summand. We will firstly prove

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
𝑧
𝑖
,
𝑡
]
=
0
		
(108)

by Doob’s Optional Stopping Theorem (Lemma F.1). According to condition 
(
𝑏
)
 in Lemma F.1, we only need to show that (1) 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
, and (2) there exists 
𝑐
∈
ℝ
 such that for all 
𝑡
<
ST
​
(
𝙰𝙻𝙶
)
, 
𝔼
​
[
|
𝑒
𝑖
⊤
​
(
𝑍
^
𝑡
−
𝑧
𝑡
)
|
|
ℋ
𝑡
−
1
]
<
𝑐
.

Condition 
(
1
)
 holds as shown in the proof Lemma D.1. We only need to show Condition 
(
2
)
. Note that 
𝔼
​
[
𝑍
^
]
=
𝑧
𝑡
,

	
𝔼
​
[
|
𝑒
𝑖
⊤
​
(
𝑍
^
𝑡
−
𝑧
𝑡
)
|
|
ℋ
𝑡
−
1
]
≤
𝔼
​
[
𝔼
​
[
𝑒
𝑖
⊤
​
𝑍
^
𝑡
∣
ℋ
𝑡
−
1
,
𝐼
𝑡
]
|
ℋ
𝑡
−
1
]
+
𝑒
𝑖
⊤
​
𝑧
𝑡
=
2
​
𝑧
𝑖
,
𝑡
≤
2
.
		
(109)

Therefore, 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
𝑧
𝑖
,
𝑡
 is well-defined and (108) is established.

We then prove the two expectations exist. Note that all involved variables are positive, we only need to show

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
<
∞
.
		
(110)

This can be obtained by noticing that 
𝑧
𝑖
,
𝑡
∈
[
0
,
1
]
 for 
𝑡
∈
ℕ
,

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
≤
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
.
		
(111)

Combining the above with Condition 2, we have

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
−
𝑝
⊤
​
𝑧
𝑡
]
+
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑧
𝑡
]
		
(112)

which means 
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
 exists.

In conclusion, by adding 
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
 on both sides of (108),

	
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑒
𝑖
⊤
​
𝑍
^
𝑡
]
=
𝔼
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑧
𝑖
,
𝑡
]
.
		
(113)

This finishes the proof of (105). ∎

E.2Proof of Lemma D.7

See D.7

Proof.

As the two bandit configurations only differ in the hyperparameter specifications, we adopt the abbreviated notation 
ℙ
𝜈
 and 
ℙ
𝜈
′
 for the induced probability 
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
 and 
ℙ
𝙰𝙻𝙶
,
pt
,
𝜈
, respectively. We use 
𝑃
𝙰𝙻𝙶
​
(
⋅
)
 to denote the output distribution of the arm selection algorithm 
𝙰𝙻𝙶
 in Line 4 in Algorithm 3.

With the bandit configuration 
𝜈
=
(
𝑃
,
𝒮
,
𝐿
)
, the probability of 
ℋ
ST
​
(
𝙰𝙻𝙶
)
 is

	
ℙ
𝜈
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
=
∏
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑃
𝙰𝙻𝙶
​
(
𝐼
𝑡
∣
ℋ
𝑡
−
1
,
pt
)
​
ℙ
​
(
𝑌
𝐼
𝑡
,
𝑡
∣
ℋ
𝑡
−
1
,
pt
,
𝐼
𝑡
)
=
∏
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑃
𝙰𝙻𝙶
​
(
𝐼
𝑡
∣
ℋ
𝑡
−
1
,
pt
)
​
𝑃
𝑆
𝐼
𝑡
​
(
𝑌
𝐼
𝑡
,
𝑡
)
.
		
(114)

Similarly, under the bandit configuration 
𝜈
′
=
(
𝑃
,
𝒮
′
,
𝐿
)
,

	
ℙ
𝜈
′
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
=
∏
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝑃
𝙰𝙻𝙶
​
(
𝐼
𝑡
∣
ℋ
𝑡
−
1
,
pt
)
​
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑌
𝐼
𝑡
,
𝑡
)
.
		
(115)

Therefore, it holds that

	
log
⁡
ℙ
𝜈
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
ℙ
𝜈
′
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
=
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑌
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑌
𝐼
𝑡
,
𝑡
)
.
		
(116)

Because 
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
<
∞
,
∀
𝑖
∈
[
𝐾
]
 under Assumption D.4, the divergence between 
ℙ
𝜈
 and 
ℙ
𝜈
′
 can be rewritten as

	
KL
​
(
ℙ
𝜈
,
ℙ
𝜈
′
)
=
𝔼
𝜈
​
[
log
⁡
ℙ
𝜈
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
ℙ
𝜈
′
​
(
ℋ
ST
​
(
𝙰𝙻𝙶
)
)
]
=
𝔼
𝜈
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑌
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑌
𝐼
𝑡
,
𝑡
)
]
<
∞
.
		
(117)

We then prove that

	
𝔼
𝜈
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑌
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑌
𝐼
𝑡
,
𝑡
)
]
=
∑
𝑖
=
1
𝐾
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
​
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
.
		
(118)

The proof is composed by two arguments:

• 

Argument 1:

	
𝔼
𝜈
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
(
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑌
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑌
𝐼
𝑡
,
𝑡
)
−
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
)
]
=
0
.
		
(119)
• 

Argument 2:

	
𝔼
𝜈
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
]
=
∑
𝑖
=
1
𝐾
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
​
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
.
		
(120)

If the two arguments are true, by summing up (119) and (120), we can obtain the desired result (118).

We prove the two above Arguments.

Argument 1. Let

	
𝑀
𝑛
:=
∑
𝑡
=
1
𝑛
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑋
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑋
𝐼
𝑡
,
𝑡
)
−
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
,
𝑛
=
1
,
2
,
…
		
(121)

and 
𝑀
0
:=
0
.

We firstly prove that 
(
𝑀
𝑛
)
𝑛
=
0
∞
 is a martingale w.r.t. 
(
ℋ
𝑛
)
𝑛
=
0
∞
: (1) 
𝔼
𝜈
​
[
|
𝑀
𝑛
|
]
<
∞
, and (2) 
𝔼
𝜈
​
[
𝑀
𝑛
+
1
∣
ℋ
𝑛
]
=
𝑀
𝑛
.

(1) 
𝔼
𝜈
​
[
|
𝑀
𝑛
|
]
<
∞
. According to Assumption D.4, for any 
𝑖
∈
[
𝐾
]
, 
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
<
∞
, this indicates there exists 
𝑐
∈
ℝ
 such that

	
∑
𝑥
=
1
𝐿
+
1
𝑃
𝑆
𝑖
​
(
𝑥
)
​
|
log
⁡
𝑃
𝑆
𝑖
​
(
𝑥
)
𝑃
𝑆
𝑖
′
​
(
𝑥
)
|
<
𝑐
<
∞
,
∀
𝑖
∈
[
𝐾
]
.
		
(122)

This indicates

		
𝔼
𝜈
​
[
|
𝑀
𝑛
|
]
≤
∑
𝑡
=
1
𝑛
𝔼
𝜈
​
[
|
log
⁡
𝑃
𝑆
𝐼
𝑡
​
(
𝑋
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑋
𝐼
𝑡
,
𝑡
)
|
+
|
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
|
]
<
∞
.
		
(123)

(2) 
𝔼
𝜈
​
[
𝑀
𝑛
+
1
∣
ℋ
𝑛
]
=
𝑀
𝑛
. By adopting the tower property.

	
𝔼
𝜈
​
[
𝑀
𝑛
+
1
∣
ℋ
𝑛
]
	
=
𝑀
𝑛
+
𝔼
𝜈
​
[
𝔼
𝜈
​
[
log
⁡
𝑃
𝑆
𝐼
𝑛
+
1
​
(
𝑌
𝐼
𝑛
+
1
,
𝑛
+
1
)
𝑃
𝑆
𝐼
𝑛
+
1
′
​
(
𝑌
𝐼
𝑛
+
1
,
𝑛
+
1
)
−
KL
​
(
𝑃
𝑆
𝐼
𝑛
+
1
,
𝑃
𝑆
𝐼
𝑛
+
1
′
)
|
ℋ
𝑛
,
𝐼
𝑛
+
1
]
|
ℋ
𝑛
]
		
(124)

		
=
𝑀
𝑛
.
		
(125)

From (123) and (125), 
(
𝑀
𝑛
)
𝑛
∈
ℕ
 is a martingale w.r.t. 
(
ℋ
𝑛
)
𝑛
∈
ℕ
.

Additionally, we will adopt Doob’s Optional Stopping Theorem (Lemma F.1) on 
𝑀
ST
​
(
𝙰𝙻𝙶
)
. The prerequisites are verified as follows:

(1) 
𝔼
​
[
ST
​
(
𝙰𝙻𝙶
)
]
<
∞
: By Assumption 4.2, 
ST
​
(
𝙰𝙻𝙶
)
 is a stopping time w.r.t. 
(
ℋ
𝑛
)
𝑛
∈
ℕ
 with 
𝔼
[
ST
(
𝙰𝙻𝙶
)
]
≤
len
(
pt
𝜏
c
<
∞
.

(2) there exists 
𝑐
¯
∈
ℝ
, such that 
𝔼
​
[
|
𝑀
𝑛
+
1
−
𝑀
𝑛
|
∣
ℋ
𝑛
]
≤
𝑐
¯
 for any 
𝑛
≤
ST
​
(
𝙰𝙻𝙶
)
: According to (122),

	
𝔼
[
|
𝑀
𝑛
+
1
−
𝑀
𝑛
|
∣
ℋ
𝑛
]
≤
𝔼
𝜈
[
|
log
𝑃
𝑆
𝐼
𝑡
​
(
𝑋
𝐼
𝑡
,
𝑡
)
𝑃
𝑆
𝐼
𝑡
′
​
(
𝑋
𝐼
𝑡
,
𝑡
)
|
|
ℋ
𝑛
]
+
𝔼
𝜈
[
KL
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
|
]
≤
2
𝑐
<
∞
.
		
(126)

Taking 
𝑐
¯
=
2
​
𝑐
 finishes the verification.

Therefore, the prerequisites in Lemma F.1 (b) are satisfied. By invoking Lemma F.1, (119) is established.

Argument 2. Firstly, by (122),

	
𝔼
𝜈
​
[
|
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
|
]
≤
𝔼
𝜈
​
[
ST
​
(
𝙰𝙻𝙶
)
]
⋅
𝑐
<
∞
.
		
(127)

So 
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
 has finite expectation. Furthermore, it can be observed that

	
𝔼
𝜈
​
[
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
KL
​
(
𝑃
𝑆
𝐼
𝑡
,
𝑃
𝑆
𝐼
𝑡
′
)
]
	
=
𝔼
𝜈
​
[
∑
𝑖
=
1
𝐾
∑
𝑡
=
1
ST
​
(
𝙰𝙻𝙶
)
𝟙
​
{
𝐼
𝑡
=
𝑖
}
​
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
]
		
(128)

		
=
∑
𝑖
=
1
𝐾
𝔼
𝜈
​
[
𝑛
𝑖
,
ST
​
(
𝙰𝙻𝙶
)
]
​
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
𝑖
′
)
.
		
(129)

Therefore, (120) is proved.

This concludes the proof of this divergence decomposition lemma. ∎

E.3Proof of Proposition 4.5

See 4.5

Proof of Proposition 4.5.

Given any 
𝑆
∈
𝒮
 with parameter 
𝑝
,

	
𝜇
𝑆
=
∑
𝑥
=
1
𝐿
+
1
𝑥
⋅
𝑃
𝑆
​
(
𝑥
)
=
∑
𝑥
=
1
𝐿
𝑥
⋅
𝑝
𝑥
−
1
​
(
1
−
𝑝
)
+
(
𝐿
+
1
)
⋅
𝑝
𝐿
=
1
−
𝑝
𝐿
+
1
1
−
𝑝
.
		
(130)

Note that if 
𝜇
𝑆
≥
𝜇
𝑖
, we have 
𝑝
>
𝑝
𝑖
. Therefore,

	
𝜇
𝑆
−
𝜇
𝑖
	
=
1
−
𝑝
𝐿
+
1
1
−
𝑝
−
1
−
𝑝
𝑖
𝐿
+
1
1
−
𝑝
𝑖
=
(
𝑝
−
𝑝
𝑖
)
+
(
𝑝
𝑖
𝐿
+
1
−
𝑝
𝐿
+
1
+
𝑝
𝑖
​
𝑝
𝐿
+
1
−
𝑝
​
𝑝
𝑖
𝐿
+
1
)
(
1
−
𝑝
)
​
(
1
−
𝑝
𝑖
)
		
(131)

		
≥
(
𝑝
−
𝑝
𝑖
)
+
(
𝑝
𝑖
𝐿
+
1
−
𝑝
𝐿
+
1
)
(
1
−
𝑝
)
​
(
1
−
𝑝
𝑖
)
≥
(
𝑝
−
𝑝
𝑖
)
​
(
1
−
𝑝
𝐿
)
(
1
−
𝑝
)
​
(
1
−
𝑝
𝑖
)
.
		
(132)

In addition, for any 
𝑖
∈
[
𝐾
]
,

	
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
)
=
∑
𝑥
=
1
𝐿
+
1
𝑃
𝑆
𝑖
​
(
𝑥
)
⋅
log
⁡
𝑃
𝑆
𝑖
​
(
𝑥
)
𝑃
𝑆
​
(
𝑥
)
=
𝑝
𝑖
−
𝑝
𝑖
𝐿
+
1
1
−
𝑝
𝑖
⋅
log
⁡
𝑝
𝑖
𝑝
+
(
1
−
𝑝
𝑖
𝐿
)
​
log
⁡
1
−
𝑝
𝑖
1
−
𝑝
.
		
(133)

By utilizing 
log
⁡
𝑥
≤
𝑥
−
1
 for 
𝑥
>
0
,

	
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
)
≤
𝑝
𝑖
−
𝑝
𝑖
𝐿
+
1
1
−
𝑝
𝑖
⋅
𝑝
𝑖
−
𝑝
𝑝
+
(
1
−
𝑝
𝑖
𝐿
)
​
𝑝
−
𝑝
𝑖
1
−
𝑝
=
(
1
−
𝑝
𝑖
𝐿
)
​
(
𝑝
𝑖
−
𝑝
)
2
𝑝
​
(
1
−
𝑝
𝑖
)
​
(
1
−
𝑝
)
		
(134)

Combining (134) and (131),

	
KL
​
(
𝑃
𝑆
𝑖
,
𝑃
𝑆
)
≤
(
𝜇
𝑆
−
𝜇
𝑖
)
2
​
(
1
−
𝑝
𝑖
)
​
(
1
−
𝑝
)
​
(
1
−
𝑝
𝑖
𝐿
)
𝑝
​
(
1
−
𝑝
𝐿
)
2
≤
(
𝜇
𝑆
−
𝜇
𝑖
)
2
𝑝
​
(
1
−
𝑝
𝐿
)
/
(
1
−
𝑝
)
.
		
(135)

According to the definition of 
kl
𝑖
 in Theorem 4.4,

	
kl
𝑖
≤
(
𝜇
𝑖
∗
−
𝜇
𝑖
)
2
𝑝
𝑖
∗
​
(
1
−
𝑝
𝑖
∗
𝐿
)
/
(
1
−
𝑝
𝑖
∗
)
=
Δ
𝑖
2
𝑝
𝑖
∗
​
(
1
−
𝑝
𝑖
∗
𝐿
)
/
(
1
−
𝑝
𝑖
∗
)
.
		
(136)

Thus, the regret is lower bounded by

	
lim inf
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
𝑚
,
𝜈
)
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
∑
𝑖
≠
𝑖
∗
1
𝜇
𝑖
∗
​
Δ
𝑖
⋅
𝑝
𝑖
∗
​
(
1
−
𝑝
𝑖
∗
𝐿
)
(
1
−
𝑝
𝑖
∗
)
.
		
(137)

Furthermore, if 
𝑝
𝑖
∗
∈
(
2
−
1
/
𝐿
,
1
)
,

	
lim inf
𝑚
→
∞
Reg
​
(
𝙰𝙻𝙶
,
pt
𝑚
,
𝜈
)
log
⁡
(
len
​
(
pt
𝜏
c
𝑚
)
)
≥
∑
𝑖
≠
𝑖
∗
𝐿
/
2
𝜇
𝑖
∗
​
Δ
𝑖
.
		
(138)

∎

Appendix FSupporting Lemmas
Lemma F.1 (Doob’s optional stopping, Theorem 3.8 in lattimore2020bandit).

Let 
𝔽
=
(
ℱ
𝑡
)
𝑡
∈
ℕ
 be a filtration and 
(
𝑋
𝑡
)
𝑡
∈
ℕ
 be an 
𝔽
-adapted martingale and 
𝜏
 an 
𝔽
-stopping time such that at least one of the following holds:

(a) 

There exists an 
𝑛
∈
ℕ
 such that 
ℙ
​
[
𝜏
>
𝑛
]
=
0
.

(b) 

𝔼
​
[
𝜏
]
<
∞
, and there exits a constant 
𝑐
∈
ℝ
 such that for all 
𝑡
∈
ℕ
, 
𝔼
​
[
|
𝑋
𝑡
+
1
−
𝑋
𝑡
|
∣
ℱ
𝑡
]
≤
𝑐
 almost surely on the event that 
𝜏
>
𝑡
.

(c) 

There exists a constant 
𝑐
 such that 
|
𝑋
𝑡
∧
𝜏
|
≤
𝑐
 almost surely for all 
𝑡
∈
ℕ
.

Then 
𝑋
𝜏
 is almost surely well-defined, and 
𝔼
​
[
𝑋
𝜏
]
=
𝔼
​
[
𝑋
0
]
. Furthermore, when 
(
𝑋
𝑡
)
 is a super/sub-martingale rather than a martingale, then equality is replaced with less/greater-than, respectively.

Lemma F.2 (Chernoff-Hoeffding bound, Fact 1 in auer2002finite).

Let 
𝑋
1
,
…
,
𝑋
𝑛
 be random variables with common range 
[
0
,
1
]
 and 
𝔼
​
[
𝑋
𝑛
∣
𝑋
1
,
…
,
𝑋
𝑛
−
1
]
=
𝜇
. Let 
𝑆
𝑛
=
𝑋
1
+
…
+
𝑋
𝑛
.
 Then for all 
𝑎
≥
0
,

	
ℙ
​
[
𝑆
𝑛
≥
𝑛
​
𝜇
+
𝑎
]
≤
exp
⁡
(
−
2
​
𝑎
2
𝑛
)
 and 
ℙ
​
[
𝑆
𝑛
≥
𝑛
​
𝜇
−
𝑎
]
≤
exp
⁡
(
−
2
​
𝑎
2
𝑛
)
		
(139)
Lemma F.3 (Confidence Intervals, Lemma 6 in Abbasi2011improved).

Assuming that the noise 
𝜂
𝑡
 is conditionally 
1
-sub-Gaussian. With probability at least 
1
−
𝛿
,

	
∀
𝑖
∈
{
1
,
2
,
…
,
𝐾
}
,
∀
𝑡
≥
0
,
|
𝜇
^
𝑖
,
𝑡
−
𝜇
𝑖
|
≤
(
1
+
𝑛
𝑖
,
𝑡
)
𝑛
𝑖
,
𝑡
2
​
(
1
+
2
​
log
⁡
(
𝐾
​
(
1
+
𝑛
𝑖
,
𝑡
)
1
/
2
𝛿
)
)
.
		
(140)
Lemma F.4 (Lemma 8 in antos2010active).

Let 
𝑎
>
0
. For any 
𝑡
≥
(
2
/
𝑎
)
​
[
log
⁡
(
1
/
𝑎
)
−
𝑏
]
+
,
𝑎
​
𝑡
+
𝑏
>
log
⁡
𝑡
.

Lemma F.5 (Exercise 3.7 in lattimore2020bandit).

Let 
𝔽
=
(
𝐹
𝑡
)
𝑡
∈
ℕ
 be a filtration, and 
𝜏
 be a stopping time with respect to 
𝔽
. Then 
ℱ
𝜏
 is a 
𝜎
-algebra.

Lemma F.6 (Bretagnolle-Huber inequality, Theorem 14.2 in lattimore2020bandit).

Let 
𝑃
 and 
𝑄
 be probability measures on the same measurable space 
(
Ω
,
ℱ
)
, and let 
𝐴
∈
ℱ
 be an arbitrary event. Then

	
𝑃
​
(
𝐴
)
+
𝑄
​
(
𝐴
𝑐
)
≥
1
2
​
exp
⁡
(
KL
​
(
𝑃
,
𝑄
)
)
,
		
(141)

where 
𝐴
𝑐
=
Ω
∖
𝐴
 is the complement of 
𝐴
.

Lemma F.7 (Pinsker’s inequality, Equation (14.12) in lattimore2020bandit).

For measures 
𝑃
 and 
𝑄
 on the same probability space 
(
Ω
,
ℱ
)
 that

	
𝑑
𝑇
​
𝑉
​
(
𝑃
,
𝑄
)
≤
1
2
​
KL
​
(
𝑃
,
𝑄
)
.
		
(142)
Lemma F.8 (Theorem 26.13 in lattimore2020bandit).

Let 
𝜂
>
0
 and 
𝑓
 be Legendre and twice differentiable with positive definite Hessian in 
𝐴
=
int
​
(
dom
​
(
𝑓
)
)
. Then for all 
𝑥
,
𝑦
∈
𝐴
, there exists a 
𝑧
∈
[
𝑥
,
𝑦
]
=
{
(
1
−
𝛼
)
​
𝑥
+
𝛼
​
𝑦
:
𝛼
∈
[
0
,
1
]
}
 such that

	
⟨
𝑥
−
𝑦
,
𝑢
⟩
−
𝐷
𝑓
​
(
𝑥
,
𝑦
)
𝜂
≤
𝜂
2
​
‖
𝑢
‖
∇
2
𝑓
​
(
𝑧
)
−
1
2
.
		
(143)
Appendix GAdditional Experimental Results
G.1Additional Experimental Details

For the experiments stated in Section 6.1, we report the memory utilization in this section. As Ealge-2 (li2024eagle2) is one of the best speculative decoding methods, we adopt it as the baseline. The Normalized Memory (NM) and Normalized Memory Bandwidth (NMB) are presented in Table 2. The result shows that the proposed methods do not incur additional memory consumption compared to the baseline method.

We further remark that this result is achieved by our superior algorithm design, where several non-parametric models (PLD, REST, Suffix Tree) to enhance a parametric SOTA model (Eagle-2). Specifically,

• 

“Non-parametric” means that these methods do not have any parameters in GPU, and directly predict the future tokens based on the past tokens according to the data structures like Trie Tree, which are Python objects and stored in CPU RAM. All these show that the storage of the draft models will not increase the GPU memory. Our model only requires approximately an additional 100MB of CPU RAM. Since CPU memory is typically much larger (1TB in our server) and cheaper than GPU memory (40 GB in our server), this cost is negligible.

• 

All the draft models share the same verifier model, which is the target model (LLaMA3-8B-Instruct (grattafiori2024llama3herdmodels) and Qwen2-7B-Instruct (yang2024qwen2technicalreport) in our experiments). So that the storage of the verifier does not increase the GPU memory.

• 

The reduction in memory usage comes from the fact that non-parametric models require fewer verification tokens (e.g., 40 for Suffix Tree) compared to the baseline Eagle-2 (e.g., 64). As a result, when invoking these models, a slight decrease in activation memory usage may be observed. Additionally, slight differences in GPU memory may be observed, arising randomly from the short-lived activation tensors rather than from the method itself.

We note that the size of SpecBench is not large enough, i.e., the number of arm pulls is not large, to derive a statistically sound result. We enable Mixture-of-Agent (wang2024mixture) on the prompts whose responses are shorter than 
100
 tokens to increase the number of arm pulls.

Table 2:The memory and memory bandwidth utilized by our method. As Eagle-2 is one of the best SD methods, we adopt it as the baseline to normalize the results of other methods. NM=Normalized Memory and NMB=Normalized Memory Bandwidth.
Methods	Spec Bench	Alpaca	Code Editor	Debug Bench
NM	NMB	NM	NMB	NM	NMB	NM	NMB
LLaMA3-8B-Instruct	
Eagle-2	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
EXP3Spec	0.9981	1.0171	0.9950	1.0170	1.0200	0.9980	1.0100	0.9960
UCBSpec	1.0059	1.0093	1.0130	1.0090	0.9990	0.9820	1.0150	1.0020
Qwen2-7B-Instruct	
Eagle-2	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
EXP3Spec	1.0043	1.0095	1.0050	0.9980	1.0400	0.9850	0.9890	0.9960
UCBSpec	0.9929	1.0036	1.0080	0.9900	1.0270	0.9930	1.0320	0.9950
G.2Experiments on Larger Models

In addition to the two models in the main paper, we conduct an additional set of experiments on a larger target model, namely LLaMA-2-13B (Touvron2023Llama2O). As Table 1 indicates, Eagle-2 (li2024eagle2) is one of the best speculative decoding methods, we adopt it as the baseline. The other setups are the same as the ones in Section 6.1.

From the result reported in Table 3, the proposed methods, UCBSpec and EXP3Spec, demonstrate their efficacy on larger models.

Table 3:Empirical Comparison between the proposed algorithms and Eagle-2 (li2024eagle2) with LLaMA-2-13B as the target model, measured by Mean Accepted Tokens (MAT) (
↑
) and Tokens/s (
↑
). The best result is highlighted in bold, while the second best result is underlined. The proposed algorithms remain effective on larger models.
Methods	Spec Bench	Alpaca	Code Editor	Debug Bench
MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)	MAT(
↑
)	Tokens/s(
↑
)
LLaMA-2-13B			
Eagle-2	4.35	91.94	4.32	96.59	5.19	107.57	5.16	108.45
EXP3Spec	4.05	95.52	4.32	99.64	5.22	115.65	5.03	116.65
UCBSpec	4.43	97.16	4.36	102.29	5.27	113.97	5.27	118.67
G.3Experiments on Different Hardwares

In the main paper, the experiments are conducted on a single A100 GPU. In this section, we conduct an additional set of experiments on GeForce RTX 4090. We adopt Eagle-2 (li2024eagle2) as the baseline and Spec Bench (xia2024unlocking) as the benchmark. The result is presented in Table 4. We observe a similar trend as the result presented in Table 1. The proposed method remains useful with a different hardware setup.

Table 4:Empirical comparison between Eagle-2 and the proposed algorithms on GeForce RTX 4090. We observe a similar trend as the result presented in Table 1.
Methods	Spec Bench	
MAT	Tokens/s	
LLaMA3-8B-Instruct	
Eagle-2	4.14	97.01	
EXP3Spec	3.95	102.24	
UCBSpec	4.16	107.38	
Qwen2-7B-Instruct	
Eagle-2	3.65	94.16	
EXP3Spec	3.96	111.74	
UCBSpec	4.17	112.21	
Report Issue
Report Issue for Selection
Generated by L A T E xml 
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button.
Open a report feedback form via keyboard, use "Ctrl + ?".
Make a text selection and click the "Report Issue for Selection" button near your cursor.
You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

