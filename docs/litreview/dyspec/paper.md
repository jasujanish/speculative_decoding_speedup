Title: Faster Speculative Decoding with Dynamic Token Tree Structure

URL Source: https://arxiv.org/html/2410.11744

Published Time: Wed, 16 Oct 2024 01:08:12 GMT

Markdown Content:
Yunfan Xiong 

Peking University 

yunfan.xiong@stu.pku.edu.cn

&Ruoyu Zhang 

Peking University 

ry_zhang@pku.edu.cn

&Yanzeng Li 

Peking University 

liyanzeng@stu.pku.edu.cn

&Tianhao Wu 

University of California, Berkeley 

thw@berkeley.edu

&Lei Zou 

Peking University 

zoulei@pku.edu.cn

###### Abstract

While speculative decoding has recently appeared as a promising direction for accelerating the inference of large language models (LLMs), the speedup and scalability are strongly bounded by the token acceptance rate. Prevalent methods usually organize predicted tokens as independent chains or fixed token trees, which fails to generalize to diverse query distributions. In this paper, we propose DySpec, a faster speculative decoding algorithm with a novel dynamic token tree structure. We begin by bridging the draft distribution and acceptance rate from intuitive and empirical clues, and successfully show that the two variables are strongly correlated. Based on this, we employ a greedy strategy to dynamically expand the token tree at run time. Theoretically, we show that our method can achieve optimal results under mild assumptions. Empirically, DySpec yields a higher acceptance rate and speedup than fixed trees. DySpec can drastically improve the throughput and reduce the latency of token generation across various data distribution and model sizes, which significantly outperforms strong competitors, including Specinfer and Sequoia. Under low temperature setting, DySpec can improve the throughput up to 9.1×\times× and reduce the latency up to 9.4×\times× on Llama2-70B. Under high temperature setting, DySpec can also improve the throughput up to 6.21×\times×, despite the increasing difficulty of speculating more than one token per step for draft model.

1 Introduction
--------------

Recent years have witnessed the prosperity of large language models (LLMs), shown by their unprecedented capabilities in understanding and generating human languages in various domains and tasks (OpenAI, [2023](https://arxiv.org/html/2410.11744v1#bib.bib12); Anthropic, [2024](https://arxiv.org/html/2410.11744v1#bib.bib1)). Despite this rapid progress, the major bottleneck in the real-world deployment of LLMs stems from their inference latency, due to the nature of auto-regressive decoding. Generating n 𝑛 n italic_n tokens requires n 𝑛 n italic_n sequential runs, making the process time-consuming and leading to under-utilizing available computation resources.

To address this challenge, recent works (Chen et al., [2023](https://arxiv.org/html/2410.11744v1#bib.bib3); Leviathan et al., [2023](https://arxiv.org/html/2410.11744v1#bib.bib8)) have proposed speculative decoding to accelerate the inference. Speculative decoding first leverages a draft model to sample a bunch of tokens as candidates, which are later verified in parallel by the target model. If the verification of a token fails, its succeeding tokens must all be rejected to ensure output distribution is unbiased. Therefore, the performance of speculative decoding is strongly bounded by the acceptance rate of predicted tokens.

![Image 1: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/structure-2.png)

(a) Chain.

![Image 2: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/structure-3.png)

(b) k 𝑘 k italic_k sequences.

![Image 3: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/structure-4.png)

(c) Tree.

Figure 1: Different structures of predicted tokens. SpecTr is [1(b)](https://arxiv.org/html/2410.11744v1#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") structure, while Specinfer, Medusa and Sequoia are [1(c)](https://arxiv.org/html/2410.11744v1#S1.F1.sf3 "In Figure 1 ‣ 1 Introduction ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") structure. 

To this end, several methods have explored tree structures to enhance the acceptance rate, as illustrated in Figure[1](https://arxiv.org/html/2410.11744v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"). For instance, Sun et al. ([2024](https://arxiv.org/html/2410.11744v1#bib.bib17)) developed SpecTr, introducing DraftSelection algorithm to make draft model select multiple candidates while maintaining the same output distribution as the target model. Miao et al. ([2023](https://arxiv.org/html/2410.11744v1#bib.bib10)) created SpecInfer, which constructs token trees using small speculative models with learnable branch numbers of each layer. Similarly, Cai et al. ([2024](https://arxiv.org/html/2410.11744v1#bib.bib2)) proposed Medusa, which bases token tree construction directly on draft model probabilities, optimizing efficiency when the draft model closely approximates the target model. Meanwhile, Chen et al. ([2024](https://arxiv.org/html/2410.11744v1#bib.bib4)) introduced Sequoia, which estimates acceptance rates for candidate tokens and uses dynamic programming to optimize the token tree based on the estimated metric. However, a common limitation of these methods is their reliance on fixed patterns of tree construction, which can lead to suboptimal performance across diverse query distributions, resulting in a relatively low acceptance rate as tree size grows. This raises an important research question:

RQ 1: How can we find a near-optimal token tree structure for speculative decoding? To answer the research question, we will first establish the connection between acceptance rate and draft distribution through the following hypothesis.

###### Hypothesis 1.

Predicted tokens of higher draft probability statistically have a higher acceptance rate.

Fortunately, this is further validated by our preliminary studies, as demonstrated in Figure [2](https://arxiv.org/html/2410.11744v1#S3.F2 "Figure 2 ‣ 3 Bridging Draft Distribution with Acceptance Rate ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"). With the observation, we propose DySpec to dynamically expand the token tree based on draft distribution. DySpec employs a greedy search strategy to maximize the expected length of the predicted sequences. Compared with its fixed counterpart, the dynamic token tree yields a higher acceptance rate and speedup. We conduct benchmarking experiments on various datasets and different model scales, the experimental results demonstrate our proposed DySpec can efficiently improve the inference performance. Specifically, on the Llama2-70B model, DySpec achieves a 9.1×\times× throughput improvement and 9.4×\times× reduction in latency.

2 Preliminary
-------------

#### Speculative Decoding.

Chen et al. ([2023](https://arxiv.org/html/2410.11744v1#bib.bib3)) and Leviathan et al. ([2023](https://arxiv.org/html/2410.11744v1#bib.bib8)) proposed speculative decoding as a means to accelerate auto-regressive decoding. This approach samples generations from an efficient draft model as speculative prefixes and verifies these tokens in parallel using a slower target model. Through rejection sampling, it ensures that the outputs have the same distribution as those from the target model alone.

We denote the distribution of the draft model as D⁢[⋅]𝐷 delimited-[]⋅D[\cdot]italic_D [ ⋅ ]1 1 1 We use D⁢[⋅]𝐷 delimited-[]⋅D[\cdot]italic_D [ ⋅ ] as an abbreviation of conditional probability D⁢(x t|x<t)𝐷 conditional subscript 𝑥 𝑡 subscript 𝑥 absent 𝑡 D(x_{t}|x_{<t})italic_D ( italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_x start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT ), and similarly for T⁢[⋅]𝑇 delimited-[]⋅T[\cdot]italic_T [ ⋅ ]., and the target distribution as T⁢[⋅]𝑇 delimited-[]⋅T[\cdot]italic_T [ ⋅ ]. In speculative decoding, a token x 𝑥 x italic_x sampled from D 𝐷 D italic_D is accepted with a probability of min⁡(1,T⁢[x]D⁢[x])1 𝑇 delimited-[]𝑥 𝐷 delimited-[]𝑥\min(1,\frac{T[x]}{D[x]})roman_min ( 1 , divide start_ARG italic_T [ italic_x ] end_ARG start_ARG italic_D [ italic_x ] end_ARG ). In case of rejection, another token y 𝑦 y italic_y will be sampled from a residual distribution norm⁢(relu⁢(T−D))norm relu 𝑇 𝐷\texttt{norm}(\texttt{relu}(T-D))norm ( relu ( italic_T - italic_D ) ) to adjust the output aligned with the target distribution.

#### Tree Attention.

Transformer(Vaswani et al., [2017](https://arxiv.org/html/2410.11744v1#bib.bib20)) models use the attention mechanism to aggregate sequential information. In implementation, the auto-regressive model uses an upper triangle mask to preserve causality. In the context of tree-based dependency, Liu et al. ([2020](https://arxiv.org/html/2410.11744v1#bib.bib9)) first proposed tree attention to represent the hierarchy as:

mask⁢(A)i,j={1,i is ancestor of j,0,otherwise.\texttt{mask}(A)_{i,j}=\left\{\begin{array}[]{ll}1&,\ \ i\texttt{ is ancestor % of }j,\\ 0&,\texttt{ otherwise.}\\ \end{array}\right.mask ( italic_A ) start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT = { start_ARRAY start_ROW start_CELL 1 end_CELL start_CELL , italic_i is ancestor of italic_j , end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL , otherwise. end_CELL end_ROW end_ARRAY

In speculative decoding, tree attention has later been adopted by SpecInfer(Miao et al., [2023](https://arxiv.org/html/2410.11744v1#bib.bib10)) and Medusa(Cai et al., [2024](https://arxiv.org/html/2410.11744v1#bib.bib2)) for parallel verification.

3 Bridging Draft Distribution with Acceptance Rate
--------------------------------------------------

![Image 4: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/heatmap2.png)

Figure 2: Connection between acceptance rate/target distribution and draft distribution on CNN DailyMail.The density of each block is normalized by column.

During verification, the acceptance probability of sampled token x 𝑥 x italic_x is given by min⁡(1,T⁢[x]D⁢[x])1 𝑇 delimited-[]𝑥 𝐷 delimited-[]𝑥\min(1,\frac{T[x]}{D[x]})roman_min ( 1 , divide start_ARG italic_T [ italic_x ] end_ARG start_ARG italic_D [ italic_x ] end_ARG ). We now derive the connection between draft distribution and acceptance rate as follows.

Since the draft distribution acts as the approximation of the target distribution, the two distributions should not be too ”far” away. Without loss of generality, we assume that the KL divergence of D 𝐷 D italic_D from T 𝑇 T italic_T is constrained by constant c 𝑐 c italic_c, i.e.,

D KL⁢(D∥T)=∑D⁢[x]⁢log⁡D⁢[x]T⁢[x]≤c.subscript 𝐷 KL conditional 𝐷 𝑇 𝐷 delimited-[]𝑥 𝐷 delimited-[]𝑥 𝑇 delimited-[]𝑥 𝑐 D_{\mathrm{KL}}(D\parallel T)=\sum D[x]\log\frac{D[x]}{T[x]}\leq c.italic_D start_POSTSUBSCRIPT roman_KL end_POSTSUBSCRIPT ( italic_D ∥ italic_T ) = ∑ italic_D [ italic_x ] roman_log divide start_ARG italic_D [ italic_x ] end_ARG start_ARG italic_T [ italic_x ] end_ARG ≤ italic_c .(1)

To satisfy the constraint, T⁢[⋅]𝑇 delimited-[]⋅T[\cdot]italic_T [ ⋅ ] should not diverge much from D⁢[⋅]𝐷 delimited-[]⋅D[\cdot]italic_D [ ⋅ ]. Nevertheless, for a token x 𝑥 x italic_x with large draft probability D⁢[x]𝐷 delimited-[]𝑥 D[x]italic_D [ italic_x ], T⁢[x]D⁢[x]𝑇 delimited-[]𝑥 𝐷 delimited-[]𝑥\frac{T[x]}{D[x]}divide start_ARG italic_T [ italic_x ] end_ARG start_ARG italic_D [ italic_x ] end_ARG cannot be too small, as it would contribute significantly to D KL subscript 𝐷 KL D_{\mathrm{KL}}italic_D start_POSTSUBSCRIPT roman_KL end_POSTSUBSCRIPT. On the other hand, tokens with small D⁢[x]𝐷 delimited-[]𝑥 D[x]italic_D [ italic_x ] have less impact to D KL subscript 𝐷 KL D_{\mathrm{KL}}italic_D start_POSTSUBSCRIPT roman_KL end_POSTSUBSCRIPT, allowing for greater variation. The above analysis implies that predicted tokens of higher draft probability statistically have a higher target probability and acceptance rate.

We further validate our hypothesis through preliminary experiments. As shown in Figure[2](https://arxiv.org/html/2410.11744v1#S3.F2 "Figure 2 ‣ 3 Bridging Draft Distribution with Acceptance Rate ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") (right), the draft distribution shows a strong correlation with the target distribution in real-world scenarios. More importantly, Figure[2](https://arxiv.org/html/2410.11744v1#S3.F2 "Figure 2 ‣ 3 Bridging Draft Distribution with Acceptance Rate ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") (left) demonstrates that the distributions of acceptance rate, under the same draft probability, resemble binomial distributions. As draft probability grows larger, predicted tokens are more likely to be accepted. These observations provide strong empirical support for our previous claim. It also inspires us to design a dynamic token tree construction algorithm to explore more on sub-trees of higher draft probability, since they are more likely to be accepted in later verification.

4 Method
--------

Under a fixed speculative budget b 𝑏 b italic_b (i.e. the number of tokens for each verification), the optimal token tree yields the highest acceptance rate. In practice, finding the optimal tree is unfeasible, since the target distribution is unknown before verification. Nevertheless, given Hypothesis [1](https://arxiv.org/html/2410.11744v1#Thmhypothesis1 "Hypothesis 1. ‣ 1 Introduction ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"), we can transform the original problem into the following problems.

### 4.1 Dynamic Token Tree Construction

Given the speculative token tree, the way we sampling this tree, the draft model output distribution, and correspond target model output distribution, we can get the expectation of the total number of Speculative decoding verification. Considering each node t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in speculative token tree independently, we denote its draft distribution as p d⁢[i,⋅]subscript 𝑝 𝑑 𝑖⋅p_{d}[i,\cdot]italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT [ italic_i , ⋅ ], and the relevant target distribution as p t⁢[i,⋅]subscript 𝑝 𝑡 𝑖⋅p_{t}[i,\cdot]italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT [ italic_i , ⋅ ].

Assume that node t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT have ancestors a 1,…,a i subscript 𝑎 1…subscript 𝑎 𝑖 a_{1},...,a_{i}italic_a start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, and previous sibling node s 1,…,s j subscript 𝑠 1…subscript 𝑠 𝑗 s_{1},...,s_{j}italic_s start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT, then the probability we verify the node t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT can be represent as ∏i P⁢[a⁢c⁢c⁢e⁢p⁢t⁢a i]×∏j P⁢[r⁢e⁢j⁢e⁢c⁢t⁢s j]subscript product 𝑖 𝑃 delimited-[]𝑎 𝑐 𝑐 𝑒 𝑝 𝑡 subscript 𝑎 𝑖 subscript product 𝑗 𝑃 delimited-[]𝑟 𝑒 𝑗 𝑒 𝑐 𝑡 subscript 𝑠 𝑗\prod_{i}P[accepta_{i}]\times\prod_{j}P[rejects_{j}]∏ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_P [ italic_a italic_c italic_c italic_e italic_p italic_t italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_P [ italic_r italic_e italic_j italic_e italic_c italic_t italic_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ].

In Speculative Decoding, the probability we accept token x 𝑥 x italic_x with draft probability p d⁢[x]subscript 𝑝 𝑑 delimited-[]𝑥 p_{d}[x]italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT [ italic_x ] and target probability p t⁢[x]subscript 𝑝 𝑡 delimited-[]𝑥 p_{t}[x]italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT [ italic_x ], is min⁡(1,p t⁢[x]p d⁢[x])1 subscript 𝑝 𝑡 delimited-[]𝑥 subscript 𝑝 𝑑 delimited-[]𝑥\min(1,\frac{p_{t}[x]}{p_{d}[x]})roman_min ( 1 , divide start_ARG italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT [ italic_x ] end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT [ italic_x ] end_ARG ), denote as S⁢D⁢[x]𝑆 𝐷 delimited-[]𝑥 SD[x]italic_S italic_D [ italic_x ]. So the probability we take verification on node t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is ∏i S⁢D⁢[a i]×∏j(1−S⁢D⁢[s j])subscript product 𝑖 𝑆 𝐷 delimited-[]subscript 𝑎 𝑖 subscript product 𝑗 1 𝑆 𝐷 delimited-[]subscript 𝑠 𝑗\prod_{i}SD[a_{i}]\times\prod_{j}(1-SD[s_{j}])∏ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_S italic_D [ italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( 1 - italic_S italic_D [ italic_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] ). Then the contribution of node t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT to expectation of total accepted token number is ∏i S⁢D⁢[a i]×∏j(1−S⁢D⁢[s j])×S⁢D⁢[t i]subscript product 𝑖 𝑆 𝐷 delimited-[]subscript 𝑎 𝑖 subscript product 𝑗 1 𝑆 𝐷 delimited-[]subscript 𝑠 𝑗 𝑆 𝐷 delimited-[]subscript 𝑡 𝑖\prod_{i}SD[a_{i}]\times\prod_{j}(1-SD[s_{j}])\times SD[t_{i}]∏ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_S italic_D [ italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( 1 - italic_S italic_D [ italic_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] ) × italic_S italic_D [ italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ].

The total expectation of accepted token number of this speculative token tree is

∑u∏i S⁢D⁢[a i,t u]×∏j(1−S⁢D⁢[s j,t u])×S⁢D⁢[t u]subscript 𝑢 subscript product 𝑖 𝑆 𝐷 delimited-[]subscript 𝑎 𝑖 subscript 𝑡 𝑢 subscript product 𝑗 1 𝑆 𝐷 delimited-[]subscript 𝑠 𝑗 subscript 𝑡 𝑢 𝑆 𝐷 delimited-[]subscript 𝑡 𝑢\sum_{u}\prod_{i}SD[a_{i,t_{u}}]\times\prod_{j}(1-SD[s_{j,t_{u}}])\times SD[t_% {u}]∑ start_POSTSUBSCRIPT italic_u end_POSTSUBSCRIPT ∏ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_S italic_D [ italic_a start_POSTSUBSCRIPT italic_i , italic_t start_POSTSUBSCRIPT italic_u end_POSTSUBSCRIPT end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( 1 - italic_S italic_D [ italic_s start_POSTSUBSCRIPT italic_j , italic_t start_POSTSUBSCRIPT italic_u end_POSTSUBSCRIPT end_POSTSUBSCRIPT ] ) × italic_S italic_D [ italic_t start_POSTSUBSCRIPT italic_u end_POSTSUBSCRIPT ](2)

With expected acceptance rate, we can construct the optimal speculative token tree. However, there are still two problems:

1.   1.When we generate speculative token tree, we cannot know the target probability to get S⁢D⁢[⋅]𝑆 𝐷 delimited-[]⋅SD[\cdot]italic_S italic_D [ ⋅ ]. 
2.   2.The draft token t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is sampled from draft output distribution, we could only decide how many sampling we take, instead of which token to take. Otherwise the take action we made will infect the probability we keep tokens in speculative token tree. 

To solve problem 1, we note that the acceptance rate is positive-related to draft output distribution. Given Hypothesis [1](https://arxiv.org/html/2410.11744v1#Thmhypothesis1 "Hypothesis 1. ‣ 1 Introduction ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"), we use draft model output distribution to estimate the acceptance rate S⁢D⁢[t i]≈p d⁢[t i]𝑆 𝐷 delimited-[]subscript 𝑡 𝑖 subscript 𝑝 𝑑 delimited-[]subscript 𝑡 𝑖 SD[t_{i}]\approx p_{d}[t_{i}]italic_S italic_D [ italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] ≈ italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT [ italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ].

To solve problem 2, we only use these estimated values to decide if we will make the sampling. For given intermediate token tree status, we can detect all expandable tree nodes, and pick the expandable tree node with maximum estimated value. Repeat this action until we reach the max tree size, DySpec will generate the optimal speculative token tree. The proof of optimality is provided in Appendix[D](https://arxiv.org/html/2410.11744v1#A4 "Appendix D prove ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

Now we can get the algorithm to generate the optimal speculative token tree.

### 4.2 Algorithm

Given the prompt, DySpec can get the logits of the last token, which is the root of the speculative token tree. Suppose we have already constructed a partial speculative token tree as Figure[3](https://arxiv.org/html/2410.11744v1#S4.F3 "Figure 3 ‣ 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"). There are two ways to expand a node:

1.   1.Any token without a leaf node can undergo the first sampling. 
2.   2.Nodes marked with ”–/–” indicate that we have already performed several samplings at the same position and have obtained an estimated value for the next sampling at this position (on the arrow line). The ”–/–” node corresponds to the result of the next sampling. 

We refer to these two types of nodes as expandable nodes in the current state.

![Image 5: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/tree_bold.png)

Figure 3: An example of the predicted token tree.

DySpec use a heap to maintain all the expandable tokens by their estimated values, that we can get the node with maximum estimated value in O⁢(l⁢o⁢g⁢N)𝑂 𝑙 𝑜 𝑔 𝑁 O(logN)italic_O ( italic_l italic_o italic_g italic_N ) time. After we make the next sampling represented by the top node of the heap. Upon determining the result of the sampling, we then update the state of the current token tree using the obtained token and its corresponding estimated value. This process generates two new expandable nodes:

1.   1.When the current node is rejected, the next sampling at the same position, with the corresponding estimated value being the probability of this sampling failure multiplied by the expected acceptance rate of the next sampling itself. 
2.   2.When the current node is accepted, proceeding with subsequent sampling, with the corresponding estimated value being the probability of this sampling success multiplied by the expected acceptance rate of the next sampling itself. 

Thus, we have successfully expanded the token tree by one node. This process is repeated until the predetermined budget is reached. The pseudo-code is presented in Algorithm [1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

Input :Prefix

x 0 subscript 𝑥 0 x_{0}italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
, draft model

D Θ(⋅|x)D_{\Theta}(\cdot|x)italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x )
, and an upper bound of guess tokens number

m 𝑚 m italic_m
.

Output :generated token tree

T⁢r 𝑇 𝑟 Tr italic_T italic_r
.

1 Initialize a heap

H 𝐻 H italic_H
, Heap Element consists of tree information

TreeInfo i subscript TreeInfo 𝑖\texttt{TreeInfo}_{i}TreeInfo start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
, residual distribution

R i subscript 𝑅 𝑖 R_{i}italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
, estimate acceptance rate

v 𝑣 v italic_v
.

2

R←D Θ(⋅|x 0),v←1,TreeInfo←…R\leftarrow D_{\Theta}(\cdot|x_{0}),v\leftarrow 1,\texttt{TreeInfo}\leftarrow\dots italic_R ← italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) , italic_v ← 1 , TreeInfo ← …

3

H.p⁢u⁢s⁢h⁢(R,v,TreeInfo)formulae-sequence 𝐻 𝑝 𝑢 𝑠 ℎ 𝑅 𝑣 TreeInfo H.push(R,v,\texttt{TreeInfo})italic_H . italic_p italic_u italic_s italic_h ( italic_R , italic_v , TreeInfo )
;

4 while _Tr.size <<< m_ do

5

R,v,TreeInfo←H.p⁢o⁢p⁢()formulae-sequence←𝑅 𝑣 TreeInfo 𝐻 𝑝 𝑜 𝑝 R,v,\texttt{TreeInfo}\leftarrow H.pop()italic_R , italic_v , TreeInfo ← italic_H . italic_p italic_o italic_p ( )
;

6

NewNodeInfo←←NewNodeInfo absent\texttt{NewNodeInfo}\leftarrow NewNodeInfo ←
Tr.add(TreeInfo, y);

7 sample

y∼R similar-to 𝑦 𝑅 y\sim R italic_y ∼ italic_R
;

8

v 0=v×R⁢[y]subscript 𝑣 0 𝑣 𝑅 delimited-[]𝑦 v_{0}=v\times R[y]italic_v start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = italic_v × italic_R [ italic_y ]
;

9

v 1=v×(1−R⁢[y])subscript 𝑣 1 𝑣 1 𝑅 delimited-[]𝑦 v_{1}=v\times(1-R[y])italic_v start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = italic_v × ( 1 - italic_R [ italic_y ] )
;

10

R⁢[y]←0←𝑅 delimited-[]𝑦 0 R[y]\leftarrow 0 italic_R [ italic_y ] ← 0
;

11

R←n⁢o⁢r⁢m⁢(R)←𝑅 𝑛 𝑜 𝑟 𝑚 𝑅 R\leftarrow norm(R)italic_R ← italic_n italic_o italic_r italic_m ( italic_R )
;

H.p⁢u⁢s⁢h⁢(R,v 1,TreeInfo)formulae-sequence 𝐻 𝑝 𝑢 𝑠 ℎ 𝑅 subscript 𝑣 1 TreeInfo H.push(R,v_{1},\texttt{TreeInfo})italic_H . italic_p italic_u italic_s italic_h ( italic_R , italic_v start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , TreeInfo )
;

/* expand neighbor node */

12 get

x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
from TreeInfo and y;

13

d i←D Θ(⋅|x i)d_{i}\leftarrow D_{\Theta}(\cdot|x_{i})italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )
;

H.p⁢u⁢s⁢h⁢(d i,v 0,NewNodeInfo)formulae-sequence 𝐻 𝑝 𝑢 𝑠 ℎ subscript 𝑑 𝑖 subscript 𝑣 0 NewNodeInfo H.push(d_{i},v_{0},\texttt{NewNodeInfo})italic_H . italic_p italic_u italic_s italic_h ( italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_v start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , NewNodeInfo )
;

/* expand child node */

14

15 end while

Algorithm 1 Speculative token tree construction algorithm with fixed number

### 4.3 Analyze Overhead

Assume the speculative token tree size is N 𝑁 N italic_N, depth is D 𝐷 D italic_D. Greedy expand method will generate the optimal token tree one by one. For each token, greedy expand method choose the expandable token with maximum estimated valueand then make a sampling to generate the next token, then update the token tree.

To quickly choose the expandable token with maximum estimated value, we can use heap to maintain all expand-able tokens’ estimated value, which introduce O⁢(l⁢o⁢g⁢N)𝑂 𝑙 𝑜 𝑔 𝑁 O(logN)italic_O ( italic_l italic_o italic_g italic_N ) time complexity to maintain the token tree and related auxiliary structures. The total time complexity of token tree construction is O⁢(N⁢l⁢o⁢g⁢N)𝑂 𝑁 𝑙 𝑜 𝑔 𝑁 O(NlogN)italic_O ( italic_N italic_l italic_o italic_g italic_N ).

Although one step inference’s time consume of draft model is usually much lower than target model, it is still non negligible. Denote draft model inference time as T d subscript 𝑇 𝑑 T_{d}italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT, target model inference time as T t subscript 𝑇 𝑡 T_{t}italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, the total time of one step of greedy expand method is

O⁢(N⁢l⁢o⁢g⁢N+T t+N⁢T d)𝑂 𝑁 𝑙 𝑜 𝑔 𝑁 subscript 𝑇 𝑡 𝑁 subscript 𝑇 𝑑 O(NlogN+T_{t}+NT_{d})italic_O ( italic_N italic_l italic_o italic_g italic_N + italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_N italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT )(3)

With accepted token number e 𝑒 e italic_e, the latency of generate one token can be represent as O⁢((N⁢l⁢o⁢g⁢N+T t+N⁢T d)/e)𝑂 𝑁 𝑙 𝑜 𝑔 𝑁 subscript 𝑇 𝑡 𝑁 subscript 𝑇 𝑑 𝑒 O((NlogN+T_{t}+NT_{d})/e)italic_O ( ( italic_N italic_l italic_o italic_g italic_N + italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_N italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ) / italic_e ).

In the implementation, the time complexity of constructing a token tree for a single operation is O⁢(v⁢o⁢c⁢a⁢b⁢_⁢s⁢i⁢z⁢e)𝑂 𝑣 𝑜 𝑐 𝑎 𝑏 _ 𝑠 𝑖 𝑧 𝑒 O(vocab\_size)italic_O ( italic_v italic_o italic_c italic_a italic_b _ italic_s italic_i italic_z italic_e ), due to the sampling and updating of the residual distribution. Typically, the inference of a draft model involves higher time complexity. However, model inference benefits from regular computational workloads and can be efficiently accelerated by GPUs, whereas the complex logical operations involved in token tree construction suffer from low efficiency when implemented in Python. To mitigate this overhead, we implemented the token tree construction in C++, making it negligible compared to the inference times of both the target and draft models.

Even if we disregard the overhead associated with constructing the token tree, accelerating the target model still requires us to achieve a speedup factor of approximately k≈1/e+N⁢T d e⁢T t 𝑘 1 𝑒 𝑁 subscript 𝑇 𝑑 𝑒 subscript 𝑇 𝑡 k\approx 1/e+\frac{NT_{d}}{eT_{t}}italic_k ≈ 1 / italic_e + divide start_ARG italic_N italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT end_ARG start_ARG italic_e italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_ARG, where 1/k 1 𝑘 1/k 1 / italic_k represents the acceleration rate. As the number of tokens N 𝑁 N italic_N increases, the term N/e 𝑁 𝑒 N/e italic_N / italic_e grows significantly. For instance, with N=64 𝑁 64 N=64 italic_N = 64, N/e 𝑁 𝑒 N/e italic_N / italic_e typically exceeds 10 10 10 10 , and for N=768 𝑁 768 N=768 italic_N = 768, N/e 𝑁 𝑒 N/e italic_N / italic_e can surpass 70 70 70 70. This rapid growth severely limits the potential for acceleration by simply increasing the size of the token tree.

To address this limitation, we need to develop a more efficient method for generating draft tokens. It’s important to note that the token tree structure will branch out significantly after a few steps, resulting in a relatively shallow depth. If we can generate draft tokens layer by layer, the latency for generating one token can be represented as O⁢((N⁢l⁢o⁢g⁢N+T t+D⁢T d)/e)𝑂 𝑁 𝑙 𝑜 𝑔 𝑁 subscript 𝑇 𝑡 𝐷 subscript 𝑇 𝑑 𝑒 O((NlogN+T_{t}+DT_{d})/e)italic_O ( ( italic_N italic_l italic_o italic_g italic_N + italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_D italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ) / italic_e ), where the time cost of one step can be considered constant for an appropriate input size. For N=64 𝑁 64 N=64 italic_N = 64, D 𝐷 D italic_D is typically less than 10 10 10 10, and for N=768 𝑁 768 N=768 italic_N = 768, D 𝐷 D italic_D is usually less than 30 30 30 30.

However, the greedy expansion method struggles to align with layer-by-layer generation because, without revealing the estimated values of all tokens, it is challenging to determine how many tokens should be included in the shadow layers.

### 4.4 Construct Token Tree with Threshold

To accelerate inference, we must reduce the number of draft generations. In the greedy expansion method, we select the token with the highest estimated value at each step, and this value monotonically decreases with each selection. Once the token tree construction is complete, all tokens with an estimated value greater than a certain threshold C 𝐶 C italic_C are chosen, while those with lower values are discarded. If we could determine this threshold c 𝑐 c italic_c at the outset, it would be possible to construct the optimal speculative token tree layer-by-layer. In practice, we can choose an appropriate threshold C 𝐶 C italic_C (typically around 1/n 1 𝑛 1/n 1 / italic_n) and relax the constraint on N 𝑁 N italic_N. This adjustment has a minimal impact on the number of accepted tokens but significantly improves latency. The pseudo-code is provided in Appendix [A.2](https://arxiv.org/html/2410.11744v1#A1.SS2 "A.2 Token Tree Construction Algorithm with Threshold ‣ Appendix A Token Tree Construction Algorithm ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

5 Empirical Results
-------------------

### 5.1 Setup

We implement DySpec using Llama models. We employs JackFram/Llama68m (JF68m) and Llama2-7B as the draft model, and Llama2-7B, Llama2-13B, Llama2-70B(Touvron et al., [2023](https://arxiv.org/html/2410.11744v1#bib.bib19)) as the target models. We conduct evaluations on various datasets with varying sizes and characteristics, including C4(en)(Raffel et al., [2020](https://arxiv.org/html/2410.11744v1#bib.bib14)), OpenWebText(Gokaslan & Cohen, [2019](https://arxiv.org/html/2410.11744v1#bib.bib5)) and CNN DailyMail(Nallapati et al., [2016](https://arxiv.org/html/2410.11744v1#bib.bib11)).

For a fair comparison, we follow the setting in Sequoia(Chen et al., [2024](https://arxiv.org/html/2410.11744v1#bib.bib4)), using the first 128 tokens as the fixed prompt and generating 128 tokens as completion. We evaluate our method with different target temperatures and set the draft temperature to 0.6. All experiments are conducted on a computation node with one NVIDIA A100 40GB GPU and 32 CPU cores.

### 5.2 Overhead of tree construction

As analyzed in the Section [4.3](https://arxiv.org/html/2410.11744v1#S4.SS3 "4.3 Analyze Overhead ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"), the construction of the token tree introduces complex logic, which is inefficient in Python despite its time complexity of O⁢(N⁢l⁢o⁢g⁢N⁢v⁢o⁢c⁢a⁢b⁢_⁢s⁢i⁢z⁢e)𝑂 𝑁 𝑙 𝑜 𝑔 𝑁 𝑣 𝑜 𝑐 𝑎 𝑏 _ 𝑠 𝑖 𝑧 𝑒 O(NlogNvocab\_size)italic_O ( italic_N italic_l italic_o italic_g italic_N italic_v italic_o italic_c italic_a italic_b _ italic_s italic_i italic_z italic_e ). To address this, we implemented the construction in C++, making the construction time negligible. The profiling results are shown in Figure [4](https://arxiv.org/html/2410.11744v1#S5.F4 "Figure 4 ‣ 5.2 Overhead of tree construction ‣ 5 Empirical Results ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"). The additional overhead introduced by DySpec is the Tree Construction, which accounts for less than two percent of the total execution time in the Llama2-68M/Llama2-7B and Llama2-68M/Llama2-13B pairs. In the Llama2-7B/Llama2-70B pair with CPU-offloading, all components except draft and target model inference cost less than two percent of the total execution time.

Generating masks, sampling tokens, and verification consume significant time under both the Llama2-68M/Llama2-7B and Llama2-68M/Llama2-13B settings. These three components represent the common overhead of all speculative decoding methods, with the primary time spent on waiting for the completion of model execution via CUDA synchronization. In the Llama2-7B/Llama2-70B setting, CPU-offloading and waiting for model execution results overlap, which is why they are not reflected in the profiling results.

![Image 6: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/pie7B.png)

(a) Llama2-68M/Llama-7B

![Image 7: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/pie13B.png)

(b) Llama2-68M/Llama-13B

![Image 8: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/pie70B.png)

(c) Llama2-7B/Llama-70B

Figure 4: The execution times of different components during the inference process.

### 5.3 Effectiveness of Dynamic Token Tree

Table[1](https://arxiv.org/html/2410.11744v1#S5.T1 "Table 1 ‣ 5.3 Effectiveness of Dynamic Token Tree ‣ 5 Empirical Results ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") presents the experimental results, detailing the number of accepted tokens and the latency per token in second, when using JF68M as the draft model and Llama2-7B as the target model. Similarly, Table[2](https://arxiv.org/html/2410.11744v1#S5.T2 "Table 2 ‣ 5.3 Effectiveness of Dynamic Token Tree ‣ 5 Empirical Results ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") shows the corresponding results for the scenario where JF68M serves as the draft model and Llama2-13B as the target model. In both cases, the maximum draft token tree size is set to 64. For the draft model, DySpec leverages CUDA Graph to capture 129 different input lengths ranging from 128 to 258, thereby accelerating inference, much like Sequoia does.

The results indicate that DySpec consistently outperforms both Sequoia and Specinfer across various data distributions and generation temperatures, leading to a higher number of accepted tokens at each decoding step. The values in the table represent the average time taken to generate a single token in seconds, with the number of tokens accepted by the target model during a single validation in parentheses.

Table 1: latency per token. The draft model is JF68m and the target model is Llama2-7B. Guess length is 64.

Table 2: latency per token. The draft model is JF68m and the target model is Llama2-13B. Guess length is 64.

For larger target models such as Llama2-70B, we employ CPU offloading due to GPU memory constraints. We selected Llama2-7B as the draft model. Despite the time consumed for data synchronization between the CPU and GPU, the inference time for the CPU-offloaded model, with a naive implementation, is approximately 15 seconds per step. By incorporating some overlapping tricks for weight loading (adapted from Sequoia), the inference time is still around 5 seconds per step. In contrast, Llama2-7B requires only about 25 milliseconds per step, resulting in a T t subscript 𝑇 𝑡 T_{t}italic_T start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT/T d subscript 𝑇 𝑑 T_{d}italic_T start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ratio of approximately 2×10 3 2 superscript 10 3 2\times 10^{3}2 × 10 start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT. Note that DySpec did not employ CUDA Graph in this scenario due to the significant GPU memory overhead associated with capturing sequences of varying lengths. With 129 distinct sequence lengths and the memory-intensive nature of the draft model Llama2-7B, this approach would be prohibitively resource-demanding.

In this scenario, the acceleration rate is roughly equivalent to the number of accepted tokens per target model step. Set the maximum draft token tree size to 64, DySpec achieves up to a 9.1x improvement in throughput and a 9.4x reduction in latency compared to auto-regressive generation, while also outperforming state-of-the-art methods in consistency, as demonstrated in Table[3](https://arxiv.org/html/2410.11744v1#S5.T3 "Table 3 ‣ 5.3 Effectiveness of Dynamic Token Tree ‣ 5 Empirical Results ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

Table 3: latency per token. The draft model is Llama2-7B and the target model is Llama2-70B. Guess length is 64.

6 Conclusion
------------

We introduce DySpec, a faster speculative decoding algorithm that incorporates a dynamic token tree structure for sampling. Based on the connection between draft probability and acceptance rate, we apply a greedy strategy to dynamically expand the token tree to maximize the expected length of predicted generations. Empirical results reveal the efficacy and scalability of DySpec by consistent improvements in acceptance rate across various datasets and generation temperatures. Specifically, on the Llama2-70B model with temperature=0, DySpec achieves a 9.1×\times× throughput improvement and 9.4×\times× reduction in latency.

References
----------

*   Anthropic (2024) Anthropic. Introducing the next generation of claude, 2024. 
*   Cai et al. (2024) Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. _arXiv preprint arXiv:2401.10774_, 2024. 
*   Chen et al. (2023) Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. _arXiv preprint arXiv:2302.01318_, 2023. 
*   Chen et al. (2024) Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable, robust, and hardware-aware speculative decoding. _arXiv preprint arXiv:2402.12374_, 2024. 
*   Gokaslan & Cohen (2019) Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. [http://Skylion007.github.io/OpenWebTextCorpus](http://skylion007.github.io/OpenWebTextCorpus), 2019. 
*   He et al. (2023) Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He. Rest: Retrieval-based speculative decoding. _arXiv preprint arXiv:2311.08252_, 2023. 
*   Lefaudeux et al. (2022) Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. [https://github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers), 2022. 
*   Leviathan et al. (2023) Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning_, pp. 19274–19286. PMLR, 2023. 
*   Liu et al. (2020) Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping Wang. K-bert: Enabling language representation with knowledge graph. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 34, pp. 2901–2908, 2020. 
*   Miao et al. (2023) Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae Ying Yee Wong, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. Specinfer: Accelerating generative llm serving with speculative inference and token tree verification. _arXiv preprint arXiv:2305.09781_, 1(2):4, 2023. 
*   Nallapati et al. (2016) Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. _arXiv preprint arXiv:1602.06023_, 2016. 
*   OpenAI (2023) OpenAI. Gpt-4 technical report, 2023. 
*   Paszke et al. (2017) Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 
*   Raffel et al. (2020) Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_, 21(140):1–67, 2020. 
*   Rasley et al. (2020) Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, KDD ’20, pp. 3505–3506, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3406703. URL [https://doi.org/10.1145/3394486.3406703](https://doi.org/10.1145/3394486.3406703). 
*   Sleator & Tarjan (1981) Daniel D Sleator and Robert Endre Tarjan. A data structure for dynamic trees. In _Proceedings of the thirteenth annual ACM symposium on Theory of computing_, pp. 114–122, 1981. 
*   Sun et al. (2024) Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix Yu. Spectr: Fast speculative decoding via optimal transport. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Tillet et al. (2019) Philippe Tillet, H.T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In _Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages_, MAPL 2019, pp. 10–19, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450367196. doi: 10.1145/3315508.3329973. URL [https://doi.org/10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973). 
*   Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023. 
*   Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I.Guyon, U.Von Luxburg, S.Bengio, H.Wallach, R.Fergus, S.Vishwanathan, and R.Garnett (eds.), _Advances in Neural Information Processing Systems_, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 

Appendix A Token Tree Construction Algorithm
--------------------------------------------

We present the details of our token tree construction algorithms and the corresponding verification method to ensure that the output probability distribution is consistent with the target model.

### A.1 Token Tree Construction Algorithm with Fixed Size

We demonstrate the proposed token tree construction algorithm with fixed size in Algorithm[1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

The optimal predicted token tree can be generated by greedily expanding the leaf node with the highest expectation. This method can be implemented using priority queues, similar to REST He et al. ([2023](https://arxiv.org/html/2410.11744v1#bib.bib6)).

Assume that we have a partial token tree. Then we use a heap to maintain all extendable nodes (leaf nodes or the last predicted node of its parent). Each time we extend the extendable node with the highest estimated acceptance rate. After adding one node to token tree, there are two more extendable node. One is its first child(the first prediction following this token). This prediction will only occur if the current node is received, so its estimated acceptance rate is previous_rate×p previous_rate 𝑝\texttt{previous\_rate}\times p previous_rate × italic_p, where p 𝑝 p italic_p is the estimated acceptance rate of current token. The other extendable node is its next neighbor(the next prediction of the same previous tokens). This prediction will only occur if the current node is rejected, so its estimated acceptance rate is previous_rate×(1−p)previous_rate 1 𝑝\texttt{previous\_rate}\times(1-p)previous_rate × ( 1 - italic_p ).

The algorithm starts with a single root node, which represents the input prefix. Then repeat the aforementioned process m 𝑚 m italic_m times. The estimated acceptance rate of the node can be expressed as the product of its all ancestor nodes’ probability multiply the probability that all its previous predictions failed under the same prefix tokens. The new extendable nodes (i.e., v 0 subscript 𝑣 0 v_{0}italic_v start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT and v 1 subscript 𝑣 1 v_{1}italic_v start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT in Algorithm[1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure")) should have the lower estimated acceptance rate than previous predicted tokens. It means that we generated tokens with decreasing acceptance rate and the residual nodes remain in heap or are not extendable have lower acceptance rate than any generated tokens, which means that we get the optimal token tree.

Note that the estimated acceptance rate is independent of its actual token, because we made this prediction before we know what the token is. If what this token is affects whether or not we keep the sample in draft token tree, then the final result will be biased.

Algorithm[1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") will call draft model m 𝑚 m italic_m times, which is inefficient for large m 𝑚 m italic_m. An alternative way is generating predicted tokens layer by layer. To do this, we can relax the fixed m 𝑚 m italic_m limitation to an appropriate threshold. Algorithm[1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") will greedily generate the first m 𝑚 m italic_m nodes with largest estimated acceptance rate. If we set the threshold to be the same as the acceptance rate of the last token, we will exactly get the same result as the previous algorithm. And it will only call the draft model layer number times.

### A.2 Token Tree Construction Algorithm with Threshold

Input :Prefix

x 0 subscript 𝑥 0 x_{0}italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
, draft model

D Θ(⋅|x)D_{\Theta}(\cdot|x)italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x )
, and a threshold

t 𝑡 t italic_t
.

Output :generated token tree

T⁢r 𝑇 𝑟 Tr italic_T italic_r
.

1

R←D Θ(⋅|x 0),v←1,TreeInfo←…R\leftarrow D_{\Theta}(\cdot|x_{0}),v\leftarrow 1,\texttt{TreeInfo}\leftarrow\dots italic_R ← italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) , italic_v ← 1 , TreeInfo ← …

2 LeafNodes

←←\leftarrow←
root;

3 while _LeafNodes ≠∅absent\neq\emptyset≠ ∅_ do

4 NewLeafNodes

←∅←absent\leftarrow\emptyset← ∅
;

5 foreach _\_node\_ i∈\_LeafNodes\_ subscript \_node\_ 𝑖 \_LeafNodes\_\texttt{node}\_{i}\in\texttt{LeafNodes}node start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ LeafNodes_ do

6 get input

x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
from

node i subscript node 𝑖\texttt{node}_{i}node start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
;

7

d i←D Θ(⋅|x i)d_{i}\leftarrow D_{\Theta}(\cdot|x_{i})italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_D start_POSTSUBSCRIPT roman_Θ end_POSTSUBSCRIPT ( ⋅ | italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )
;

8 get estimate acceptance rate

v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
from

node i subscript node 𝑖\texttt{node}_{i}node start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
;

9 while _v i<t subscript 𝑣 𝑖 𝑡 v\_{i}<t italic\_v start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT < italic\_t_ do

10 sample

y∼d i similar-to 𝑦 subscript 𝑑 𝑖 y\sim d_{i}italic_y ∼ italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
;

11 NewNode

←←\leftarrow←
Tr.add(

n⁢o⁢d⁢e i,y 𝑛 𝑜 𝑑 subscript 𝑒 𝑖 𝑦 node_{i},y italic_n italic_o italic_d italic_e start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_y
) ;

NewLeafNodes.append(NewNode,

v i∗d i⁢[y]subscript 𝑣 𝑖 subscript 𝑑 𝑖 delimited-[]𝑦 v_{i}*d_{i}[y]italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∗ italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT [ italic_y ]
) ;

/* expand child node */

12

v i=v i∗(1−d i⁢[y])subscript 𝑣 𝑖 subscript 𝑣 𝑖 1 subscript 𝑑 𝑖 delimited-[]𝑦 v_{i}=v_{i}*(1-d_{i}[y])italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∗ ( 1 - italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT [ italic_y ] )
;

13

d i⁢[y]=0 subscript 𝑑 𝑖 delimited-[]𝑦 0 d_{i}[y]=0 italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT [ italic_y ] = 0
;

14

d i←n⁢o⁢r⁢m⁢(d i)←subscript 𝑑 𝑖 𝑛 𝑜 𝑟 𝑚 subscript 𝑑 𝑖 d_{i}\leftarrow norm(d_{i})italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_n italic_o italic_r italic_m ( italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )
;

15

16 end while

17

18 end foreach

19 LeafNodes

←←\leftarrow←
NewLeafNodes ;

20

21 end while

Algorithm 2 Token tree construction algorithm with threshold

We present our token tree construction algorithm with threshold in Algorithm[2](https://arxiv.org/html/2410.11744v1#algorithm2 "In A.2 Token Tree Construction Algorithm with Threshold ‣ Appendix A Token Tree Construction Algorithm ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"). The different between Algorithm[1](https://arxiv.org/html/2410.11744v1#algorithm1 "In 4.2 Algorithm ‣ 4 Method ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") and Algorithm[2](https://arxiv.org/html/2410.11744v1#algorithm2 "In A.2 Token Tree Construction Algorithm with Threshold ‣ Appendix A Token Tree Construction Algorithm ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") is that we extend all nodes with estimated acceptance rate above the threshold.

### A.3 Verification

After the process of token tree, we need a corresponding verification method to ensure that the output probability distribution is consistent with the target model. Our method can be seen as the method dynamically choose the branch number of each token. So the verification method is similar to SpecInfer (Miao et al., [2023](https://arxiv.org/html/2410.11744v1#bib.bib10)) and Sequoia (Chen et al., [2024](https://arxiv.org/html/2410.11744v1#bib.bib4)). We present our verification algorithm in Algorithm[3](https://arxiv.org/html/2410.11744v1#algorithm3 "In A.3 Verification ‣ Appendix A Token Tree Construction Algorithm ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

Input :draft model distribution

D⁢r⁢a⁢f⁢t⁢(⋅)𝐷 𝑟 𝑎 𝑓 𝑡⋅Draft(\cdot)italic_D italic_r italic_a italic_f italic_t ( ⋅ )
, target model distribution

T⁢a⁢r⁢g⁢e⁢t⁢(⋅)𝑇 𝑎 𝑟 𝑔 𝑒 𝑡⋅Target(\cdot)italic_T italic_a italic_r italic_g italic_e italic_t ( ⋅ )
, speculated token tree Tr.

Output :Accepted token sequence

A 𝐴 A italic_A
.

1

2 CurrentNode

←←\leftarrow←
Tr.root;

3

A←∅←𝐴 A\leftarrow\emptyset italic_A ← ∅
;

4 while _CurrentNode.branches ≠∅absent\neq\emptyset≠ ∅_ do

5

D←D⁢r⁢a⁢f⁢t⁢(CurrentNode,⋅)←𝐷 𝐷 𝑟 𝑎 𝑓 𝑡 CurrentNode⋅D\leftarrow Draft(\texttt{CurrentNode},\cdot)italic_D ← italic_D italic_r italic_a italic_f italic_t ( CurrentNode , ⋅ )
;

6

T←T⁢a⁢r⁢g⁢e⁢t⁢(CurrentNode,⋅)←𝑇 𝑇 𝑎 𝑟 𝑔 𝑒 𝑡 CurrentNode⋅T\leftarrow Target(\texttt{CurrentNode},\cdot)italic_T ← italic_T italic_a italic_r italic_g italic_e italic_t ( CurrentNode , ⋅ )
;

7

R←T←𝑅 𝑇 R\leftarrow T italic_R ← italic_T
;

8 for _\_node\_ i∈\_CurrentNode.branches\_ subscript \_node\_ 𝑖 \_CurrentNode.branches\_\texttt{node}\_{i}\in\texttt{CurrentNode.branches}node start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ CurrentNode.branches_ do

9 get token

y 𝑦 y italic_y
from node_i ;

10 sample

c∼N⁢(0,1)similar-to 𝑐 𝑁 0 1 c\sim N(0,1)italic_c ∼ italic_N ( 0 , 1 )
;

11 if _c≤R⁢[y]D⁢[y]𝑐 𝑅 delimited-[]𝑦 𝐷 delimited-[]𝑦 c\leq\frac{R[y]}{D[y]}italic\_c ≤ divide start\_ARG italic\_R [ italic\_y ] end\_ARG start\_ARG italic\_D [ italic\_y ] end\_ARG_ then

12 A.append(y);

13 CurrentNode

←←\leftarrow←
node_i;

14 break;

15

16 else

17

R←n⁢o⁢r⁢m⁢(m⁢a⁢x⁢(R−D,0))←𝑅 𝑛 𝑜 𝑟 𝑚 𝑚 𝑎 𝑥 𝑅 𝐷 0 R\leftarrow norm(max(R-D,0))italic_R ← italic_n italic_o italic_r italic_m ( italic_m italic_a italic_x ( italic_R - italic_D , 0 ) )
;

18

D⁢[y]←0←𝐷 delimited-[]𝑦 0 D[y]\leftarrow 0 italic_D [ italic_y ] ← 0
;

19 if _D is all 0_ then

20 break;

21 end if

22

D←n⁢o⁢r⁢m⁢(D)←𝐷 𝑛 𝑜 𝑟 𝑚 𝐷 D\leftarrow norm(D)italic_D ← italic_n italic_o italic_r italic_m ( italic_D )
;

23

24 end if

25

26 end for

27 if _CurrentNode isn’t updated_ then

28 sample

y∼R similar-to 𝑦 𝑅 y\sim R italic_y ∼ italic_R
;

29 A.append(y);

30 break;

31

32 end if

33

34 end while

Algorithm 3 Verify Algorithm

The major difference between Sequoia and ours is that we directly return when the distribution of draft output become all zeros. In that case the estimated acceptance rate in our method is 0 and will never be extended.

Appendix B Additional Experiments
---------------------------------

For all experiments, we selected 1000 pieces of data from each dataset to conduct the experiment. For CNN daily we used test splits. For openwebtext we used train split. For C4 we used en splits. All the results were the result of a single run.

### B.1 DySpec with large token tree size

Under CPU-offloading setting, target model inference is extremely larger than draft model. For Llama2-70B as target and llama2-7b as draft on A100 40G, target model inference time is 2000 ×\times× larger than draft model, which gives us the opportunity to construct a larger token tree. Following Sequoia’s setting, we also make the guess token tree size up to 768. The result shows that our method can achieve a higher accepted token per step, and lower latency per token than SOTA at 0 target temperature.

Table 4: Latency per token(accepted token per step). The draft model is Llama2-7B and the target model is Llama2-70B. Guess length is 768.

*   *This data is sourced from Chen et al. ([2024](https://arxiv.org/html/2410.11744v1#bib.bib4)). 

On higher temperatures, DySpec demonstrates superior performance compared to Specinfer, but it does not surpass Sequoia. This is due to efficiency constraints that prevent us from implementing the full version of DySpec’s greedy method. Instead, we must employ a threshold to construct the token tree layer by layer. The exact threshold varies over time, which limits our ability to fully utilize the 768-token budget. For instance, at a target temperature of 0.6 on the OpenWebText dataset, with a maximum tree size set to 768 and a threshold of 0.001, the average tree size is 551.79. Figure[5](https://arxiv.org/html/2410.11744v1#A2.F5 "Figure 5 ‣ B.1 DySpec with large token tree size ‣ Appendix B Additional Experiments ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") illustrates the token tree size at each step alongside the number of accepted tokens.

To maximize the potential of DySpec’s greedy expansion method, we need to develop mechanisms for dynamically adjusting the threshold or create an alternative algorithm that eliminates the draft model inference overhead while preserving the token-by-token expansion mechanism.

![Image 9: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/treesize.png)

Figure 5: Token Tree size with accepted token number each step. 

Appendix C Block-Sparsity Friendly Token Order
----------------------------------------------

The special sparsity in tree attention brings opportunity to further optimize the attention operation. Since modern attention libraries (e.g. FlashAttention) compute block by block, different token permutations can have distinct computation workloads. To find the optimal token order, we formalize the optimization problem as below:

###### Definition 1(Block-Sparsity Friendly Token Order).

Given a tree 𝒯 𝒯\mathcal{T}caligraphic_T with size n 𝑛 n italic_n and computation block size b 𝑏 b italic_b, find a permutation 𝒫 𝒫\mathcal{P}caligraphic_P, s.t. the attention mask of tree 𝒫⁢(𝒯)𝒫 𝒯\mathcal{P}(\mathcal{T})caligraphic_P ( caligraphic_T ) has the minimal number of non-zero blocks.

Exhaustively searching through all permutations is computationally prohibitive. A near-optimal solution to this problem is heavy path decomposition (HPD)(Sleator & Tarjan, [1981](https://arxiv.org/html/2410.11744v1#bib.bib16)), which traverses nodes in descending order of their subtree sizes. This approach is effective because it groups nodes along longer paths into the same blocks whenever possible, while the long path contribute a lot to the total number of blocks in the tree attention mask (O⁢(L 2)𝑂 superscript 𝐿 2 O(L^{2})italic_O ( italic_L start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) blocks for path with length L 𝐿 L italic_L). Given the way DySpec constructs the speculative token tree, previous sibling nodes are often allocated more budget to constrain their subtrees. Consequently, the depth-first search (DFS) order closely approximates the HPD order. DySpec leverages DFS to rearrange node indices, thereby reducing the number of non-zero blocks in the attention mask. As illustrated in Figure [6](https://arxiv.org/html/2410.11744v1#A3.F6 "Figure 6 ‣ Appendix C Block-Sparsity Friendly Token Order ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") and Figure [7](https://arxiv.org/html/2410.11744v1#A3.F7 "Figure 7 ‣ Appendix C Block-Sparsity Friendly Token Order ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure"), DFS order is typically more conducive to block sparsity.

![Image 10: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/attention-1-1.png)

Figure 6: Comparing DFS order with original order.

![Image 11: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/origin.png)

(a) original order

![Image 12: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/reorder.png)

(b) DFS order

Figure 7: Tree attention mask of predicted token tree in different order. 

### C.1 Efficiency of Optimized Tree Attention

For different tasks, there exist diverse patterns of attention masks. In response to the block sparsity of these masks, numerous implementations of attention operators based on FlashAttention have been developed, However, those methods are not well-suited to support arbitrary patterns of attention masks. XFormers(Lefaudeux et al., [2022](https://arxiv.org/html/2410.11744v1#bib.bib7)) and DeepSpeed(Rasley et al., [2020](https://arxiv.org/html/2410.11744v1#bib.bib15)) have no specific API for arbitrary custom mask. Recently, PyTorch(Paszke et al., [2017](https://arxiv.org/html/2410.11744v1#bib.bib13)) introduces FlexAttention, which optimizes for arbitrary attention masks. However, to fully leverage its optimization, we must compile the kernel for different masks, which is not suitable for our target scenario of tree-based speculative decoding, where the tree attention mask changes with each iteration.

We have implemented a version of FlashAttention that supports custom masks, enabling the efficient handling of empty blocks in Triton(Tillet et al., [2019](https://arxiv.org/html/2410.11744v1#bib.bib18)). Our experiments with a random tree attention mask demonstrate that DySpec Tree Reordering can reduce the number of attention mask blocks by up to 5.9×\times×, and the attention operation can run up to 2.1 ×\times× faster, as detailed in Table[5](https://arxiv.org/html/2410.11744v1#A3.T5 "Table 5 ‣ C.1 Efficiency of Optimized Tree Attention ‣ Appendix C Block-Sparsity Friendly Token Order ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure").

In the experiment, we set Q, K, V as shape (batch=1, head_num=64, seqlen, head_dim=128), where head_num=64 and head_dim=128 is the parameter used by Llama2-70B. The block size is 32, which is usually used in attention kernel according to limited shared memory size, and it can also provide considerable block sparsity. The seqlen is varies from 256 to 2048. We also compared our custom kernel with Manual Attention and Xformer, which demonstrates that our implementation kernel is on par with the on-shelf kernel in terms of performance. And the negligible performance improvement of this kernel demonstrates that the performance enhancement of our method is entirely attributable to the reduction in the number of blocks.

In our experiment, we configured Q, K, and V with the shape (batch=1, head_num=64, seqlen, head_dim=128), aligning with the parameters used by Llama2-70B, where head_num=64 and head_dim=128. The block size was set to 32, a common choice in attention kernels due to the constraints of shared memory size, which also facilitates significant block sparsity. The sequence length (seqlen) varied from 256 to 2048. We benchmarked our custom kernel against Manual Attention and Xformers, revealing that our implementation performs comparably to existing kernels. The marginal performance improvement observed in those kernels underscores that the enhanced performance of our method is entirely due to the reduction in the number of blocks.

Table 5: Efficiency of Optimized Tree Attention with random tree structure. 

![Image 13: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/fused-attention-batch1-head64-d128-fwd.png)

Figure 8: Efficiency of Optimized Tree Attention with random tree structure.

However, this improvement is not significant in end-to-end situation. These are two problems:

1. The improvement is only significant with large context length, where extremely large sizes will result in diminishing marginal benefits of increasing size on the acceptance rate of speculative decoding. Despite the decline in acceptance rate as tree size increases, the ratio of inference speeds between the target model and the draft model itself limits the size of the tree.

Using large model like Llama2-70B with CPU-offloading will the ratio of inference speeds between the target model and the draft model, however, there is a new problem that under this setting, the most time cost operation is moving weight between CPU and GPU, and the attention operation only contribute a little in end–to-end latency.

2. The prompt is included in attention mask. As the context becomes longer, the majority of the attention calculations involve interactions between the newly added tokens and the existing context tokens. Consequently, the influence of the tree structure diminishes.

Figure[9](https://arxiv.org/html/2410.11744v1#A3.F9 "Figure 9 ‣ C.1 Efficiency of Optimized Tree Attention ‣ Appendix C Block-Sparsity Friendly Token Order ‣ DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure") illustrates the block count on a real workload tree attention mask with varying prefix lengths. Specifically, for a tree size of 768, the block count with reordering is 218.31, compared to 366.12 with the original order. Similarly, for a tree size of 1024, the block count with reordering is 295.59, while it is 580.07 with the original order.

![Image 14: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/reorder768.png)

(a) block count with tree size 768. 

![Image 15: Refer to caption](https://arxiv.org/html/2410.11744v1/extracted/5913908/figures/reorder1024.png)

(b) block count with tree size 1024.

Figure 9: Block Count with tree attention mask with/without tree reorder, with different prefix length.

Only when these two issues are resolved can reordering effectively accelerate the end-to-end latency of tree-based speculative decoding. The first issue requires a more advanced speculative decoding method capable of handling extremely large tree sizes. The second issue likely necessitates optimizing the attention computation between the prompt sequence and new tokens, thereby shifting the bottleneck to the tree attention mask itself.

Appendix D prove
----------------

The goal is to maximize the expected total acceptance tokens, denoted as T=∑i p i 𝑇 subscript 𝑖 subscript 𝑝 𝑖 T=\sum_{i}p_{i}italic_T = ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, where p i subscript 𝑝 𝑖 p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT represents the expected acceptance rate of token t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT within the predicted token tree.

Given the assumptions that (1) the probability of a token appearing in the draft model outputs, denoted as d⁢r⁢a⁢f⁢t i 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑖 draft_{i}italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , can approximate its acceptance rate, and (2) the acceptance rate of a token is independent of its preceding tokens, we can express the expected acceptance rate p i subscript 𝑝 𝑖 p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT as:

p i≈P⁢[P⁢a⁢t⁢h i]⁢d⁢r⁢a⁢f⁢t i subscript 𝑝 𝑖 𝑃 delimited-[]𝑃 𝑎 𝑡 subscript ℎ 𝑖 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑖 p_{i}\approx P[Path_{i}]draft_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ≈ italic_P [ italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT(4)

Where P⁢[P⁢a⁢t⁢h i]𝑃 delimited-[]𝑃 𝑎 𝑡 subscript ℎ 𝑖 P[Path_{i}]italic_P [ italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] represents the probability of accepting all the ancestor tokens of t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in the predicted token tree.

For multi-branch tokens under the same ancestor path, the acceptance of subsequent tokens is depends on the rejection of preceding sibling tokens. Assuming all ancestor tokens along the path have been accepted, the probability of verifying token t k subscript 𝑡 𝑘 t_{k}italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT can be expressed as:

P⁢[v⁢e⁢r⁢i⁢f⁢y i|P⁢a⁢t⁢h i]=∏j<k(1−d⁢r⁢a⁢f⁢t j)𝑃 delimited-[]conditional 𝑣 𝑒 𝑟 𝑖 𝑓 subscript 𝑦 𝑖 𝑃 𝑎 𝑡 subscript ℎ 𝑖 subscript product 𝑗 𝑘 1 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑗 P[verify_{i}|Path_{i}]=\prod_{j<k}(1-draft_{j})italic_P [ italic_v italic_e italic_r italic_i italic_f italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] = ∏ start_POSTSUBSCRIPT italic_j < italic_k end_POSTSUBSCRIPT ( 1 - italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT )(5)

Where t j<k subscript 𝑡 𝑗 𝑘 t_{j<k}italic_t start_POSTSUBSCRIPT italic_j < italic_k end_POSTSUBSCRIPT denote t k subscript 𝑡 𝑘 t_{k}italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT’s previous sibling tokens.

Put all three component together, we have

p i=P⁢[P⁢a⁢t⁢h i]×∏j j<k⁢(1−d⁢r⁢a⁢f⁢t j)×d⁢r⁢a⁢f⁢t k subscript 𝑝 𝑖 𝑃 delimited-[]𝑃 𝑎 𝑡 subscript ℎ 𝑖 subscript product 𝑗 𝑗 𝑘 1 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑗 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑘 p_{i}=P[Path_{i}]\times\prod_{j}{j<k}(1-draft_{j})\times draft_{k}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_P [ italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_j < italic_k ( 1 - italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) × italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT(6)

Although we have a method to estimate the expected acceptance token number, there are still challenges in finding the optimal structure for speculative decoding. The expectation can only be known after we have completed the sampling process. After sampling, the predicted token tree must be updated, otherwise some tokens with low acceptance rates will be pre-pruned, leading to a slightly skewed output distribution that deviates from the sole target mode. An alternative solution is to only decide whether to perform the sampling, rather than whether to add it to the predicted tree.

Assuming that all single samplings have the same acceptance rate, the target can be modified as:

T=∑p i=∑s i⁢ρ=P⁢[P⁢a⁢t⁢h i]×∏j j<k⁢(1−d⁢r⁢a⁢f⁢t j)×ρ 𝑇 absent subscript 𝑝 𝑖 subscript 𝑠 𝑖 𝜌 𝑃 delimited-[]𝑃 𝑎 𝑡 subscript ℎ 𝑖 subscript product 𝑗 𝑗 𝑘 1 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑗 𝜌\begin{array}[]{r l}T=&\sum p_{i}=\sum s_{i}\rho\\ =&P[Path_{i}]\times\prod_{j}{j<k}(1-draft_{j})\times\rho\\ \end{array}start_ARRAY start_ROW start_CELL italic_T = end_CELL start_CELL ∑ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = ∑ italic_s start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_ρ end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL italic_P [ italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] × ∏ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_j < italic_k ( 1 - italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) × italic_ρ end_CELL end_ROW end_ARRAY(7)

where s i subscript 𝑠 𝑖 s_{i}italic_s start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denotes the probability that we make this sampling, and ρ 𝜌\rho italic_ρ denotes the acceptance rate of a single isolated sampling.

For multi-branch tokens under the same ancestor path, after we sample the first token t 1 subscript 𝑡 1 t_{1}italic_t start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, the second token t 2 subscript 𝑡 2 t_{2}italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT should never be t 1 subscript 𝑡 1 t_{1}italic_t start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT because it will never pass the verification (The residual probability of target will be zero.). We should only sample the second one from the remaining tokens. Let d i subscript 𝑑 𝑖 d_{i}italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denote the original output distribution of the draft model, then the probability of sampling the second token t 2 subscript 𝑡 2 t_{2}italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT can be expressed as d⁢r⁢a⁢f⁢t 2=d t 2/(1−d t 1)𝑑 𝑟 𝑎 𝑓 subscript 𝑡 2 subscript 𝑑 subscript 𝑡 2 1 subscript 𝑑 subscript 𝑡 1 draft_{2}=d_{t_{2}}/(1-d_{t_{1}})italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_POSTSUBSCRIPT / ( 1 - italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT ).

More generally, for the k 𝑘 k italic_k-th token t k subscript 𝑡 𝑘 t_{k}italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, the probability of sampling it can be calculated as:

d⁢r⁢a⁢f⁢t k=d t k 1−(∑j<k d t j)𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑘 subscript 𝑑 subscript 𝑡 𝑘 1 subscript 𝑗 𝑘 subscript 𝑑 subscript 𝑡 𝑗 draft_{k}=\frac{d_{t_{k}}}{1-(\sum_{j<k}d_{t_{j}})}italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = divide start_ARG italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_ARG start_ARG 1 - ( ∑ start_POSTSUBSCRIPT italic_j < italic_k end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) end_ARG(8)

Combining the previous formulations, the probability of verifying the i 𝑖 i italic_i-th token given the ancestor P⁢a⁢t⁢h i 𝑃 𝑎 𝑡 subscript ℎ 𝑖 Path_{i}italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, P⁢[v⁢e⁢r⁢i⁢f⁢y i|P⁢a⁢t⁢h i]𝑃 delimited-[]conditional 𝑣 𝑒 𝑟 𝑖 𝑓 subscript 𝑦 𝑖 𝑃 𝑎 𝑡 subscript ℎ 𝑖 P[verify_{i}|Path_{i}]italic_P [ italic_v italic_e italic_r italic_i italic_f italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ], can be expressed as:

P⁢[v⁢e⁢r⁢i⁢f⁢y i|P⁢a⁢t⁢h i]=∏j<i(1−d⁢r⁢a⁢f⁢t j)=∏j<i⁢(1−d t j 1−(∑k<j d t k))=∏j<i⁢1−(∑k<j d t k)−d t j 1−(∑k<j d t k)=1−∑j<i d t j missing-subexpression 𝑃 delimited-[]conditional 𝑣 𝑒 𝑟 𝑖 𝑓 subscript 𝑦 𝑖 𝑃 𝑎 𝑡 subscript ℎ 𝑖 subscript product 𝑗 𝑖 1 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑗 product 𝑗 𝑖 1 subscript 𝑑 subscript 𝑡 𝑗 1 subscript 𝑘 𝑗 subscript 𝑑 subscript 𝑡 𝑘 product 𝑗 𝑖 1 subscript 𝑘 𝑗 subscript 𝑑 subscript 𝑡 𝑘 subscript 𝑑 subscript 𝑡 𝑗 1 subscript 𝑘 𝑗 subscript 𝑑 subscript 𝑡 𝑘 1 subscript 𝑗 𝑖 subscript 𝑑 subscript 𝑡 𝑗\begin{array}[]{rl}&P[verify_{i}|Path_{i}]=\prod_{j<i}(1-draft_{j})\\ =&\prod{j<i}(1-\frac{d_{t_{j}}}{1-(\sum_{k<j}d_{t_{k}})})\\ =&\prod{j<i}\frac{1-(\sum_{k<j}d_{t_{k}})-d_{t_{j}}}{1-(\sum_{k<j}d_{t_{k}})}% \\ =&1-\sum_{j<i}d_{t_{j}}\\ \end{array}start_ARRAY start_ROW start_CELL end_CELL start_CELL italic_P [ italic_v italic_e italic_r italic_i italic_f italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] = ∏ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT ( 1 - italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∏ italic_j < italic_i ( 1 - divide start_ARG italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_ARG start_ARG 1 - ( ∑ start_POSTSUBSCRIPT italic_k < italic_j end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) end_ARG ) end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∏ italic_j < italic_i divide start_ARG 1 - ( ∑ start_POSTSUBSCRIPT italic_k < italic_j end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) - italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_ARG start_ARG 1 - ( ∑ start_POSTSUBSCRIPT italic_k < italic_j end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) end_ARG end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL 1 - ∑ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW end_ARRAY(9)

For the probability of the path, P⁢[p⁢a⁢t⁢h i]𝑃 delimited-[]𝑝 𝑎 𝑡 subscript ℎ 𝑖 P[path_{i}]italic_P [ italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ], where p⁢a⁢t⁢h i=x 1,…,x i−1 𝑝 𝑎 𝑡 subscript ℎ 𝑖 subscript 𝑥 1…subscript 𝑥 𝑖 1 path_{i}=x_{1},...,x_{i-1}italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT, and under the independence assumption, we have:

P⁢[p⁢a⁢t⁢h i]=∏j<i P⁢[a⁢c⁢c⁢e⁢p⁢t⁢x j|p⁢a⁢t⁢h j]=∏j<i P⁢[v⁢e⁢r⁢i⁢f⁢y j|P⁢a⁢t⁢h j]×d⁢r⁢a⁢f⁢t j=∏j<i(1−∑k<j d t k)⁢d t j 1−∑k<j d t k=∏j<i d t j missing-subexpression 𝑃 delimited-[]𝑝 𝑎 𝑡 subscript ℎ 𝑖 subscript product 𝑗 𝑖 𝑃 delimited-[]conditional 𝑎 𝑐 𝑐 𝑒 𝑝 𝑡 subscript 𝑥 𝑗 𝑝 𝑎 𝑡 subscript ℎ 𝑗 subscript product 𝑗 𝑖 𝑃 delimited-[]conditional 𝑣 𝑒 𝑟 𝑖 𝑓 subscript 𝑦 𝑗 𝑃 𝑎 𝑡 subscript ℎ 𝑗 𝑑 𝑟 𝑎 𝑓 subscript 𝑡 𝑗 subscript product 𝑗 𝑖 1 subscript 𝑘 𝑗 subscript 𝑑 subscript 𝑡 𝑘 subscript 𝑑 subscript 𝑡 𝑗 1 subscript 𝑘 𝑗 subscript 𝑑 subscript 𝑡 𝑘 subscript product 𝑗 𝑖 subscript 𝑑 subscript 𝑡 𝑗\begin{array}[]{rl}&P[path_{i}]=\prod_{j<i}P[acceptx_{j}|path_{j}]\\ =&\prod_{j<i}P[verify_{j}|Path_{j}]\times draft_{j}\\ =&\prod_{j<i}(1-\sum_{k<j}d_{t_{k}})\frac{d_{t_{j}}}{1-\sum_{k<j}d_{t_{k}}}\\ =&\prod_{j<i}d_{t_{j}}\\ \end{array}start_ARRAY start_ROW start_CELL end_CELL start_CELL italic_P [ italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] = ∏ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT italic_P [ italic_a italic_c italic_c italic_e italic_p italic_t italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT | italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∏ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT italic_P [ italic_v italic_e italic_r italic_i italic_f italic_y start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT | italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] × italic_d italic_r italic_a italic_f italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∏ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT ( 1 - ∑ start_POSTSUBSCRIPT italic_k < italic_j end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) divide start_ARG italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_ARG start_ARG 1 - ∑ start_POSTSUBSCRIPT italic_k < italic_j end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_ARG end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∏ start_POSTSUBSCRIPT italic_j < italic_i end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW end_ARRAY(10)

Combining these, the final target expression becomes:

T=∑p i=∑i P⁢[p⁢a⁢t⁢h i]⁢P⁢[v⁢e⁢r⁢i⁢f⁢y i|P⁢a⁢t⁢h i]⁢ρ=∑i∏j∈p⁢a⁢t⁢h i d t j⁢ρ×(1−∑k⁢is the sibling token before⁢i d t k)𝑇 absent subscript 𝑝 𝑖 subscript 𝑖 𝑃 delimited-[]𝑝 𝑎 𝑡 subscript ℎ 𝑖 𝑃 delimited-[]conditional 𝑣 𝑒 𝑟 𝑖 𝑓 subscript 𝑦 𝑖 𝑃 𝑎 𝑡 subscript ℎ 𝑖 𝜌 subscript 𝑖 subscript product 𝑗 𝑝 𝑎 𝑡 subscript ℎ 𝑖 subscript 𝑑 subscript 𝑡 𝑗 𝜌 missing-subexpression absent 1 subscript 𝑘 is the sibling token before 𝑖 subscript 𝑑 subscript 𝑡 𝑘\begin{array}[]{rl}T=&\sum p_{i}\\ =&\sum_{i}P[path_{i}]P[verify_{i}|Path_{i}]\rho\\ =&\sum_{i}\prod_{j\in path_{i}}d_{t_{j}}\rho\\ &\times(1-\sum_{k\text{ is the sibling token before }i}d_{t_{k}})\\ \end{array}start_ARRAY start_ROW start_CELL italic_T = end_CELL start_CELL ∑ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_P [ italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] italic_P [ italic_v italic_e italic_r italic_i italic_f italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | italic_P italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] italic_ρ end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∏ start_POSTSUBSCRIPT italic_j ∈ italic_p italic_a italic_t italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT italic_ρ end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL × ( 1 - ∑ start_POSTSUBSCRIPT italic_k is the sibling token before italic_i end_POSTSUBSCRIPT italic_d start_POSTSUBSCRIPT italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) end_CELL end_ROW end_ARRAY(11)

Note that for deeper tokens and sibling tokens after, the acceptance rate p i subscript 𝑝 𝑖 p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT will monotonically decrease, which means we can construct the predicted tree greedily.

Our method ensures that at each step, we perform sampling with the maximum expected acceptance rate. To demonstrate this, assume that there exists an alternative method that can generate a better tree of the same size n 𝑛 n italic_n. There must be at least one leaf node that differs between this alternative method and our method. Let’s denote the leaf nodes from the alternative method as N c subscript 𝑁 𝑐 N_{c}italic_N start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT and the corresponding leaf nodes from our method as N o⁢u⁢r subscript 𝑁 𝑜 𝑢 𝑟 N_{our}italic_N start_POSTSUBSCRIPT italic_o italic_u italic_r end_POSTSUBSCRIPT. Furthermore, let’s denote the first ancestor node of N c subscript 𝑁 𝑐 N_{c}italic_N start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT that is not present in our result as M c subscript 𝑀 𝑐 M_{c}italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT, and assume that there are k 𝑘 k italic_k nodes in the sub-tree of M c subscript 𝑀 𝑐 M_{c}italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT.

Denote the expected acceptance rate of this sample as P⁢[M c]𝑃 delimited-[]subscript 𝑀 𝑐 P[M_{c}]italic_P [ italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ]. Then, the contribution of the entire sub-tree is at most k×P⁢[M c]𝑘 𝑃 delimited-[]subscript 𝑀 𝑐 k\times P[M_{c}]italic_k × italic_P [ italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ]. The fact that our method did not choose this sub-tree implies that the last k 𝑘 k italic_k samples we made, which are not present in the alternative method, have an expected acceptance rate higher than P⁢[M c]𝑃 delimited-[]subscript 𝑀 𝑐 P[M_{c}]italic_P [ italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ]. The contribution of these k 𝑘 k italic_k samples to the expectation of the total number is larger than k×P⁢[M c]𝑘 𝑃 delimited-[]subscript 𝑀 𝑐 k\times P[M_{c}]italic_k × italic_P [ italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ].

By eliminating these k 𝑘 k italic_k nodes and applying induction, we can show that E n−k,o⁢u⁢r⁢s≥E n−k,c subscript 𝐸 𝑛 𝑘 𝑜 𝑢 𝑟 𝑠 subscript 𝐸 𝑛 𝑘 𝑐 E_{n-k,ours}\geq E_{n-k,c}italic_E start_POSTSUBSCRIPT italic_n - italic_k , italic_o italic_u italic_r italic_s end_POSTSUBSCRIPT ≥ italic_E start_POSTSUBSCRIPT italic_n - italic_k , italic_c end_POSTSUBSCRIPT, where E n−k,o⁢u⁢r⁢s subscript 𝐸 𝑛 𝑘 𝑜 𝑢 𝑟 𝑠 E_{n-k,ours}italic_E start_POSTSUBSCRIPT italic_n - italic_k , italic_o italic_u italic_r italic_s end_POSTSUBSCRIPT and E n−k,c subscript 𝐸 𝑛 𝑘 𝑐 E_{n-k,c}italic_E start_POSTSUBSCRIPT italic_n - italic_k , italic_c end_POSTSUBSCRIPT represent the expected number of accepted tokens for our method and the alternative method, respectively. Additionally, we have ∑k P⁢[M i,o⁢u⁢r⁢s]≥k×P⁢[M c]≥∑k P⁢[M i′,c]superscript 𝑘 𝑃 delimited-[]subscript 𝑀 𝑖 𝑜 𝑢 𝑟 𝑠 𝑘 𝑃 delimited-[]subscript 𝑀 𝑐 superscript 𝑘 𝑃 delimited-[]subscript 𝑀 superscript 𝑖′𝑐\sum^{k}P[M_{i,ours}]\geq k\times P[M_{c}]\geq\sum^{k}P[M_{i^{\prime},c}]∑ start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_P [ italic_M start_POSTSUBSCRIPT italic_i , italic_o italic_u italic_r italic_s end_POSTSUBSCRIPT ] ≥ italic_k × italic_P [ italic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ] ≥ ∑ start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_P [ italic_M start_POSTSUBSCRIPT italic_i start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , italic_c end_POSTSUBSCRIPT ], where M i,o⁢u⁢r⁢s subscript 𝑀 𝑖 𝑜 𝑢 𝑟 𝑠 M_{i,ours}italic_M start_POSTSUBSCRIPT italic_i , italic_o italic_u italic_r italic_s end_POSTSUBSCRIPT and M i′,c subscript 𝑀 superscript 𝑖′𝑐 M_{i^{\prime},c}italic_M start_POSTSUBSCRIPT italic_i start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , italic_c end_POSTSUBSCRIPT are the corresponding ancestor nodes in our method and the alternative method, respectively. Combining these results, we can conclude that E n,o⁢u⁢r⁢s≥E n,c subscript 𝐸 𝑛 𝑜 𝑢 𝑟 𝑠 subscript 𝐸 𝑛 𝑐 E_{n,ours}\geq E_{n,c}italic_E start_POSTSUBSCRIPT italic_n , italic_o italic_u italic_r italic_s end_POSTSUBSCRIPT ≥ italic_E start_POSTSUBSCRIPT italic_n , italic_c end_POSTSUBSCRIPT, proving that our method can maximize the expected number of accepted tokens.

### D.1 Greedy Optimal Proof

The search space for the responses form a hierarchical k 𝑘 k italic_k-wise tree S 𝑆 S italic_S, with k 𝑘 k italic_k being the number of tokens in the vocabulary. For a model M 𝑀 M italic_M, it induce a set of weights on the search space. More specifically, for any node u n subscript 𝑢 𝑛 u_{n}italic_u start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT, assume the unique path starting from the root that lead to u n subscript 𝑢 𝑛 u_{n}italic_u start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT is u 0,u 1,…,u n subscript 𝑢 0 subscript 𝑢 1…subscript 𝑢 𝑛 u_{0},u_{1},...,u_{n}italic_u start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_u start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_u start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT, define the weight for node u n subscript 𝑢 𝑛 u_{n}italic_u start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT to be:

w u n=Π m=0 n−1⁢P M⁢(u m+1|u 0:m)subscript 𝑤 subscript 𝑢 𝑛 superscript subscript Π 𝑚 0 𝑛 1 subscript 𝑃 𝑀 conditional subscript 𝑢 𝑚 1 subscript 𝑢:0 𝑚 w_{u_{n}}=\Pi_{m=0}^{n-1}P_{M}(u_{m+1}|u_{0:m})italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT end_POSTSUBSCRIPT = roman_Π start_POSTSUBSCRIPT italic_m = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n - 1 end_POSTSUPERSCRIPT italic_P start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT ( italic_u start_POSTSUBSCRIPT italic_m + 1 end_POSTSUBSCRIPT | italic_u start_POSTSUBSCRIPT 0 : italic_m end_POSTSUBSCRIPT )(12)

Consider a subset S′superscript 𝑆′S^{\prime}italic_S start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT of the space S 𝑆 S italic_S, the weight of the set w S′subscript 𝑤 superscript 𝑆′w_{S^{\prime}}italic_w start_POSTSUBSCRIPT italic_S start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT is defined as the summation of all the nodes’ weights in the subset, i.e.:

w S′=∑v∈S′w v subscript 𝑤 superscript 𝑆′subscript 𝑣 superscript 𝑆′subscript 𝑤 𝑣 w_{S^{\prime}}=\sum_{v\in{S^{\prime}}}w_{v}italic_w start_POSTSUBSCRIPT italic_S start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_v ∈ italic_S start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT(13)

Define 𝒯 𝒯\mathcal{T}caligraphic_T to be the collection of all connected sub-trees that contain the root. We are interested in finding sub-trees with the max weight with number of nodes less than N 𝑁 N italic_N, i.e.

𝒯 N∗={T|w T=max T∈𝒯⁡w T}superscript subscript 𝒯 𝑁 conditional-set 𝑇 subscript 𝑤 𝑇 subscript 𝑇 𝒯 subscript 𝑤 𝑇\mathcal{T}_{N}^{*}=\{T|w_{T}=\max_{T\in\mathcal{T}}w_{T}\}caligraphic_T start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = { italic_T | italic_w start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT = roman_max start_POSTSUBSCRIPT italic_T ∈ caligraphic_T end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT }(14)

Algorithm (Greedy): Suppose we start from the set that only contain the root M 1={r⁢o⁢o⁢t}subscript 𝑀 1 𝑟 𝑜 𝑜 𝑡 M_{1}=\{root\}italic_M start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = { italic_r italic_o italic_o italic_t }.

Define the candidate set C⁢(M i)=N⁢(M i)\M i 𝐶 subscript 𝑀 𝑖\𝑁 subscript 𝑀 𝑖 subscript 𝑀 𝑖 C(M_{i})=N(M_{i})\backslash M_{i}italic_C ( italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = italic_N ( italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) \ italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

Pick the node v∗=arg⁡max v∈C⁢(M i)⁡w v superscript 𝑣 subscript 𝑣 𝐶 subscript 𝑀 𝑖 subscript 𝑤 𝑣 v^{*}=\arg\max_{v\in C(M_{i})}w_{v}italic_v start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = roman_arg roman_max start_POSTSUBSCRIPT italic_v ∈ italic_C ( italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT

M i+1=M i∪{v∗}subscript 𝑀 𝑖 1 subscript 𝑀 𝑖 superscript 𝑣 M_{i+1}=M_{i}\cup\{v^{*}\}italic_M start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∪ { italic_v start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT }

Theorem:

(A) M N∈𝒯 subscript 𝑀 𝑁 𝒯 M_{N}\in\mathcal{T}italic_M start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT ∈ caligraphic_T

(B) M N∈𝒯 N∗subscript 𝑀 𝑁 subscript superscript 𝒯 𝑁 M_{N}\in\mathcal{T}^{*}_{N}italic_M start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT ∈ caligraphic_T start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT

###### Proof.

We will prove each part of the theorem separately.

We first prove (A), which is equivalent to verify M N subscript 𝑀 𝑁 M_{N}italic_M start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT forms a connected tree that contain the root. The latter fact is trivial since r⁢o⁢o⁢t∈M 1⊂M N 𝑟 𝑜 𝑜 𝑡 subscript 𝑀 1 subscript 𝑀 𝑁 root\in M_{1}\subset M_{N}italic_r italic_o italic_o italic_t ∈ italic_M start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ⊂ italic_M start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT. It’s also straightforward to see the connectivity as at every step the new added node belongs to the neighbor. Finally, since a connected subset of a tree S 𝑆 S italic_S is also a tree, therefore we conclude (A).

For (B), we prove by induction. For N=1 𝑁 1 N=1 italic_N = 1, this is trivial. Suppose for N≤k 𝑁 𝑘 N\leq k italic_N ≤ italic_k, M N∈𝒯 N∗subscript 𝑀 𝑁 subscript superscript 𝒯 𝑁 M_{N}\in\mathcal{T}^{*}_{N}italic_M start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT ∈ caligraphic_T start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT, we prove this for N=k+1 𝑁 𝑘 1 N=k+1 italic_N = italic_k + 1. For any M k+1′∈𝒯 k+1 superscript subscript 𝑀 𝑘 1′subscript 𝒯 𝑘 1 M_{k+1}^{\prime}\in\mathcal{T}_{k+1}italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ∈ caligraphic_T start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT, and any M k∈𝒯 k∗subscript 𝑀 𝑘 subscript superscript 𝒯 𝑘 M_{k}\in\mathcal{T}^{*}_{k}italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∈ caligraphic_T start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, we show w M k+max v∈C⁢(M k)⁡w v≥w M k+1′subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑣 𝐶 subscript 𝑀 𝑘 subscript 𝑤 𝑣 subscript 𝑤 superscript subscript 𝑀 𝑘 1′w_{M_{k}}+\max_{v\in C(M_{k})}w_{v}\geq w_{M_{k+1}^{\prime}}italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT + roman_max start_POSTSUBSCRIPT italic_v ∈ italic_C ( italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ≥ italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT.

To show this, note that |M k+1′|=k+1>k=|M k|superscript subscript 𝑀 𝑘 1′𝑘 1 𝑘 subscript 𝑀 𝑘|M_{k+1}^{\prime}|=k+1>k=|M_{k}|| italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | = italic_k + 1 > italic_k = | italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT |, there exist at least one leaf node v∈M k+1′𝑣 superscript subscript 𝑀 𝑘 1′v\in M_{k+1}^{\prime}italic_v ∈ italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT such that v∉M k 𝑣 subscript 𝑀 𝑘 v\notin M_{k}italic_v ∉ italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. Consider the unique path that connect the root and v 𝑣 v italic_v as u 0,…,u p=v subscript 𝑢 0…subscript 𝑢 𝑝 𝑣 u_{0},...,u_{p}=v italic_u start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , … , italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT = italic_v. Since u 0∈M k subscript 𝑢 0 subscript 𝑀 𝑘 u_{0}\in M_{k}italic_u start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ∈ italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and u p∉M k subscript 𝑢 𝑝 subscript 𝑀 𝑘 u_{p}\notin M_{k}italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT ∉ italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, there must be some q∈{1,…,p}𝑞 1…𝑝 q\in\{1,...,p\}italic_q ∈ { 1 , … , italic_p } satisfy u q−1∈M k subscript 𝑢 𝑞 1 subscript 𝑀 𝑘 u_{q-1}\in M_{k}italic_u start_POSTSUBSCRIPT italic_q - 1 end_POSTSUBSCRIPT ∈ italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and u q∉M k subscript 𝑢 𝑞 subscript 𝑀 𝑘 u_{q}\notin M_{k}italic_u start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ∉ italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. By definition, u q∈C⁢(M k)subscript 𝑢 𝑞 𝐶 subscript 𝑀 𝑘 u_{q}\in C(M_{k})italic_u start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ∈ italic_C ( italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) since it’s the neighbor of M k subscript 𝑀 𝑘 M_{k}italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. And according to the definition of the weight, w u q≥w u p subscript 𝑤 subscript 𝑢 𝑞 subscript 𝑤 subscript 𝑢 𝑝 w_{u_{q}}\geq w_{u_{p}}italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT end_POSTSUBSCRIPT ≥ italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT. Now consider the fact that M k+1′\w u p\superscript subscript 𝑀 𝑘 1′subscript 𝑤 subscript 𝑢 𝑝 M_{k+1}^{\prime}\backslash w_{u_{p}}italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT \ italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT is still a tree since u p subscript 𝑢 𝑝 u_{p}italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT is a leaf, so by induction, we have w M k≥w M k+1′\w u p subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑤\superscript subscript 𝑀 𝑘 1′subscript 𝑤 subscript 𝑢 𝑝 w_{M_{k}}\geq w_{M_{k+1}^{\prime}\backslash w_{u_{p}}}italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ≥ italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT \ italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT. Therefore, we have

w M k+max v∈C⁢(M k)⁡w v≥w M k+w u q≥w M k+w u p≥w M k+1′\w u p+w u p=w M k+1′missing-subexpression subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑣 𝐶 subscript 𝑀 𝑘 subscript 𝑤 𝑣 subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑤 subscript 𝑢 𝑞 subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑤 subscript 𝑢 𝑝 subscript 𝑤\superscript subscript 𝑀 𝑘 1′subscript 𝑤 subscript 𝑢 𝑝 subscript 𝑤 subscript 𝑢 𝑝 subscript 𝑤 superscript subscript 𝑀 𝑘 1′\begin{array}[]{rl}&w_{M_{k}}+\max_{v\in C(M_{k})}w_{v}\\ \geq&w_{M_{k}}+w_{u_{q}}\\ \geq&w_{M_{k}}+w_{u_{p}}\\ \geq&w_{M_{k+1}^{\prime}\backslash w_{u_{p}}}+w_{u_{p}}\\ =&w_{M_{k+1}^{\prime}}\end{array}start_ARRAY start_ROW start_CELL end_CELL start_CELL italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT + roman_max start_POSTSUBSCRIPT italic_v ∈ italic_C ( italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL ≥ end_CELL start_CELL italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL ≥ end_CELL start_CELL italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL ≥ end_CELL start_CELL italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT \ italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_u start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL = end_CELL start_CELL italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT end_CELL end_ROW end_ARRAY(15)

Because M k+1′superscript subscript 𝑀 𝑘 1′M_{k+1}^{\prime}italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT is chosen arbitrarily, we proved that w M k+max v∈C⁢(M k)⁡w v=w M k+1′subscript 𝑤 subscript 𝑀 𝑘 subscript 𝑣 𝐶 subscript 𝑀 𝑘 subscript 𝑤 𝑣 subscript 𝑤 superscript subscript 𝑀 𝑘 1′w_{M_{k}}+\max_{v\in C(M_{k})}w_{v}=w_{M_{k+1}^{\prime}}italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT + roman_max start_POSTSUBSCRIPT italic_v ∈ italic_C ( italic_M start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT = italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT, completing the proof of (B). ∎

