Title: Sequoia: Scalable and Robust Speculative Decoding

URL Source: https://arxiv.org/html/2402.12374

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
2Background
3Sequoia
4Evaluation
5Conclusion
 References

HTML conversions sometimes display errors due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: fontawesome5
failed: savetrees
failed: savetrees

Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.

License: arXiv.org perpetual non-exclusive license
arXiv:2402.12374v3 [cs.CL] 05 Jul 2025
Sequoia: Scalable and Robust Speculative Decoding
Zhuoming Chen1  Avner May2∗  Ruslan Svirschevski3,4∗
Yuhsun Huang1  Max Ryabinin2  Zhihao Jia1  Beidi Chen1,5
1Carnegie Mellon University  2Together AI  3Yandex
4National Research University Higher School of Economics  5FAIR, Meta
zhuominc@andrew.cmu.edu, avner@together.ai, ruslansv@gmail.com
yuhsunh@andrew.cmu.edu, mryab@together.ai, {zhihaoj2,beidic}@andrew.cmu.edu
Equal contribution
Abstract

As the usage of large language models (LLMs) grows, it becomes increasingly important to serve them quickly and efficiently. While speculative decoding has recently emerged as a promising direction for accelerating LLM serving, existing methods are limited in their ability to scale to larger speculation budgets and adapt to different hyperparameters. This paper introduces Sequoia, a scalable and robust algorithm for speculative decoding. To improve scalability, Sequoia introduces a dynamic programming algorithm to find an optimal tree structure for the speculated tokens. To achieve robust speculative decoding, Sequoia uses a novel sampling and verification method that outperforms prior work across different decoding temperatures. Sequoia improves the decoding speed of Llama2-7B, Llama2-13B, and Vicuna-33B on an A100 GPU by up to 
4.04
×
, 
3.73
×
, and 
2.27
×
. To serve Llama3-70B-Instruct on a single L40 GPU through offloading, Sequoia reduces the per-token decoding latency to 0.60 s/token, 
9.5
×
 faster than DeepSpeed-Zero-Inference. The code is available at https://github.com/Infini-AI-Lab/Sequoia.

1Introduction

As large language models (LLMs) gain widespread adoption [3, 43, 7], efficiently serving these LLMs becomes increasingly important. However, accelerating LLM inference is challenging since generating a single new token requires accessing all parameters of the LLM [34]. As a result of this I/O bottleneck, the hardware is poorly utilized during generation. This problem is exacerbated in both small-batch and offloading-based inference settings, where generating one token takes as much time as processing a prompt with hundreds or thousands of tokens on modern GPUs.

Figure 1:Sequoia is a scalable method for speculative decoding. Left: Sequoia tree construction algorithm is able to generate trees whose average number of generated tokens (after verification) continues to grow with the tree size while existing tree structures asymptote. This allows Sequoia to perform much better than existing methods in very memory-bound regimes like offloading. Right: A visualization to contrast Sequoia tree structure with other common handcrafted ones.

To address this challenge, recent work has introduced speculative decoding to accelerate LLM inference while preserving the LLM’s output distribution [24, 5, 28, 40]. These approaches leverage one or multiple draft models to predict the LLM’s output; the predictions are organized in a token tree, whose nodes represent different sequences of speculated tokens. The correctness of these speculated tokens is then verified in parallel through a single forward pass of the LLM. Using a token tree—instead of a sequence—can increase the number of tokens accepted by the LLM by providing several options for each token position.

While there are substantial studies on tree-based speculative decoding methods [28, 40], we see in our experiments that they have a couple of limitations. First, we observe that existing token tree construction algorithms perform well for small token trees but are sub-optimal for large tree sizes. For example, SpecInfer constructs a token tree using 
𝑘
 independent sequences, a topology that is bounded by the expected number of tokens it can accept, regardless of the tree size (Figure 1). Second, we observe that existing token tree sampling and verification algorithms are unable to perform well across inference hyperparameter configurations; for example, SpecInfer [28] and SpecTr [40] often perform poorly at low temperatures (Figure 3) since they can repeatedly sample an incorrect token with high draft model probability.

In this paper, we aim to answer the following research question: how can we design an optimal tree-based speculative decoding method to maximize speedups on modern hardware? Realizing this goal requires addressing several technical challenges. First, for any tree size and depth, we must be able to efficiently search the exponentially large space of tree topologies to find the one that maximizes the expected number of generated tokens. Second, we must design a tree sampling and verification procedure that performs well across inference hyperparameters, avoids repeatedly sampling incorrect tokens, and maintains the correct output distribution.

This paper introduces Sequoia, a scalable and robust speculative decoding algorithm. As shown in Figure 1, Sequoia can attain up to 9.5
×
 speedups over incremental decoding and introduces several key techniques to address the aforementioned challenges.

• 

In Section 3.1, to solve the first challenge, we formulate tree construction as a constrained optimization problem and employ a dynamic programming algorithm to discover the optimal speculative token tree. Theoretically and empirically, we demonstrate that the number of tokens generated with this tree structure is unbounded, growing roughly logarithmically with the tree’s size.

• 

In Section 3.2, to address the second challenge, we build upon the SpecInfer [28] algorithm by performing sampling without replacement from the draft model—thereby preventing the draft model from making the same mistake twice, while maintaining the target model’s output distribution. We prove that this new sampling and verification method can attain high acceptance rates at both high and low temperatures and validate this claim empirically.

In Section 4, we perform extensive end-to-end experiments and ablation studies to demonstrate the effectiveness of Sequoia. We implement Sequoia on top of Hugging Face [45] with CUDA Graphs [31, 32]. We show that Sequoia achieves up to 
4.04
×
 speedup for Llama2-7B on a single A100 GPU and 
9.5
×
 for Llama3-70B-Instruct in the offloading setting on an L40 GPU. The latency of Llama3-70B-Instruct offloading on L40 can be reduced to 0.60 s/token with Sequoia while the inference speed of state-of-the-art offloading system (DeepSpeed-Zero-Inference [2]) is 5.7 s/token. We also present ablation studies to show that: (1) the Sequoia tree structure can generate up to 
33
%
 more tokens per decoding step compared to 
𝑘
 independent sequences (tree size 
≤
512
), demonstrating better scalability; (2) the Sequoia sampling and verification algorithm is robust to the choice of hyperparameters (temperature, top-
𝑝
), providing up to 
65
%
 and 
27
%
 speedup compared to SpecInfer and top-
𝑘
 sampling and verification algorithms, respectively.

2Background

Here, we review tree-based speculative decoding methods. In particular, we discuss the way existing methods choose the speculated tree structure (Section 2.1) and the algorithms they use to sample and verify the token trees (Section 2.2).

2.1Tree construction

The primary tree structure used by existing methods is one composed of 
𝑘
 independent sequences of length 
𝐿
 that branch from the tree root (which corresponds to the current prefix). The SpecTr paper additionally considers arbitrary branching patterns 
(
𝑘
1
,
𝑘
2
,
…
,
𝑘
𝑡
)
, but says that this did not perform better in their experiments than independent sequences. Medusa constructs a full 
𝑘
-ary tree, which increases the success rate at each layer but cannot form a deep tree under moderate token budgets [4].

2.2Tree sampling and verification

We now review how SpecInfer [28], SpecTr [40], naive sampling [28], and top-
𝑘
 sampling1 perform token tree sampling and verification. With regard to sampling, SpecInfer, SpecTr, and naive sampling all perform i.i.d. sampling with replacement from the draft model, while top-
𝑘
 sampling selects the top-
𝑘
 highest probability tokens from the draft model. In terms of verification, SpecInfer and SpecTr compare the draft and target model probabilities for the sampled tokens to decide which (if any) to accept; naive and top-
𝑘
 sampling, on the other hand, sample a token from the target model distribution and accept it if it corresponds to one of the tokens from the speculated tree. These methods all verify a speculated token tree in a recursive manner—starting at the root of the tree—differing only in the verification algorithm they apply at each node.

SpecInfer:

The SpecInfer method iteratively verifies tokens that were sampled from one or more draft models. Like the original speculative decoding method [24], it compares the draft model probabilities to those from the target model to decide if to accept. Note that while the SpecInfer method allows sampling from 
𝑘
 different draft models to generate 
𝑘
 children for a node, in this work we consider the more common setting where only one draft model is available. Therefore, we compare with the version of SpecInfer which samples from a single draft model 
𝑘
 times instead. To see pseudocode for SpecInfer, please see Algorithm 2 and ignore all blue lines (lines 10-16).

SpecTr:

The SpecTr algorithm is similar in spirit to the SpecInfer algorithm. It iterates through the children of a node, and uses a sampling procedure to decide if to accept a child, in such a way that the output distribution is unchanged. One important property of this algorithm is that it is within a factor of 
(
1
−
1
/
𝑒
)
 of the best possible verification algorithm (i.e., the one with highest possible acceptance rate). For brevity, we refer readers to Algorithm 3 in the SpecTr paper for the exact pseudocode for this algorithm.

Naive sampling and top-
𝑘
 sampling:

Given a node in a token tree, the verification algorithm for naive sampling and top-
𝑘
 sampling first samples from the target model’s distribution 
𝒫
(
⋅
|
𝑥
<
𝑛
)
 at that node, and then accepts this sample if it is equal to one of the children of that node. This verification algorithm trivially maintains the target model output distribution—regardless of how the token tree was generated—given that one always samples from the target model in this algorithm (as opposed to from the draft model, like in SpecTr and SpecInfer). This observation motivates our choice—for the top-
𝑘
 sampling method—to populate the tree by taking the top-
𝑘
 children of each node, instead of the naive sampling approach of taking 
𝑘
 i.i.d. samples (with replacement). We use the top-
𝑘
 sampling method in our experiments in Section 3.2, to better understand the limits of this verification algorithm.

3Sequoia

We now present Sequoia, a scalable and robust speculative decoding algorithm.

• 

In Section 3.1, we present our scalable tree construction algorithm, which uses dynamic programming to solve for the optimal tree structure. We demonstrate both theoretically and empirically that the number of tokens generated by verifying Sequoia trees scales nearly logarithmically in the size of the tree, while existing tree structures asymptote in the number of tokens they can generate.

• 

In Section 3.2, we present our robust tree verification algorithm, which modifies the SpecInfer algorithm by sampling without replacement from the draft model. We show both theoretically and empirically that Sequoia is robust, performing well across temperature values, while existing verification methods are not.

3.1Tree construction

We now present the Sequoia tree construction algorithm (Section 2), and prove that the expected number of tokens generated when verifying for these trees scales well with the tree size (Section 3.1.2).

3.1.1Algorithm
Figure 2:Left: Recursive sub-structure use by the dynamic programming algorithm. Right: Real example of Sequoia tree of size 64, and maximum depth 12. We present more examples of Sequoia trees in Figure 5 in Appendix E.

To derive the Sequoia tree construction algorithm, we first express the tree construction problem as a constrained optimization problem, and then use dynamic programming to solve this problem optimally and efficiently. In this optimization problem, we aim to maximize the expected number of tokens 
𝐹
⁢
(
𝒯
)
 generated by verifying a token tree 
𝒯
, under a constraint on the size of 
𝒯
. We begin by presenting a closed form expression for 
𝐹
⁢
(
𝒯
)
 (Proposition 3.4). We then present our tree construction algorithm, which uses dynamic programming to find the tree of size 
𝑛
 which maximizes this expression (for any value of the speculation budget 
𝑛
).

We first present a number of important definitions:

Definition 3.1.

Under the positional acceptance assumption, the probability of a verification algorithm accepting a token 
𝑡
 which is the 
𝑘
𝑡
⁢
ℎ
 child of an already accepted token depends only on the value of 
𝑘
.

Definition 3.2.

The acceptance vector is the vector 
𝑝
=
(
𝑝
1
,
𝑝
2
,
…
,
𝑝
𝑘
,
…
)
 containing the probabilities 
𝑝
𝑘
 that the verification algorithm accepts a token at child position 
𝑘
. Under the positional acceptance assumption, the acceptance dynamics of a verification algorithm can be completely described by the acceptance vector.

Definition 3.3.

Given an acceptance vector 
𝑝
 and a tree 
𝒯
, we define the score function 
𝑓
⁢
(
𝑣
)
 for a node 
𝑣
∈
𝒯
 as 
𝑓
⁢
(
𝑣
)
=
∏
𝑖
∈
Path
⁢
(
𝑣
)
𝑝
𝑖
. where 
Path
⁢
(
𝑣
)
 is equal to the list of child indices along the path from the root to a node 
𝑣
∈
𝒯
. For example, if 
𝑣
 is the 
3
𝑟
⁢
𝑑
 child of the root’s 
2
𝑛
⁢
𝑑
 child, then 
Path
⁢
(
𝑣
)
=
[
2
,
3
]
. We define 
𝑓
⁢
(
𝑟
⁢
𝑜
⁢
𝑜
⁢
𝑡
)
=
1
.

We are now ready to present Proposition 3.4 (proof in Appendix F.1.2), which shows the closed form equation for the expected number of tokens generated by verifying a token tree 
𝒯
, under the positional acceptance assumption. This is the equation which our Sequoia dynamic program will optimize.

Proposition 3.4.

Let 
𝒯
 be a token tree that is verified with the positional acceptance assumption, and let 
𝑓
⁢
(
𝑣
)
 denote the score function for a node 
𝑣
∈
𝒯
. Then the the expected number of tokens 
𝐹
⁢
(
𝒯
)
 generated by verifying 
𝒯
 equals

	
𝐹
⁢
(
𝒯
)
=
∑
𝑣
∈
𝒯
𝑓
⁢
(
𝑣
)
.
	
Sequoia Dynamic Programing Algorithm.

The Sequoia tree construction algorithm finds the tree 
𝒯
 of size 
𝑁
 which maximizes 
𝐹
⁢
(
𝒯
)
, using dynamic programming. Our algorithm works by iteratively filling in the following 2-dimension tensor 
𝑇
:

	
𝑇
⁢
(
𝑛
,
𝑏
)
=
max
𝒯
,
|
𝒯
|
=
𝑛
,
FirstBranch
⁢
(
𝒯
)
=
𝑏
⁡
𝐹
⁢
(
𝒯
)
,
∀
  0
≤
𝑛
≤
𝑁
,
  0
≤
𝑏
≤
𝐵
.
		
(1)

Here, 
FirstBranch
⁢
(
𝒯
)
 denotes the number of direct children the root of 
𝒯
 has, and 
𝐵
 denotes an upper bound we impose on the number of direct children any node in the tree can have (we can let 
𝐵
=
𝑁
−
1
 to make this constraint vacuous). Given the tensor 
𝑇
, the maximum expected number of generated tokens for any tree of size 
𝑛
≤
𝑁
 can be found by searching over all possible first-branch values 
𝑏
: 
max
0
≤
𝑏
≤
𝐵
⁡
𝑇
⁢
[
𝑛
,
𝑏
]
.

We now show how to iteratively fill in the tensor 
𝑇
 (which we initialize to negative 
∞
). Pseudocode for the full dynamic programming method is shown in Algorithm 1).

As the base case, we set 
𝑇
⁢
[
1
,
0
]
=
1
, representing the tree composed of just the root node, because 1 token is generated per iteration of speculative decoding when no tokens are speculated.

For the recursive case, we can consider the tree composed of the root node and its first 
𝑏
−
1
 children and their descendants (tree #1), as well as the tree whose root is the last child of the root node and its descendants (tree #2). Letting 
𝑚
≥
1
 denote the number of nodes in tree #2, we can see that the expected number of generated tokens for tree #1 is 
𝑇
⁢
[
𝑛
−
𝑚
,
𝑏
−
1
]
. Furthermore, the expected number of generated tokens for tree #2 is 
max
0
≤
𝑗
≤
𝐵
⁡
𝑇
⁢
[
𝑚
,
𝑗
]
, but this sub-tree is only considered in the case where the 
𝑏
𝑡
⁢
ℎ
 child of the primary root node is accepted (which happens with probability 
𝑃
⁢
[
𝑏
]
). Therefore, we can compute 
𝑇
⁢
[
𝑛
,
𝑏
]
 by searching over all possible sizes 
𝑚
 for tree #2 to find the one which maximizes the expected number of generated tokens for the full tree:

	
𝑇
[
𝑛
,
𝑏
]
=
max
1
≤
𝑚
≤
𝑛
−
1
(
𝑇
[
𝑛
−
𝑚
,
𝑏
−
1
]
+
𝑃
[
𝑏
]
⋅
max
0
≤
𝑗
≤
𝐵
𝑇
[
𝑚
,
𝑗
]
)
)
.
	

We show in Appendix F.1.1 that by keeping track of the values of 
𝑚
 and 
𝑏
 that maximize the 
max
 expressions on lines 9 and 11, we can easily reconstruct the optimal tree 
𝒯
 of size 
𝑁
 (and 
FirstBranch
⁢
(
𝒯
)
≤
𝐵
) that attains the maximum expected number of generated tokens. We additionally demonstrate in this appendix (with python implementation) that we can extend this algorithm in a couple important ways:

• 

Bounded tree-depth: Because the amount of time it takes to speculate a token tree is proportional to the depth of the tree, it can be very beneficial to find the tree of depth 
≤
𝐷
 that maximizes the expected number of generated tokens. We demonstrate in Algorithm 4 that we can extend the Sequoia dynamic program to find the optimal tree of bounded depth.

• 

Compatibility with self-speculation: For self-speculation methods like Medusa [4], Eagle [25], and GLIDE [11] which leverage the target model’s representations on the current prefix during decoding, the acceptance rates can meaningfully degrade as you get deeper into the speculation tree (i.e., further away from the current prefix). We demonstrate in Algorithm 4 that it is simple to extend our Sequoia dynamic program to take as input a 2-D acceptance rate matrix (instead of a 1-D vector) containing the average acceptance rate vectors at different tree depths. Thus, Sequoia is compatible with the latest advances in self-speculation methods, which can attain meaningfully higher acceptance rates than “standalone” draft models.

This algorithm can be run offline, and thus does not slow down inference.

Algorithm 1 Sequoia Dynamic program
1:  Input: 
𝑁
 for the maximum tree size, 
𝐵
 for the maximum number of branches of any node. 
𝑃
⁢
[
1
]
,
𝑃
⁢
[
2
]
,
…
,
𝑃
⁢
[
𝐵
]
 for the probability of acceptance for each branch.
2:  Output: 
𝑇
⁢
[
𝑛
,
𝑏
]
∀
  0
≤
𝑛
≤
𝑁
,
  0
≤
𝑏
≤
𝐵
.
3:  Initialize array 
𝑇
, of size 
(
𝑁
+
1
,
𝐵
+
1
)
, with 
−
∞
 in all entries.
4:  Initialize array 
𝑇
𝑚
⁢
𝑎
⁢
𝑥
, of size 
(
𝑁
+
1
)
, with 
−
∞
 in all entries.
5:  
𝑇
⁢
[
1
,
  0
]
=
1
6:  
𝑇
𝑚
⁢
𝑎
⁢
𝑥
⁢
[
1
]
=
1
7:  for 
𝑛
=
2
→
𝑁
 do
8:     for 
𝑏
=
1
→
𝐵
 do
9:        
𝑇
⁢
[
𝑛
,
𝑏
]
=
max
1
≤
𝑚
≤
𝑛
−
1
⁡
(
𝑇
⁢
[
𝑛
−
𝑚
,
𝑏
−
1
]
+
𝑃
⁢
[
𝑏
]
⋅
𝑇
𝑚
⁢
𝑎
⁢
𝑥
⁢
[
𝑚
]
)
10:     end for
11:     
𝑇
𝑚
⁢
𝑎
⁢
𝑥
⁢
[
𝑛
]
=
max
0
≤
𝑏
≤
𝐵
⁡
𝑇
⁢
[
𝑛
,
𝑏
]
12:  end for
13:  Return array 
𝑇
3.1.2Theoretical Results

We now prove that the Sequoia tree construction algorithm scales well with the size of the speculated tree. In particular, we show that under certain assumptions on the acceptance rates of the verification algorithm, the number of generated tokens is lower-bounded by a function which is (roughly) logarithmic in the size of the tree. This is in contrast to existing tree construction algorithms, which are upper bounded in the expected number of tokens they generate, regardless of the size of the tree. For example, a single sequence of tokens has upper bound 
1
/
(
1
−
𝑃
1
)
 [24]; 
𝑘
 independent sequences can only increase this upper bound by 1, because they only increase the chance of acceptance of the first token. Even an infinitely deep binary tree is upper bounded by 
1
/
(
1
−
𝑃
2
)
.

We first define what it means for a verification algorithm to have a 
𝑏
 power-law acceptance rate, and then present our theorem on the scalability of Sequoia trees, under the assumption that the verification algorithm has a 
𝑏
 power-law acceptance rate.

Definition 3.5.

We say that a tree verification algorithm has a 
𝑏
 power-law acceptance rate if the chance 
𝑟
𝑘
 of the tree verification algorithm rejecting all 
𝑘
 speculated children of a node in a tree is upper bounded by a power-law of 
𝑘
 with exponent 
𝑏
—meaning, 
𝑟
𝑘
≤
1
/
𝑘
𝑏
 
∀
𝑘
∈
ℕ
, for 
𝑏
>
0
∈
ℝ
.

The above definition is motivated by our observation (Figure 3) that the Sequoia sampling/verification algorithm attains power-law acceptance rates in practice. We now state the theorem (proof in App. F.3).

Theorem 3.6.

Using a tree verification algorithm with a 
𝑏
 power-law acceptance rate, the expected number of tokens 
𝐺
⁢
(
𝑛
)
 generated by verifying the Sequoia tree of size 
𝑛
 is in 
Ω
(
𝑏
⋅
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
.

Figure 3:Rejection rate vs. number speculated tokens: We plot the average rejection rate (
1
−
𝑎
⁢
𝑐
⁢
𝑐
⁢
𝑒
⁢
𝑝
⁢
𝑡
⁢
𝑎
⁢
𝑛
⁢
𝑐
⁢
𝑒
⁢
_
⁢
𝑟
⁢
𝑎
⁢
𝑡
⁢
𝑒
) for the different verification algorithms, as a function of the number of speculated tokens 
𝑘
. Across temperature settings (
{
0.2
,
0.6
,
1.0
}
, left to right), the Sequoia verification algorithm attains the lowest rejection rates, and consistently has a power-law acceptance rate (Definition 3.5).
3.1.3Empirical Validation

In Figure 1, we plot the average number of tokens generated by Sequoia trees relative to various baseline tree structures, as a function of the number of tokens 
𝑛
 in the tree, using Pythia-2.8B as a draft model for Pythia-12B, and WikiText-103. We see that the number of generated tokens for Sequoia trees is unbounded—scaling roughly logarithmically with the tree size—whereas the other tree structures asymptote. We show results for more draft/target model pairs in Figure 6 in Appendix G.3.

3.2Tree sampling and verification

We present our token tree sampling and verification algorithm, and prove it is the first such algorithm to satisfy two important robustness properties, while maintaining the target model’s output distribution.

3.2.1Algorithm

We present the pseudocode for the Sequoia Tree sampling and verification algorithm in Algorithm 2. As discussed in Section 2, an important motivation for designing the Sequoia verification algorithm was the observation that SpecInfer and SpecTr both perform poorly at low temperatures, due to the fact that they can repeatedly sample (and then reject) a low-quality token that the draft model is confident in. Thus, we wanted to design an algorithm that would never make the same mistake twice—meaning, once a token was rejected, it would never propose that token again. Toward this end, Sequoia introduces two changes to the SpecInfer algorithm (shown in blue text in Algorithm 2): First, it performs sampling without replacement using the draft model distribution. Second, if all the tokens with non-zero draft model probability have already been sampled and rejected, it uses the uniform distribution over all tokens that have not yet been sampled as the new draft model distribution. These changes significantly improve the robustness of Sequoia relative to SpecInfer, while maintaining the guarantee that the output distribution is identical to that of the target model (proof in Appendix F.2.1).

Algorithm 2 Sequoia Sampling and Verification
(The blue lines [10-16] distinguish Sequoia’s sampling/verification from SpecInfer’s [28])
1:  Input: Prefix 
[
𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑛
−
1
]
, target model probabilities 
𝒫
(
⋅
|
𝑥
<
𝑛
)
, draft model probabilities 
𝒬
(
⋅
|
𝑥
<
𝑛
)
, and number of branches 
𝑘
≤
𝑣
⁢
𝑜
⁢
𝑐
⁢
𝑎
⁢
𝑏
⁢
_
⁢
𝑠
⁢
𝑖
⁢
𝑧
⁢
𝑒
.
2:  Output: A token 
𝑥
 sampled using Sequoia.
3:  Initialize residual 
𝑅
 with 
𝒫
, draft 
𝐷
 with 
𝒬
, and the set of rejected tokens 
𝑆
 with 
∅
4:  for 
𝑖
=
1
→
𝑘
 do
5:    sample 
𝑠
𝑖
∼
𝐷
, 
𝑟
𝑖
∼
Uniform
⁢
(
0
,
1
)
6:    if 
𝑟
𝑖
<
𝑅
⁢
[
𝑠
𝑖
]
𝐷
⁢
[
𝑠
𝑖
]
 then
7:       Return 
𝑠
𝑖
   # Accept 
𝑠
𝑖
8:    else
9:       
𝑅
←
norm
⁢
(
max
⁡
(
𝑅
−
𝐷
,
0
)
)
10:       
𝐷
⁢
[
𝑠
𝑖
]
←
0
11:       
𝑆
⁢
.add
⁢
(
𝑠
𝑖
)
12:       if sum
(
𝐷
)
=
0
 then
13:          # Let 
𝐷
 be uniform over non-rejected set
14:          
𝐷
⁢
[
𝑡
]
←
0
 if 
𝑡
∈
𝑆
, else 1
15:       end if
16:       
𝐷
←
norm
⁢
(
𝐷
)
17:    end if
18:  end for
19:  Return 
𝑥
∼
𝑅


3.2.2Theoretical Results

We now prove that the Sequoia verification algorithm is robust, in the sense that it satisfies both of the properties below, while existing verification algorithms do not.

• 

The optimal transport property: When 
𝑘
=
1
, the acceptance rate is equal to 
1
−
‖
𝑃
−
𝑄
‖
1
2
.2

• 

The cover property: If the support of the draft model probability distribution 
𝑄
 is of size 
𝑘
 and is a superset of the support of the target model probability distribution 
𝑃
, at most 
𝑘
 speculations will be needed to attain an acceptance rate of 1. Furthermore, if 
𝑘
 is equal to the vocabulary size, the acceptance rate should always be 1 as well, regardless of the draft model used.

Intuitively, satisfying the optimal transport property results in strong performance at high temperatures (because 
𝑃
 and 
𝑄
 will approach uniform distributions), while satisfying the cover property results in strong performance at low temperatures (if top target model token is in the top-
𝑘
 draft model tokens).

We now present our main robustness result (proof in Appendix F.3):

Theorem 3.7.

Sequoia verification satisfies both properties (optimal transport, cover); SpecInfer & SpecTr only satisfy the optimal transport property; top-
𝑘
 sampling only satisfies the cover property.

3.2.3Empirical Validation

In Figure 3, we plot the average rejection rates (equal to 
1
−
acceptance rates
) for the different verification algorithms, as a function of the number of speculated child tokens for a fixed token prefix, for various temperatures (0.2, 0.6, 1.0), measured on WikiText-103. We can see that across all temperature settings, the rejection rates for Sequoia decay faster than for the other algorithms. In general, we observe that the rejection rates 
𝑟
𝑘
 for Sequoia follow a power-law, where 
𝑟
𝑘
≈
1
/
𝑘
𝑏
 for some 
𝑏
>
0
. We can also see that while SpecTr and SpecInfer perform relatively well at high temperatures, they struggle at lower temperatures, and that the opposite is true for top-
𝑘
 sampling.

4Evaluation

In this section, we aim to demonstrate that Sequoia can speed up LLM inference by a large margin in wall-clock time. We first present our end-to-end system results showing total speedup, followed by validating our claims that Sequoia is scalable and robust.

• 

In Section 4.1, we demonstrate Sequoia’s superior end-to-end performance. Specifically, Sequoia achieves up-to 
4.04
×
 speed-up for Llama2-7B on A100 and 
9.5
×
 for Llama3-70B on L40 offloading (achieving the latency as low as 0.60 s/token).

• 

In Section 4.2.1, we show that the Sequoia tree can generate on average 33% more tokens than a tree of 16 independent sequences (tree size 512).

• 

In Section 4.2.2, show Sequoia’s sampling and verification algorithm is robust to temperature, consistently outperforming SpecInfer (by up to 
1.65
×
) and top-
𝑘
 sampling (by up to 
1.27
×
).

4.1End-to-end Results
Table 1:On-device results (A100): The optimal tree configuration and speedup for different pairs of draft and target models, and different temperatures, for Sequoia vs. SpecInfer. We specify the average number of generated tokens per decoding step in parentheses, next to the speedup factor. Sequoia attains up to 
4.04
×
 speedup on an A100. The speed of incremental decoding is 24.2ms/token with Huggingface. The draft model speed is 0.5ms/token. TBT refers to time between tokens.
Target LLM	Draft Model	T	Dataset	Tree Config.	Speedup	TBT	SpecInfer	Speedup
(size, depth)	ms/token	
5
×
8
	vs SpecInfer
Llama2-7B	JF68M	0	C4	(128,10)	4.04 
×
(5.08)	6.0	3.45
×
(3.96)	1.17
×

Llama2-7B	JF68M	0.6	C4	(128,7)	3.18
×
(3.92)	7.6	2.47
×
(2.97)	1.29
×

Llama2-7B	JF68M	0	OpenWebText	(128,7)	3.22
×
(3.86)	7.5	2.79
×
(3.15)	1.15
×

Llama2-7B	JF68M	0.6	OpenWebText	(128,6)	2.71
×
(3.33)	8.9	2.10
×
(2.54)	1.29
×

Llama2-7B	JF68M	0	CNN Daily	(128,7)	3.41
×
(4.05)	7.1	2.95
×
(3.27)	1.16
×

Llama2-7B	JF68M	0.6	CNN Daily	(128,6)	2.83
×
(3.45)	8.5	2.11
×
(2.58)	1.34
×

Llama2-7B	JF68M	0	MT Bench	(128,10)	4.03
×
(4.98)	6.0	3.84
×
(4.01)	1.05
×

Llama2-7B	JF68M	0.6	MT Bench	(128,7)	3.18
×
(3.96)	7.6	2.45
×
(2.97)	1.30
×
Table 2:Offloading results (L40): The optimal tree configuration and speedup for different pairs of draft and target models, and different temperatures, for Sequoia vs. SpecInfer. We specify the average number of generated tokens per decoding step in parentheses, next to the speedup factor. Sequoia attains up to 
9.5
×
 speedup in the offloading setting on an L40. The speed of incremental decoding is 5.7s/token with DeepSpeed Zero Inference. TBT refers to time between tokens.
Target LLM	Draft Model	T	Dataset	Tree Config.	Speedup	TBT	SpecInfer	Speedup
(size, depth)	s/token	
16
×
48
	vs SpecInfer
Llama2-70B-chat	Llama2-7B-chat	0	MT Bench	(768,18)	8.6
×
(10.30)	0.66	5.7
×
(7.63)	1.51
×

Llama2-70B-chat	Llama2-7B-chat	0.6	MT Bench	(768,18)	8.4
×
(9.91)	0.68	5.2
×
(7.03)	1.62
×

Llama3-70B-Instruct	Llama3-8B-Instruct	0	MT Bench	(768,18)	9.5
×
(11.68)	0.60	7.0
×
(9.07)	1.36
×

Llama3-70B-Instruct	Llama3-8B-Instruct	0.6	MT Bench	(768,18)	9.3
×
(11.37)	0.61	6.1
×
(8.29)	1.52
×

We now demonstrate that Sequoia speeds up LLM decoding in the on-device setting by up 
4.04
×
 on an A100 GPU, and up to 
9.5
×
 with offloading on an L40 GPU.

Setup.

Our experiments are based on Llama and Vicuna models. For the on-device setting, we use JackFram/Llama-68m (JF68m) [28] and princeton-nlp/Sheared-Llama-1.3B (SL1.3B) [46] as the draft models, and Llama2-7B [43], Llama2-13B, and Vicuna-33B [6] as the target models. For the offloading setting, we use Llama2-7B-chat/Llama3-8B-Instruct as the draft model and Llama2-70B-chat/Llama3-70B-Instruct as the target model. We evaluate our results on C4(en) [35] validation dataset, OpenWebText [14], CNN DailyMail [36] and MT Bench [52]. In each experiment, we use 200 examples to measure the acceptance rate vector (mentioned in Section 3.1) and sample another 200 examples for evaluation (50 for offloading). The prompt length and generation length are both set to 128 tokens except MT Bench. We evaluate Sequoia on different hardware including on-device experiments on L40 and A100(-PCIE 80GB) GPUs, as well as offloading experiments on an L40 GPU (with PCIE4). We also compare Sequoia with SpecInfer [28] with 
5
×
8
 trees (5 independent sequences of length 8, the tree structure used in [28] for batch size 1) for the on-device setting, and 
16
×
48
 trees for the offloading setting.

Implementation Details.

We implement the draft and target models using Transformers [45]. Because we determine the optimal tree structure in advance, we are able to use PyTorch CUDA graphs [31, 32] to reduce the overhead of kernel launching during speculative decoding. To accelerate sampling without replacement—which is not efficient in PyTorch 2.1 [32]—we use the exponential-sort algorithm [44], combined with PyTorch CUDA graphs [31, 32]. For offloading setting, we used an DeepSpeed-Zero-Inference [2] as baseline, which is 5.7 s/token.

Hardware-Aware Optimization.

For each hardware setting we consider in our experiments, we use the following method for selecting the size and depth of the Sequoia tree we should use to maximize speedups, while avoiding doing an exhaustive grid search. Letting 
𝐺
⁢
(
𝑛
,
𝑑
)
 denote the expected number of tokens generated by verifying the Sequoia tree of size 
𝑛
 and depth 
𝑑
 (computed via dynamic programming), 
𝑡
⁢
(
𝑛
)
 denote the (hardware-dependent) amount of time it takes the target model to verify 
𝑛
 tokens divided by the time to verify 1 token, and 
𝑐
 denote the (hardware-dependent) time to draft 
1
 token divided by the time to verify 1 token, the speedup attained by Sequoia can be expressed as 
Speedup
⁢
(
𝑛
,
𝑑
)
=
𝐺
⁢
(
𝑛
,
𝑑
)
𝑡
⁢
(
𝑛
)
+
𝑑
⋅
𝑐
. We measure 
𝑡
⁢
(
𝑛
)
 and 
𝑐
 empirically for each type of model and inference hardware, and then search over possible values of 
𝑛
, 
𝑑
 to find the pair that gives the largest speedup.

Main Results.

We evaluate Sequoia using different temperatures, draft and target model pairs, and hardware configurations. Results are shown in Table 1 (A100 on-device) and Table 2 (L40 offloading). We observe that Sequoia consistently speeds up LLM decoding in a wide range of settings. Sequoia reaches up to 
4.04
×
 speedup for the on-device setting, and up to 
9.5
×
 speedup for the offloading setting, as a result of the huge gap between computation capacity and memory bandwidth. Notably, for the offloading setting on L40, Sequoia can achieve as low as 0.60 s/token latency. We present additional on-device results (A100 and L40) in Appendix G.

Analysis.

We made several interesting observations on the interplay between Sequoia tree construction, sampling and verification, and hardware-aware optimizer. (1) Sequoia selects much larger trees in the offloading setting (768 tokens) than in the on-device setting (64 to 128 tokens). (2) In general, the average number of generated tokens is close to the wall-clock time speedup (especially when JF68M is used as the draft) as a result of the hardware-aware tree optimizer. (3) The optimal trees found by Sequoia for slightly different configurations—e.g., different temperatures and model pairs—can be very different from one another. (4) Sequoia chooses deeper trees at low temperature than high temperature, due to the acceptance rates being higher for low temperature.

4.2Ablations

We present our ablation experiments validating the scalability of the Sequoia tree construction algorithm (Section 4.2.1), and the robustness of Sequoia tree sampling and verification algorithm (Section 4.2.2). For each of these experiments, we only vary one element at a time (e.g., the tree structure for Section 4.2.1) to study the gains attained by each component of Sequoia.

Figure 4: Left: We compare the number of tokens generated on average by Sequoia trees vs. 
𝑘
 independent sequences, where we use Sequoia sampling and verification for both tree structures. Right: We compare the speedups attained by the Sequoia sampling and verification algorithm relative to SpecInfer and top-
𝑘
 sampling, across various temperatures, holding the tree structure fixed.
4.2.1The Scalability of Sequoia

In Figure 4 (left) we compare the average number of generated tokens for the Sequoia tree construction method, relative to 
𝑘
 independent sequences, at different budgets; we use Sequoia’s sampling and verification algorithm for all trees. The Sequoia tree is able to generate up to 
33
%
 more tokens per decoding step, demonstrating the effectiveness of Sequoia’s tree construction algorithm. Here, we use JackFram/Llama-68m as the draft model, Llama2-13B as the target model, 
0.6
 as the temperature, and CNN Daily Mail as the dataset.

4.2.2Robustness of Sequoia Sampling Algorithm

In Figure 4 (right) we compare the Sequoia sampling and verification algorithm to SpecInfer and top-
𝑘
 sampling across different temperature values, holding the tree structure fixed. We can see that Sequoia achieves the largest speedups across all temperatures, attaining up to 
1.65
×
 and 
1.27
×
 speedup relative to SpecInfer and top-
𝑘
 sampling, respectively. Here, we use JackFram/Llama-68m as the draft model, Llama2-7B as the target model, CNN Daily Mail as the dataset, and the corresponding Sequoia tree from Table 1 (temperature 
0.6
) as the tree structure. In Table 8 in Appendix G.4, we additionally show that the Sequoia sampling/verification algorithm is robust to the top-
𝑝
 parameter.

5Conclusion

We presented Sequoia, a scalable and robust speculative decoding method. By improving the topology of the token tree and the sampling algorithms, Sequoia is able to speed up autoregressive LLM inference up to 
4.04
×
 on GPU and 
9.5
×
 with offloading. In addition to providing real speedups, we believe Sequoia also provides insight into both the large potential and fundamental limits of speculative decoding systems. We hope that this understanding inspires future work in this area, or even informs the design of custom chips for LLM inference.

Acknowledgments

We thank Xinyu Yang, Harry Dong, Ranajoy Sadhukhan, Hanshi Sun, Silong Yong and the anonymous reviewers for their helpful discussions and feedback on the paper. This work was partially supported by the National Science Foundation under grant numbers CNS-2147909, CNS-2211882, and CNS-2239351, along with gift awards from Amazon, Cisco, Google, Intel, Li Auto, Meta, Moffet AI, Oracle, Qualcomm, and Samsung.

References
Ainslie et al. [2023]
↑
	Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Ontañón, Siddhartha Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo, James Lee-Thorp, Yi Tay, et al.Colt5: Faster long-range transformers with conditional computation.arXiv preprint arXiv:2303.09752, 2023.
Aminabadi et al. [2022]
↑
	Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, et al.Deepspeed inference: Enabling efficient inference of transformer models at unprecedented scale.arXiv preprint arXiv:2207.00032, 2022.
Brown et al. [2020]
↑
	Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.Language models are few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
Cai et al. [2024]
↑
	Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao.Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024.
Chen et al. [2023]
↑
	Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper.Accelerating large language model decoding with speculative sampling.CoRR, abs/2302.01318, 2023.doi: 10.48550/ARXIV.2302.01318.URL https://doi.org/10.48550/arXiv.2302.01318.
Chiang et al. [2023]
↑
	Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing.Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.URL https://lmsys.org/blog/2023-03-30-vicuna/.
Chowdhery et al. [2022]
↑
	Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.PaLM: Scaling language modeling with pathways.arXiv preprint arXiv:2204.02311, 2022.
Dao [2023]
↑
	Tri Dao.Flashattention-2: Faster attention with better parallelism and work partitioning.CoRR, abs/2307.08691, 2023.doi: 10.48550/ARXIV.2307.08691.URL https://doi.org/10.48550/arXiv.2307.08691.
Dao et al. [2022]
↑
	Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.Flashattention: Fast and memory-efficient exact attention with io-awareness.In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
Dettmers et al. [2022]
↑
	Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.Llm.int8(): 8-bit matrix multiplication for transformers at scale.CoRR, abs/2208.07339, 2022.doi: 10.48550/ARXIV.2208.07339.URL https://doi.org/10.48550/arXiv.2208.07339.
Du et al. [2024]
↑
	Cunxiao Du, Jing Jiang, Yuanchen Xu, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, and Yang You.Glide with a cape: A low-hassle method to accelerate speculative decoding.CoRR, abs/2402.02082, 2024.doi: 10.48550/ARXIV.2402.02082.URL https://doi.org/10.48550/arXiv.2402.02082.
Frantar and Alistarh [2023]
↑
	Elias Frantar and Dan Alistarh.Massive language models can be accurately pruned in one-shot.arXiv preprint arXiv:2301.00774, 2023.
Frantar et al. [2022]
↑
	Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh.GPTQ: accurate post-training quantization for generative pre-trained transformers.CoRR, abs/2210.17323, 2022.doi: 10.48550/ARXIV.2210.17323.URL https://doi.org/10.48550/arXiv.2210.17323.
Gokaslan and Cohen [2019]
↑
	Aaron Gokaslan and Vanya Cohen.Openwebtext corpus, 2019.
Gu and Dao [2023]
↑
	Albert Gu and Tri Dao.Mamba: Linear-time sequence modeling with selective state spaces.CoRR, abs/2312.00752, 2023.doi: 10.48550/ARXIV.2312.00752.URL https://doi.org/10.48550/arXiv.2312.00752.
Gu et al. [2022]
↑
	Albert Gu, Karan Goel, and Christopher Ré.Efficiently modeling long sequences with structured state spaces.In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.URL https://openreview.net/forum?id=uYLFoz1vlAC.
Han et al. [2015]
↑
	Song Han, Huizi Mao, and William J Dally.Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding.arXiv preprint arXiv:1510.00149, 2015.
Hinton et al. [2015]
↑
	Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al.Distilling the knowledge in a neural network.arXiv preprint arXiv:1503.02531, 2(7), 2015.
Hoefler et al. [2021]
↑
	Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste.Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks.J. Mach. Learn. Res., 22(241):1–124, 2021.
Jacob et al. [2018]
↑
	Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko.Quantization and training of neural networks for efficient integer-arithmetic-only inference.In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2704–2713, 2018.
Katharopoulos et al. [2020]
↑
	Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret.Transformers are rnns: Fast autoregressive transformers with linear attention.In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 2020.URL http://proceedings.mlr.press/v119/katharopoulos20a.html.
Kim et al. [2023]
↑
	Sehoon Kim, Karttikeya Mangalam, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer.Big little transformer decoder.CoRR, abs/2302.07863, 2023.doi: 10.48550/ARXIV.2302.07863.URL https://doi.org/10.48550/arXiv.2302.07863.
Kwon et al. [2023]
↑
	Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.Efficient memory management for large language model serving with pagedattention.In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors, Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611–626. ACM, 2023.doi: 10.1145/3600006.3613165.URL https://doi.org/10.1145/3600006.3613165.
Leviathan et al. [2023]
↑
	Yaniv Leviathan, Matan Kalman, and Yossi Matias.Fast inference from transformers via speculative decoding.In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.
Li et al. [2024]
↑
	Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang.Eagle: Speculative sampling requires rethinking feature uncertainty.In International Conference on Machine Learning, 2024.
Lin et al. [2023]
↑
	Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han.AWQ: activation-aware weight quantization for LLM compression and acceleration.CoRR, abs/2306.00978, 2023.doi: 10.48550/ARXIV.2306.00978.URL https://doi.org/10.48550/arXiv.2306.00978.
Liu et al. [2018]
↑
	Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell.Rethinking the value of network pruning.arXiv preprint arXiv:1810.05270, 2018.
Miao et al. [2023]
↑
	Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae Ying Yee Wong, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia.Specinfer: Accelerating generative llm serving with speculative inference and token tree verification.arXiv preprint arXiv:2305.09781, 2023.
Molchanov et al. [2016]
↑
	Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz.Pruning convolutional neural networks for resource efficient inference.arXiv preprint arXiv:1611.06440, 2016.
Nagel et al. [2019]
↑
	Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling.Data-free quantization through weight equalization and bias correction.In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1325–1334, 2019.
NVIDIA et al. [2020]
↑
	NVIDIA, Péter Vingelmann, and Frank H.P. Fitzek.Cuda, release: 10.2.89, 2020.URL https://developer.nvidia.com/cuda-toolkit.
Paszke et al. [2019]
↑
	Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.Pytorch: An imperative style, high-performance deep learning library.Advances in neural information processing systems, 32, 2019.
Pope et al. [2022a]
↑
	Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean.Efficiently scaling transformer inference.ArXiv, abs/2211.05102, 2022a.URL https://api.semanticscholar.org/CorpusID:253420623.
Pope et al. [2022b]
↑
	Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean.Efficiently scaling transformer inference.arXiv preprint arXiv:2211.05102, 2022b.
Raffel et al. [2019]
↑
	Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.Exploring the limits of transfer learning with a unified text-to-text transformer.arXiv e-prints, 2019.
See et al. [2017]
↑
	Abigail See, Peter J. Liu, and Christopher D. Manning.Get to the point: Summarization with pointer-generator networks.In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073–1083, Vancouver, Canada, July 2017. Association for Computational Linguistics.doi: 10.18653/v1/P17-1099.URL https://www.aclweb.org/anthology/P17-1099.
Sheng et al. [2023]
↑
	Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang.Flexgen: High-throughput generative inference of large language models with a single GPU.In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 31094–31116. PMLR, 2023.URL https://proceedings.mlr.press/v202/sheng23a.html.
Stern et al. [2018]
↑
	Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit.Blockwise parallel decoding for deep autoregressive models.Advances in Neural Information Processing Systems, 31, 2018.
Sun et al. [2023a]
↑
	Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter.A simple and effective pruning approach for large language models.arXiv preprint arXiv:2306.11695, 2023a.
Sun et al. [2023b]
↑
	Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix Yu.Spectr: Fast speculative decoding via optimal transport.arXiv preprint arXiv:2310.15141, 2023b.
Tang et al. [2019]
↑
	Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin.Distilling task-specific knowledge from bert into simple neural networks.arXiv preprint arXiv:1903.12136, 2019.
Touvron et al. [2021]
↑
	Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou.Training data-efficient image transformers & distillation through attention.In International Conference on Machine Learning, pages 10347–10357. PMLR, 2021.
Touvron et al. [2023]
↑
	Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom.Llama 2: Open foundation and fine-tuned chat models, 2023.
Vieira [2014]
↑
	Tim Vieira.Gumbel-max trick and weighted reservoir sampling, 2014.
Wolf et al. [2019]
↑
	Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.Huggingface’s transformers: State-of-the-art natural language processing.arXiv preprint arXiv:1910.03771, 2019.
Xia et al. [2023]
↑
	Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen.Sheared llama: Accelerating language model pre-training via structured pruning.arXiv preprint arXiv:2310.06694, 2023.
Xu et al. [2023]
↑
	Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and Xuanzhe Liu.Llmcad: Fast and scalable on-device large language model inference.arXiv preprint arXiv:2309.04255, 2023.
Yin et al. [2023]
↑
	Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, and Shiwei Liu.Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity.arXiv preprint arXiv:2310.05175, 2023.
Yu and Jeong [2022]
↑
	Gyeong-In Yu and Joo Seong Jeong.Orca: A distributed serving system for transformer-based generative models.In USENIX Symposium on Operating Systems Design and Implementation, 2022.URL https://api.semanticscholar.org/CorpusID:251734964.
Zhang et al. [2023]
↑
	Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra.Draft & verify: Lossless large language model acceleration via self-speculative decoding.CoRR, abs/2309.08168, 2023.doi: 10.48550/ARXIV.2309.08168.URL https://doi.org/10.48550/arXiv.2309.08168.
Zhao et al. [2019]
↑
	Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Chris De Sa, and Zhiru Zhang.Improving neural network quantization without retraining using outlier channel splitting.In International conference on machine learning, pages 7543–7552. PMLR, 2019.
Zheng et al. [2023]
↑
	Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
Zhou et al. [2023]
↑
	Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal.Distillspec: Improving speculative decoding via knowledge distillation.arXiv preprint arXiv:2310.08461, 2023.
Appendix ABroader Impacts

In this paper, we present a new algorithm for accelerating speculative decoding. While there are numerous application scenarios of large language models that warrant additional study regarding possible societal impact, we would like to highlight that our work does not advance the capabilities of these models. Our work is primarily an algorithmic study with no specific usage limitations, and while LLMs themselves can be used with malicious purpose, we believe that none of such use cases are specific to this paper.

Appendix BLimitations
Theoretical limitations:

On the theoretical front, there are two primary limitations to our results:

1. 

The positional acceptance assumption (Definition 3.1: The optimality of our dynamic program depends on this assumption. In particular, this assumption states that the only factor influencing the acceptance rate for a token is what “number child” it is to it’s “parent token” (e.g., if it is the first or fifth sampled token to follow the “parent” token). This allows us to model the acceptance dynamics using simple closed form equations, which ignore all contextual factors impacting acceptance rates (e.g., the current prefix, the confidence of the draft model, etc.).

2. 

The 
𝑏
 power law acceptance rate (Definition 3.5): While we observe in our experiments that Sequoia satisfies this assumption (see Figure 3), it’s important to note that need this assumption for our theoretical results on the scalability of Sequoia trees to hold (Theorem F.2).

Methodological limitations:

In terms of the limitations of Sequoia in practice, the most important limitation/challenge is likely that the structure of the optimal Sequoia tree depends on the exact (average) acceptance rate vector, which depends on the draft/target model pair, temperature value, data domain, etc. The optimal tree also depends on the batch size, which can be considered by the hardware-aware optimizer. It is relatively work-intensive to have to measure the acceptance rate vector for each setting, and use this vector to compute the optimal tree. In practice, we believe computing a single tree for a typical use case can work well for other use cases (e.g., higher/lower temperatures, different data domains), but we leave a more thorough analysis of this issue for future work.

Appendix CRelated Work

This work introduces a new algorithm in the family of speculative decoding methods that aims to maintain the exact output distribution of the target model by improving the structure and sampling/verification algorithm for the speculated token tree. There exist many other directions within this line of work—for example, methods which introduce leniency into the speculative decoding algorithm to attain increased speed at the cost of accuracy [22, 38], methods that reuse layers or representations from the target model as the draft model [50, 4], etc. Alternatively, the draft model can be distilled to better approximate the target model; DistillSpec [53, 18, 41, 42] improves that process by using model-generated data and adjusting the objective depending on the task and the decoding strategy. Finally, LLMCad [47] proposes an advanced algorithm for token tree generation and verification in the context of on-device LLM inference.

In addition to speculative decoding, there exist many other methods aimed at improving the speed of LLM inference. For example, model quantization is another very promising way of dealing with the I/O bottleneck during inference, by reducing the number of bits per parameter. However, unlike speculative decoding, these methods generally deteriorate the quality of the model to some degree, depending on the amount of quantization [17, 20, 30, 51, 26, 13, 10] or sparsity [29, 27, 19].

Meanwhile, various works [12, 39, 48, 1] have studied ways to improve LLM serving throughput. Pope et al. [33] investigated the batching effect in scaling up LLM. Orca [49] proposed a distributed LLM serving system that uses a finegrained scheduling policy to improve GPU utilization under various request lengths. vLLM [23] used page tables to manage GPU memory to increase memory utilization, which significantly boosts inference throughput. FlexGen [37] proposed an offloading mechanism to support larger batches to achieve high throughput.

FlashAttention [9, 8] is another algorithm that aims to improve the speed of LLMs (at both training and inference time) by considering the I/O cost of different operations.

Another promising approaching to speeding up inference is to change the fundamental building blocks of the model. Recently, numerous sub-quadratic architectures—including SSMs [16, 15] and linear attention models [21]—have been proposed. These models are particularly beneficial for long inputs.

Appendix DBackground: Sequence-based speculative decoding

The original speculative decoding method [24, 5] proposes using a small “draft model” to speculate 
𝛾
 tokens into the future, and then using the “target model” to in parallel process these tokens and decide which of the tokens to “accept”, in such a way that the output distribution of the target model is unchanged. This algorithm is presented in Algorithm 3.

Leviathan et al. [24] analyze the performance of this algorithm, presenting equations for the expected number of accepted tokens from one run of the algorithm, and the expected wall-clock speed up from using speculative decoding (relative to standard autoregressive inference with the target model). In this analysis, they introduce the acceptance rate 
𝛼
∈
[
0
,
1
]
, corresponding to the probability that a token 
𝑥
𝑖
 is accepted by Algorithm 3, under the simplifying assumption that the acceptance decisions are i.i.d.3 Under this assumption, they show that the expected number of generated tokens in each run of Algorithm 3 is 
1
−
𝛼
𝛾
+
1
1
−
𝛼
. Additionally, letting 
𝑐
 denote the ratio between the time to run the draft model and the time to run the target model, they show that the expected wall-clock speed-up from using this algorithm is 
1
−
𝛼
𝛾
+
1
(
1
−
𝛼
)
⁢
(
𝛾
⁢
𝑐
+
1
)
.

Algorithm 3 Sequence-based Speculative Decoding
1:  Input: Prefix 
[
𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑛
−
1
]
, Target model 
𝑀
𝑝
, draft model 
𝑀
𝑞
, and number of tokens 
𝛾
 to speculate.
2:  Output: A sequence of tokens generated using speculative decoding.
3:  for 
𝑖
=
𝑛
→
𝑛
+
𝛾
 - 1 do 
▷
 Sample sequence of 
𝛾
 tokens from draft model
4:     
𝑞
𝑖
⁢
(
𝑥
)
←
𝑀
𝑞
⁢
(
[
𝑥
1
,
…
,
𝑥
𝑖
−
1
]
)
5:     
𝑥
𝑖
∼
𝑞
𝑖
⁢
(
𝑥
)
6:  end for
7:  for 
𝑖
=
𝑛
→
𝑛
+
𝛾
 do 
▷
 For loop below can be run in parallel with a single forward pass of 
𝑀
𝑝
8:     
𝑝
𝑖
⁢
(
𝑥
)
←
𝑀
𝑞
⁢
(
[
𝑥
1
,
…
,
𝑥
𝑖
−
1
]
)
9:  end for
10:  
𝑠
←
𝑛
−
1
▷
 Choose how many tokens 
𝑛
 to accept
11:  for 
𝑖
=
𝑛
→
𝑛
+
𝛾
 - 1 do
12:     
𝑟
𝑖
∼
Uniform
⁢
(
0
,
1
)
13:     if 
𝑟
𝑖
<
𝑝
𝑖
⁢
(
𝑥
𝑖
)
𝑞
𝑖
⁢
(
𝑥
𝑖
)
 then
14:        
𝑠
←
𝑠
+
1
15:     else
16:        break
17:     end if
18:  end for
19:  
𝑝
′
⁢
(
𝑥
)
←
𝑝
𝑠
+
1
⁢
(
𝑥
)
20:  if 
𝑡
<
𝑛
+
𝛾
−
1
 then
21:     
𝑝
′
⁢
(
𝑥
)
←
norm
⁢
(
max
⁡
(
0
,
𝑝
𝑠
+
1
⁢
(
𝑥
)
−
𝑞
𝑠
+
1
⁢
(
𝑥
)
)
)
22:  end if
23:  
𝑡
∼
𝑝
′
⁢
(
𝑥
)
▷
 Sample a final token from 
𝑝
′
⁢
(
𝑥
)
24:  Return 
𝑥
1
,
…
,
𝑥
𝑠
,
𝑡
Appendix EExamples of Sequoia trees

Below we show more examples of Sequoia trees of various sizes. Note that for these plots we do not limit the depth of the tree. The acceptance rate vector we used for this (shown below) was computed with Llama3-70B-Instruct target model, Llama3-8B-Instruct draft model, on CNN daily news dataset:
[0.7732, 0.1039, 0.0402, 0.0206, 0.0128, 0.0081, 0.0064, 0.0043, 0.0035, 0.0026, 0.0025, 0.0021, 0.0016, 0.0014, 0.0010, 0.0010, 0.0010, 0.0007, 0.0007, 0.0006, 0.0007, 0.0006, 0.0004, 0.0004, 0.0005, 0.0006, 0.0004, 0.0003, 0.0002, 0.0004, 0.0001].

(a)8 node Sequoia tree
(b)16 node Sequoia tree
(c)32 node Sequoia tree
(d)64 node Sequoia tree
(e)128 node Sequoia tree
(f)256 node Sequoia tree
Figure 5:A set of increasingly large Sequoia trees.
Appendix FMethod details and theoretical results

We present additional details (as well as proofs for theorems) about the Sequoia tree construction (Section F.1) and tree sampling and verification (Section F.2) methods.

F.1Sequoia tree construction algorithm

We begin by presenting details about the Sequoia tree construction algorithm, and its corresponding theoretical properties.

F.1.1Sequoia dynamic program details

In this section, we present an extended version of the Sequoia tree construction dynamic programming (DP) algorithm (Algorithm 1), including a full python implementation of this extended algorithm (Algorithm 4). In Algorithm 1, we showed how to compute the expected number of generated tokens for the optimal tree of size 
𝑁
 (and branching factor 
≤
𝐵
). Here, we extend the algorithm to be able to handle:

1. 

An upper bound 
𝐷
 on the depth of the token tree, and

2. 

Self-speculation methods like Eagle [25] whose acceptance rates decay for tokens that are deeper in the speculated tree.

We then show how to additionally generate the optimal tree structure using dynamic programming, for these more general settings.

Extensions to bounded depth and self-speculation methods:

To handle the above cases, we assume that we have a 2-D array 
𝑝
, where 
𝑃
⁢
[
𝑑
,
𝑏
]
 is the probability of acceptance for a node at depth 
𝑑
 and branch number 
𝑏
. Here we assume 
𝑝
 is zero-indexed, so depth 
0
 corresponds to the direct children of the root node. We also assume 
𝑝
 has shape 
(
𝐷
−
 1
,
𝐵
+
1
)
, where 
𝐷
 is the limit on the depth of the speculated tree, and 
𝐵
 is the limit on the branch factor of the tree (max number of children per node). This allows us to infer that when we are computing 
𝑇
⁢
[
𝑛
,
𝑑
,
𝑏
]
 (during the internal running of the DP algorithm), in the case where the root node has a depth limit of 
𝐷
, the node being considered has depth limit 
𝑑
 it must be at depth 
𝐷
−
𝑑
; thus, 
𝐷
−
𝑑
 is the index of the 
𝑃
 array (at dimension 0) that should be used at that time. Using this fact, we can show that the recursion equation for Eagle (with bounded depth) is quite similar the one from Equation 1 (and Algorithm 1) in Section 2:

	
𝑇
[
𝑛
,
𝑑
,
𝑏
]
=
max
1
≤
𝑚
≤
𝑛
−
1
(
𝑇
[
𝑛
−
𝑚
,
𝑑
,
𝑏
−
1
]
+
𝑃
[
𝐷
−
𝑑
,
𝑏
]
⋅
max
0
≤
𝑗
≤
𝐵
𝑇
[
𝑚
,
𝑑
−
1
,
𝑗
]
)
)
	
	
∀
  2
≤
𝑛
≤
𝑁
,
    2
≤
𝑑
≤
𝐷
,
    2
≤
𝑏
≤
𝐵
.
	
Constructing the optimal tree structure:

In the python implementation below of the extended Sequoia DP algorithm (Algorithm 4), we show how to recursively construct the optimal tree structure for each tree size 
𝑛
 and depth limit 
𝑑
. Throughout the DP we maintain the following data structures:

• 

𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑛
⁢
𝑒
⁢
𝑤
⁢
_
⁢
𝑛
⁢
𝑜
⁢
𝑑
⁢
𝑒
⁢
[
𝑛
,
𝑑
,
𝑏
]
: A pointer to the root of the best sub-tree to add as the 
𝑏
𝑡
⁢
ℎ
 child of the tree root with budget n, depth <= d, and b children.

• 

𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑡
⁢
𝑟
⁢
𝑒
⁢
𝑒
⁢
[
𝑛
,
𝑑
]
: A pointer to root of the best tree with n nodes and depth <= d.

Line 32, and then lines 40-45, demonstrate the recursive relationship between these tree structures:

• 

If 
𝑚
∗
 is the optimal number of tokens that should be assigned to the tree rooted at the 
𝑏
𝑡
⁢
ℎ
 (and last) child for the tree (For the 
(
𝑛
,
𝑑
,
𝑏
)
 tree), then we can look up the optimal tree of that size in 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑡
⁢
𝑟
⁢
𝑒
⁢
𝑒
⁢
[
𝑚
∗
,
𝑑
−
1
]
, and set 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑛
⁢
𝑒
⁢
𝑤
⁢
_
⁢
𝑛
⁢
𝑜
⁢
𝑑
⁢
𝑒
⁢
[
𝑛
,
𝑑
,
𝑏
]
=
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑡
⁢
𝑟
⁢
𝑒
⁢
𝑒
⁢
[
𝑚
∗
,
𝑑
−
1
]
.

• 

If 
𝑏
∗
 is the optimal number of children for a tree of size 
𝑛
 and depth 
≤
𝑑
, we can look-up 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑛
⁢
𝑒
⁢
𝑤
⁢
_
⁢
𝑛
⁢
𝑜
⁢
𝑑
⁢
𝑒
⁢
[
𝑛
,
𝑑
,
𝑏
∗
]
 (the root of a tree of size 
𝑚
′
) and assign that as the last child of 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑡
⁢
𝑟
⁢
𝑒
⁢
𝑒
⁢
[
𝑛
,
𝑑
]
. To then find the optimal 
(
𝑏
−
1
)
𝑡
⁢
ℎ
 child of this tree, we can look-up 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑛
⁢
𝑒
⁢
𝑤
⁢
_
⁢
𝑛
⁢
𝑜
⁢
𝑑
⁢
𝑒
⁢
[
𝑛
−
𝑚
′
,
𝑑
,
𝑏
∗
−
1
]
, and we can continue in this manner until we have added all 
𝑏
∗
 children to 
𝑏
⁢
𝑒
⁢
𝑠
⁢
𝑡
⁢
_
⁢
𝑡
⁢
𝑟
⁢
𝑒
⁢
𝑒
⁢
[
𝑛
,
𝑑
]
.

This demonstrates how to build the optimal tree as part of the dynamic program.

Algorithm 4 Sequoia tree construction algorithm: Python implementation
import numpy as np
class Node:
def __init__(self, children=None):
self.children = children if children is not None else []
self.num_nodes_in_tree = 1 + sum(c.num_nodes_in_tree for c in self.children)
def sequoia_tree_construction(acc_rates, max_tree_size, max_tree_depth, max_branch):
P, N, D, B = acc_rates, max_tree_size, max_tree_depth, max_branch
if P.ndim == 1:
P = np.tile(P, (D - 1, 1))
assert P.shape == (D - 1, B + 1)
T = np.full(shape=(N + 1, D + 1, B + 1), fill_value=-float(’inf’))
T_max = np.full(shape=(N + 1, D + 1), fill_value=-float(’inf’))
T[1, 1:, 0] = 1.0
T_max[1, 1:] = 1.0
# best_new_node[n, d, b] = A pointer to the best node (tree root node) to add
# as the b^th child of the tree root with budget n, depth <= d, and b children.
# best_tree[n, d] = A pointer to root of the best tree with n nodes and depth <= d.
best_new_node = {(1, d, 0): None for d in range(1, D + 1)}
best_tree = {(1, d): Node() for d in range(1, D + 1)}
for n in range(2, N + 1):
for d in range(2, D + 1):
for b in range(1, B + 1):
x = np.nan_to_num(T[n - 1: 0: -1, d, b - 1] + P[D - d, b] * T_max[1: n, d - 1],
nan=0.0, neginf=-float(’inf’))
T[n, d, b] = np.max(x)
if T[n, d, b] > 0.0:
best_new_node[n, d, b] = best_tree[np.argmax(x) + 1, d - 1]
T_max[n, d] = np.max(T[n, d, :])
if T_max[n, d] > 0:
best_b = np.argmax(T[n, d, :])
best_n_budget_depth_d_tree_children = []
remaining_budget = n
# Find the ‘best_b‘ children of the root node, starting with the last.
for b in range(best_b, 0, -1):
next_child = best_new_node[remaining_budget, d, b]
best_n_budget_depth_d_tree_children.insert(0, next_child)
remaining_budget -= next_child.num_nodes_in_tree
assert remaining_budget == 1
best_tree[n, d] = Node(children=best_n_budget_depth_d_tree_children)
return T, best_tree
F.1.2Proof of Proposition 3.4: Closed-form expression for 
𝐹
⁢
(
𝒯
)

We now prove Proposition 3.4 by deriving the closed-form expression for 
𝐹
⁢
(
𝒯
)
 (the expected number of tokens generated by verifying tree 
𝒯
), and show how to use dynamic programming to find the optimal tree 
𝒯
 under a tree budget size.

Proposition F.1.

Let 
𝒯
 be a token tree that is verified with the positional acceptance assumption, and let 
𝑓
⁢
(
𝑣
)
 denote the score function for a node 
𝑣
∈
𝒯
. Then the expected number of tokens 
𝐹
⁢
(
𝒯
)
 generated by verifying 
𝒯
 is equal to

	
𝐹
⁢
(
𝒯
)
=
∑
𝑣
∈
𝒯
𝑓
⁢
(
𝑣
)
.
	
Proof.

Let 
𝐷
⁢
(
𝒯
)
 denote the expected number of tokens generated by verifying tree 
𝒯
. We would like to prove that 
𝐷
⁢
(
𝒯
)
=
𝐹
⁢
(
𝒯
)
⁢
∀
𝒯
. We will prove this by induction on the size of 
𝒯
.

Base case (
𝑁
=
1
):

A tree of size 1 is composed solely of the root node. By definition of the score function 
𝑓
⁢
(
𝑣
)
 (Definition 3.3), we know that 
𝑓
⁢
(
𝑣
)
=
1
 for the root node, so 
𝐹
⁢
(
𝒯
)
=
1
. 
𝐷
⁢
(
𝒯
)
=
1
 also, because verifying a tree composed of a root node with no children will simply sample from the target model, and generate 1 token.

Inductive step (
𝑁
>
1
):

For 
|
𝒯
|
=
𝑁
>
1
, let 
𝑣
 be a leaf of 
𝒯
 at child index 
𝑖
𝑣
 of depth 
𝑑
 with parent 
𝑣
𝑝
 and sibling 
𝒮
𝑣
 (set of sibling indices). We can then consider the tree 
𝒯
′
=
𝒯
−
{
𝑣
}
. Based on the inductive assumption, we know that 
𝑔
⁢
(
𝒯
′
)
=
𝐷
⁢
(
𝒯
′
)
. Using this assumption, we can express 
𝐷
⁢
(
𝒯
)
 in terms of 
𝐷
⁢
(
𝒯
′
)
:

	
𝐷
⁢
(
𝒯
)
	
=
𝐷
⁢
(
𝒯
′
)
−
(
𝑑
−
1
)
⋅
𝑓
⁢
(
𝑣
𝑝
)
⋅
(
1
−
∑
𝑖
∈
𝒮
𝑣
𝑝
𝑖
)
+
(
𝑑
−
1
)
⋅
𝑓
⁢
(
𝑣
𝑝
)
⋅
(
1
−
∑
𝑖
∈
𝒮
𝑣
∪
{
𝑖
𝑣
}
𝑝
𝑖
)
+
𝑑
⋅
𝑓
⁢
(
𝑣
)
	
		
=
𝐷
⁢
(
𝒯
′
)
−
(
𝑑
−
1
)
⁢
𝑓
⁢
(
𝑣
𝑝
)
⁢
𝑝
𝑖
𝑣
+
𝑑
⋅
𝑓
⁢
(
𝑣
)
	
		
=
∑
𝑣
′
∈
𝒯
′
𝑓
⁢
(
𝑣
′
)
−
(
𝑑
−
1
)
⁢
𝑓
⁢
(
𝑣
)
+
𝑑
⋅
𝑓
⁢
(
𝑣
)
	
		
=
𝐹
⁢
(
𝒯
′
)
+
𝑓
⁢
(
𝑣
)
	
		
=
𝐹
⁢
(
𝒯
)
	

Note that we use the inductive hypothesis, along with the fact the 
𝑓
⁢
(
𝑣
𝑝
)
⋅
𝑝
𝑖
𝑣
=
𝑓
⁢
(
𝑣
)
 (by definition of 
𝑓
⁢
(
𝑣
)
). ∎

F.1.3Proof of Theorem 3.6: Main scalability results for Sequoia trees

We now prove that, under certain assumptions on the acceptance rates of the tree verification algorithm, the expected number of tokens generated by verifying the Sequoia tree is lower bounded by a function which is roughly logarithmic in the size of the tree. We will do this by showing that a simpler tree—the 
𝑘
∗
⁢
(
𝑛
)
 tree (defined below)—also has this lower bound, and using the fact that the Sequoia tree is by construction the tree with the largest expected number of generated tokens.

We define the 
𝑘
∗
⁢
(
𝑛
)
 tree to be the 
𝑘
-ary tree4 with 
≤
𝑛
 nodes that has the highest expected accepted sequence length. Letting 
𝐺
⁢
(
𝑛
)
 denote the expected accepted sequence length for the 
𝑘
∗
⁢
(
𝑛
)
 tree, we will now prove that 
𝐺
(
𝑛
)
∈
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
 (meaning, it is lower-bounded by a scalar multiple of 
𝑏
⁢
log
⁡
(
𝑛
)
/
log
⁡
(
log
⁡
(
𝑛
)
)
), under the assumption that the rejection rate 
𝑟
𝑘
 is upper-bounded by a power-law of 
𝑘
. It then follows directly (as a corollary) that the growth rate of the tree generated by the Sequoia algorithm will also be in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
.

Theorem F.2.

Assume the chance 
𝑟
𝑘
 of a token tree verification algorithm rejecting all 
𝑘
 speculated tokens (
𝑘
 child nodes of some node in the tree) is upper bounded by a power-law of 
𝑘
; so 
𝑟
𝑘
≤
1
/
𝑘
𝑏
 for some 
𝑏
>
0
∈
ℝ
. Then the growth rate 
𝐺
⁢
(
𝑛
)
 for the 
𝑘
∗
⁢
(
𝑛
)
 tree is in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
.

Proof.

We will let 
𝑘
(
𝑛
)
=
⌊
log
(
𝑛
)
1
/
𝑏
⌋
 denote the branch-width chosen for tree size 
𝑛
, and show that under this assumption, the growth rate 
𝐺
′
⁢
(
𝑛
)
 of the corresponding 
𝑘
⁢
(
𝑛
)
-tree is at least 
𝑏
⁢
log
⁡
(
𝑛
)
10
log
(
log
(
𝑛
)
, assuming that 
𝑛
 is large enough. Given that 
𝐺
′
⁢
(
𝑛
)
 is a lower bound on 
𝐺
⁢
(
𝑛
)
 (because the above choice of 
𝑘
⁢
(
𝑛
)
 might not be fully optimal), and using the definition of 
Ω
, this proves that 
𝐺
(
𝑛
)
∈
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
. Note that we will abbreviate 
𝑘
⁢
(
𝑛
)
 as 
𝑘
 in many places throughout the proof, for brevity.

If we let 
𝑑
 denote the depth of the tree, the number of nodes in the tree is 
1
+
𝑘
+
𝑘
2
+
…
+
𝑘
𝑑
=
𝑘
𝑑
+
1
−
1
𝑘
−
1
≤
𝑛
. This implies 
𝑑
≤
log
𝑘
⁡
(
𝑛
)
, which we can prove as follows:

		
𝑘
𝑑
+
1
−
1
≤
𝑛
⁢
(
𝑘
−
1
)
	
	
⇒
	
𝑘
𝑑
+
1
≤
𝑛
⁢
𝑘
−
𝑛
+
1
≤
𝑛
⁢
𝑘
	
	
⇒
	
𝑑
+
1
≤
log
𝑘
⁡
(
𝑛
⁢
𝑘
)
=
log
𝑘
⁡
(
𝑛
)
+
1
	
	
⇒
	
𝑑
≤
log
𝑘
⁡
(
𝑛
)
	

We can assume 
𝑑
 is the largest integer such that 
𝑑
≤
log
𝑘
⁡
(
𝑛
)
, so it also follows that 
𝑑
+
1
≥
log
𝑘
⁡
(
𝑛
)
.

Letting 
𝛼
𝑘
≔
1
−
𝑟
𝑘
, the expected length 
𝐺
′
⁢
(
𝑛
)
 of the accepted token sequence can be expressed as 
1
⋅
(
1
−
𝛼
𝑘
)
+
2
⁢
𝛼
𝑘
⋅
(
1
−
𝛼
𝑘
)
+
3
⁢
𝛼
𝑘
2
⁢
(
1
−
𝛼
𝑘
)
+
…
+
(
𝑑
+
1
)
⁢
𝛼
𝑘
𝑑
=
1
+
𝛼
𝑘
+
𝛼
𝑘
2
+
…
+
𝛼
𝑘
𝑑
=
1
−
𝛼
𝑘
𝑑
+
1
1
−
𝛼
𝑘
 (the first equality is a result of telescoping sums, the second is from the sum of a finite geometric series). We will now lower bound this expression, making use of Lemma F.4 (defined and proven below).

	
𝐺
⁢
(
𝑛
)
≥
𝐺
′
⁢
(
𝑛
)
	
=
1
−
𝛼
𝑘
𝑑
+
1
1
−
𝛼
𝑘
	
		
=
1
−
(
1
−
𝑟
𝑘
)
𝑑
+
1
𝑟
𝑘
	
		
≥
𝑑
+
1
10
applying Lemma 
F.4
, and assuming 
⁢
𝑟
𝑘
⋅
(
𝑑
+
1
)
≤
log
⁡
(
1.9
)
	
		
≥
log
𝑘
⁡
(
𝑛
)
10
	
		
=
log
⁡
(
𝑛
)
10
⁢
log
⁡
(
𝑘
)
	
		
≤
log
⁡
(
𝑛
)
10
log
(
log
(
𝑛
)
1
/
𝑏
)
	
		
=
𝑏
⁢
log
⁡
(
𝑛
)
10
⁢
log
⁡
(
log
⁡
(
𝑛
)
)
	

Now we simply need to understand when 
𝑟
𝑘
⋅
(
𝑑
+
1
)
≤
log
⁡
(
1.9
)
:

	
𝑟
𝑘
⋅
(
𝑑
+
1
)
	
≤
1
𝑘
𝑏
⁢
(
log
𝑘
⁡
(
𝑛
)
+
1
)
	
		
≤
2
⁢
log
𝑘
⁡
(
𝑛
)
(
log
(
𝑛
)
1
/
𝑏
−
1
)
𝑏
using 
𝑘
(
𝑛
)
=
⌊
log
(
𝑛
)
1
/
𝑏
⌋
≥
log
(
𝑛
)
1
/
𝑏
−
1
	
		
≤
2
⁢
log
𝑘
⁡
(
𝑛
)
(
1
2
log
(
𝑛
)
1
/
𝑏
)
𝑏
assuming 
log
(
𝑛
)
1
/
𝑏
≥
2
⇔
𝑛
≥
exp
(
2
𝑏
)
	
		
=
2
𝑏
+
1
⁢
log
⁡
(
𝑛
)
log
⁡
(
𝑘
)
⁢
log
⁡
(
𝑛
)
	
		
=
2
𝑏
+
1
log
⁡
(
𝑘
)
	

So if 
2
𝑏
+
1
log
⁡
(
𝑘
)
≤
log
⁡
(
1.9
)
, then it follows that 
𝑟
𝑘
⋅
(
𝑑
+
1
)
≤
log
⁡
(
1.9
)
.

	
2
𝑏
+
1
log
⁡
(
𝑘
)
	
≤
log
⁡
(
1.9
)
⇔
2
𝑏
+
1
log
⁡
(
1.9
)
≤
log
⁡
(
𝑘
)
⇔
exp
⁡
(
2
𝑏
+
1
log
⁡
(
1.9
)
)
≤
𝑘
	

Given that 
𝑘
(
𝑛
)
=
⌊
log
(
𝑛
)
1
/
𝑏
⌋
≥
log
(
𝑛
)
1
/
𝑏
−
1
, we know that if 
log
(
𝑛
)
1
/
𝑏
−
1
≥
exp
(
2
𝑏
+
1
log
⁡
(
1.9
)
)
, then it must hold that 
𝑘
⁢
(
𝑛
)
≥
exp
⁡
(
2
𝑏
+
1
log
⁡
(
1.9
)
)
 as well. We can see that this holds if:

	
log
(
𝑛
)
1
/
𝑏
−
1
≥
exp
(
2
𝑏
+
1
log
⁡
(
1.9
)
)
⇔
𝑛
≥
exp
(
(
1
+
exp
(
2
𝑏
+
1
log
⁡
(
1.9
)
)
)
𝑏
)
	

Thus, we have shown that as long as 
𝑛
 is greater than the above expression, then 
𝐺
′
⁢
(
𝑛
)
≥
𝑏
⁢
log
⁡
(
𝑛
)
10
log
(
log
(
𝑛
)
. Because we know that 
𝐺
⁢
(
𝑛
)
≥
𝐺
′
⁢
(
𝑛
)
, this concludes the proof that 
𝐺
⁢
(
𝑛
)
 is in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
. ∎

We now prove, as a corollary of Theorem F.2, that the growth rate of the Sequoia tree is also in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
.

Corollary F.3.

Under the same assumptions on the rejection rates as Theorem F.2, it holds that the growth rate for the Sequoia tree is in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
.

Proof.

By construction, for every tree size 
𝑛
, the Sequoia tree is the tree that has the largest expected number of generated tokens. Thus, for every value of 
𝑛
 the expected number of generated tokens for the Sequoia tree must be larger than that of the 
𝑘
∗
⁢
(
𝑛
)
 tree, which was shown in Theorem F.2 to be in 
Ω
(
𝑏
log
(
𝑛
)
/
log
(
log
(
𝑛
)
)
. This concludes the proof. ∎

We now prove the lemma that we used to prove Theorem F.2:

Lemma F.4.

For any real number 
𝑥
∈
(
0
,
1
]
, and integer 
𝑚
>
0
 such that 
𝑚
⁢
𝑥
≤
log
⁡
(
1.9
)
, it holds that 
1
−
(
1
−
𝑥
)
𝑚
𝑥
≥
𝑚
10
.

Proof.
	
1
−
(
1
−
𝑥
)
𝑚
𝑥
	
=
1
−
(
1
−
𝑚
⁢
𝑥
+
(
𝑚
2
)
⁢
𝑥
2
−
(
𝑚
3
)
⁢
𝑥
3
+
(
𝑚
4
)
⁢
𝑥
4
−
…
+
(
−
1
)
𝑚
⁢
𝑥
𝑚
)
𝑥
	
		
=
𝑚
𝑥
−
(
𝑚
2
)
𝑥
2
+
(
𝑚
3
)
𝑥
3
−
(
𝑚
4
)
𝑥
4
+
…
−
(
−
1
)
𝑚
𝑥
𝑚
)
𝑥
	
		
=
𝑚
−
(
𝑚
2
)
⁢
𝑥
+
(
𝑚
3
)
⁢
𝑥
2
−
(
𝑚
4
)
⁢
𝑥
3
+
…
−
(
−
1
)
𝑚
⁢
𝑥
𝑚
−
1
	
		
≥
𝑚
−
(
𝑚
2
)
⁢
𝑥
−
(
𝑚
3
)
⁢
𝑥
2
−
(
𝑚
4
)
⁢
𝑥
3
−
…
−
𝑥
𝑚
−
1
	
		
≥
𝑚
−
𝑚
2
2
!
⁢
𝑥
−
𝑚
3
3
!
⁢
𝑥
2
−
𝑚
4
4
!
⁢
𝑥
3
−
…
	
		
=
𝑚
⁢
(
1
−
𝑚
⁢
𝑥
2
!
−
(
𝑚
⁢
𝑥
)
2
3
!
−
(
𝑚
⁢
𝑥
)
3
4
!
−
(
𝑚
⁢
𝑥
)
4
5
!
−
(
𝑚
⁢
𝑥
)
5
6
!
−
…
)
	
		
=
𝑚
⁢
(
2
−
1
−
𝑚
⁢
𝑥
2
!
−
(
𝑚
⁢
𝑥
)
2
3
!
−
(
𝑚
⁢
𝑥
)
3
4
!
−
(
𝑚
⁢
𝑥
)
4
5
!
−
(
𝑚
⁢
𝑥
)
5
6
!
−
…
)
	
		
=
𝑚
(
2
−
(
1
+
𝑚
⁢
𝑥
2
!
+
(
𝑚
⁢
𝑥
)
2
3
!
+
(
𝑚
⁢
𝑥
)
3
4
!
+
(
𝑚
⁢
𝑥
)
4
5
!
+
(
𝑚
⁢
𝑥
)
5
6
!
−
…
)
	
		
≥
𝑚
⁢
(
2
−
(
1
+
𝑚
⁢
𝑥
+
(
𝑚
⁢
𝑥
)
2
2
!
+
(
𝑚
⁢
𝑥
)
3
3
!
+
(
𝑚
⁢
𝑥
)
4
4
!
+
(
𝑚
⁢
𝑥
)
5
5
!
+
…
)
)
	
		
=
𝑚
⁢
(
2
−
𝑒
𝑚
⁢
𝑥
)
	
		
≥
𝑚
10
Assuming 
⁢
𝑒
𝑚
⁢
𝑥
≤
1.9
⁢
, which is true by our initial assumption.
	

∎

F.2Sequoia sampling and verification algorithm

We now move on to presenting proofs about the correctness and robustness of the Sequoia sampling and verification method.

F.2.1Proof of correctness for the Sequoia sampling and verification algorithm

We prove now that the Sequoia verification algorithm maintains the output distribution of the target model. We assume we have a target model 
𝑡
, and a list of draft models 
(
𝑑
1
,
…
⁢
𝑑
𝑛
,
𝑑
𝑛
+
1
,
…
)
, where 
𝑑
𝑖
 in this case depends on the previously rejected samples 
𝑥
1
,
…
,
𝑥
𝑖
−
1
, and where 
𝑑
𝑖
⁢
(
𝑢
)
 and 
𝑡
⁢
(
𝑢
)
 denote the probabilities of sampling token 
𝑢
∈
𝑉
 from 
𝑑
𝑖
 or 
𝑡
 respectively (where 
𝑉
 is the token vocabulary). We let 
𝑡
𝑖
 denote the residual at iteration 
𝑖
 of Sequoia loop, (after 
𝑖
−
1
 nodes have been rejected (so 
𝑡
1
=
𝑡
, as can be seen in Algorithm 2)

We will prove by induction on the number of proposed tokens 
𝑛
 that the Sequoia verification algorithm is correct.

Base case (
𝑛
=
0
): Sequoia is trivially correct, as it will simply sample from the residual 
𝑡
1
, which is equal to 
𝑡
.


Recursive case: We assume Sequoia is correct for 
𝑛
−
1
 proposed samples and prove it is correct for 
𝑛
 proposed samples.

We first show that at stage 
𝑖
 in the speculative decoding algorithm, the chance of Sequoia choosing to reject the proposed sample is equal to 
∑
𝑥
max
⁡
(
0
,
𝑡
𝑖
⁢
(
𝑥
)
−
𝑑
𝑖
⁢
(
𝑥
)
)
:

Lemma F.5.

P(No token accepted at iteration 
𝑖
) = 
∑
𝑥
max
⁡
(
0
,
𝑡
𝑖
⁢
(
𝑥
)
−
𝑑
𝑖
⁢
(
𝑥
)
)
.

Proof.
	
𝑃
⁢
(
No token accepted at iteration 
𝑖
)
	
=
∑
𝑥
𝑃
⁢
(
sample 
𝑥
)
⋅
𝑃
⁢
(
reject 
𝑥
|
𝑥
 is sampled
)
	
		
=
∑
𝑥
𝑑
𝑖
⁢
(
𝑥
)
⋅
(
1
−
min
⁡
(
𝑡
𝑖
⁢
(
𝑥
)
𝑑
𝑖
⁢
(
𝑥
)
,
1
)
)
	
		
=
∑
𝑥
𝑑
𝑖
⁢
(
𝑥
)
−
∑
𝑥
min
⁡
(
𝑡
𝑖
⁢
(
𝑥
)
,
𝑑
𝑖
⁢
(
𝑥
)
)
	
		
=
∑
𝑥
𝑡
𝑖
⁢
(
𝑥
)
−
∑
𝑥
min
⁡
(
𝑡
𝑖
⁢
(
𝑥
)
,
𝑑
𝑖
⁢
(
𝑥
)
)
	
		
=
∑
𝑥
𝑡
𝑖
⁢
(
𝑥
)
+
max
⁡
(
−
𝑡
𝑖
⁢
(
𝑥
)
,
−
𝑑
𝑖
⁢
(
𝑥
)
)
	
		
=
∑
𝑥
𝑡
𝑖
⁢
(
𝑥
)
−
𝑡
𝑖
⁢
(
𝑥
)
+
max
⁡
(
0
,
𝑡
𝑖
⁢
(
𝑥
)
−
𝑑
𝑖
⁢
(
𝑥
)
)
	
		
=
∑
𝑥
max
⁡
(
0
,
𝑡
𝑖
⁢
(
𝑥
)
−
𝑑
𝑖
⁢
(
𝑥
)
)
	

∎

We are now ready to prove the recursive case of the Sequoia algorithm. By the inductive hypothesis, we know that for all 
𝑢
∈
𝑉
,

	
𝑡
⁢
(
𝑢
)
=
𝑃
⁢
(
𝑢
 accepted in first 
𝑛
−
1
 iterations
)
+
𝑃
⁢
(
No token accepted in first 
𝑛
−
1
 iterations
)
⋅
𝑡
𝑛
⁢
(
𝑢
)
	

What this means is that in the case where we run Sequoia for 
𝑛
−
1
 iterations (and if no token is accepted we sample from the residual 
𝑡
𝑛
), this is equivalent to sampling from the target distribution 
𝑡
 directly. We would like to show that this output distribution is equivalent to the one we would get if we run Sequoia for 
𝑛
 iterations (and if no token is accepted we sample from the residual 
𝑡
𝑛
+
1
). The output distribution of this scenario can be written as follows:

	
𝑃
(
𝑢
 accepted in first 
𝑛
−
1
 iterations
)
+
𝑃
(
No token accepted in first 
𝑛
−
1
 iterations
)
⋅
	
	
(
𝑑
𝑛
⁢
(
𝑢
)
⋅
𝑃
⁢
(
𝑢
 accepted at iteration 
𝑛
)
+
𝑃
⁢
(
No token accepted in iteration 
𝑛
)
⋅
𝑡
𝑛
+
1
⁢
(
𝑢
)
)
	

Thus, all we must show is that

	
𝑡
𝑛
⁢
(
𝑢
)
=
𝑑
𝑛
⁢
(
𝑢
)
⋅
𝑃
⁢
(
𝑢
 accepted at iteration 
𝑛
)
+
𝑃
⁢
(
No token accepted in iteration 
𝑛
)
⋅
𝑡
𝑛
+
1
⁢
(
𝑢
)
	



We now show this desired result. We will use Lemma F.5, and the fact that by definition of the SpecInfer algorithm (see Algorithm 2, ignoring blue lines), we know that 
𝑡
𝑛
+
1
⁢
(
𝑢
)
=
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
)
∑
𝑥
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑥
)
−
𝑑
𝑛
⁢
(
𝑥
)
)
.

	
𝑑
𝑛
⁢
(
𝑢
)
⋅
𝑃
⁢
(
𝑢
 accepted at iteration 
𝑛
)
+
𝑃
⁢
(
No token accepted in iteration 
𝑛
)
⋅
𝑡
𝑛
+
1
⁢
(
𝑢
)
	
	
=
𝑑
𝑛
⁢
(
𝑢
)
⋅
min
⁡
(
1
,
𝑡
𝑛
⁢
(
𝑢
)
𝑑
𝑛
⁢
(
𝑢
)
)
+
(
∑
𝑥
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑥
)
−
𝑑
𝑛
⁢
(
𝑥
)
)
)
⁢
𝑡
𝑛
+
1
⁢
(
𝑢
)
	
	
=
min
⁡
(
𝑑
𝑛
⁢
(
𝑢
)
,
𝑡
𝑛
⁢
(
𝑢
)
)
+
(
∑
𝑥
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑥
)
−
𝑑
𝑛
⁢
(
𝑥
)
)
)
⋅
(
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
)
∑
𝑥
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑥
)
−
𝑑
𝑛
⁢
(
𝑥
)
)
)
	
	
=
min
⁡
(
𝑑
𝑛
⁢
(
𝑢
)
,
𝑡
𝑛
⁢
(
𝑢
)
)
+
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
)
	
	
=
𝑡
𝑛
⁢
(
𝑢
)
	

To see that this last equality holds, we consider two cases:

1. 

Case 1 
(
𝑡
𝑛
⁢
(
𝑢
)
≥
𝑑
𝑛
⁢
(
𝑢
)
)
: 
min
⁡
(
𝑑
𝑛
⁢
(
𝑢
)
,
𝑡
𝑛
⁢
(
𝑢
)
)
+
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
)
=
𝑑
𝑛
⁢
(
𝑢
)
+
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
=
𝑡
𝑛
⁢
(
𝑢
)
.

2. 

Case 1 
(
𝑡
𝑛
⁢
(
𝑢
)
<
𝑑
𝑛
⁢
(
𝑢
)
)
: 
min
⁡
(
𝑑
𝑛
⁢
(
𝑢
)
,
𝑡
𝑛
⁢
(
𝑢
)
)
+
max
⁡
(
0
,
𝑡
𝑛
⁢
(
𝑢
)
−
𝑑
𝑛
⁢
(
𝑢
)
)
=
𝑡
𝑛
⁢
(
𝑢
)
+
0
=
𝑡
𝑛
⁢
(
𝑢
)
.

This completes the proof.

F.3Proof of Theorem 3.7: Main robustness result for Sequoia sampling and verification

We now prove the robustness results for the Sequoia verification algorithm.

Theorem F.6.

The Sequoia verification algorithm satisfies both the optimal transport and the cover properties, while SpecInfer and SpecTr only satisfy the optimal transport property, and (top-
𝑘
) naive sampling only satisfies the cover property.

Proof.

This proof is quite straightforward:

• 

Sequoia satisfies the optimal transport property: It is clear that Sequoia satisfies the optimal transport property, because at 
𝑘
=
1
, it is identical to the original speculative decoding algorithm [24].

• 

Sequoia satisfies the cover property: To see why Sequoia satisfies the cover property, we will use the following two facts:

– 

If the support of 
𝑄
 is of size 
𝑘
 and 
𝑘
 tokens are speculated by the draft model, the set of speculated tokens will always exactly equal the 
𝑘
 tokens in the support of 
𝑄
 (because Sequoia does sampling without replacement from the draft model).

– 

During the verification for-loop in Algorithm 2, the support of the residual will always be contained in the support of 
𝑃
 intersected with the set of tokens that have not yet been rejected. This is because the support of the residual can never grow (because 
𝑝
𝑖
⁢
(
𝑥
)
=
0
⇒
𝑝
𝑖
+
1
⁢
(
𝑥
)
=
𝑛
⁢
𝑜
⁢
𝑟
⁢
𝑚
⁢
(
𝑚
⁢
𝑎
⁢
𝑥
⁢
(
𝑝
𝑖
−
𝑞
𝑖
,
0
)
)
⁢
(
𝑥
)
=
0
, where 
𝑝
𝑖
 and 
𝑞
𝑖
 denote the residual and draft probabilities at iteration 
𝑖
, respectively), and because if a token 
𝑥
 is rejected it will “exit” the residual (because 
𝑥
 is rejected implies 
𝑞
𝑖
⁢
(
𝑥
)
>
𝑝
𝑖
⁢
(
𝑥
)
 which implies that 
𝑝
𝑖
+
1
⁢
(
𝑥
)
=
𝑛
⁢
𝑜
⁢
𝑟
⁢
𝑚
⁢
(
𝑚
⁢
𝑎
⁢
𝑥
⁢
(
𝑝
𝑖
−
𝑞
𝑖
,
0
)
)
⁢
(
𝑥
)
=
0
).

Combining these two facts, we can see that if the first 
𝑘
−
1
 tokens were rejected, then the 
𝑘
𝑡
⁢
ℎ
 token must be accepted, because the residual must be a one-hot vector with probability 1 at the only remaining token, and the (updated) draft probabilities will also be this same one-hot vector (and thus, accepted with probability 1). Additionally, we can see that if 
𝑉
 tokens are sampled (where 
𝑉
 is the vocab size), these must exactly equal the 
𝑉
 tokens in the vocabulary, and thus one of those tokens must be accepted. In the case where the support of 
𝑄
 is equal to the full vocabulary, this result follows directly from the discussion above. In the case where the support of 
𝑄
 does not equal the full vocabulary, this is a result of the fact that once all tokens in the support of 
𝑄
 have been sampled and rejected, we begin sampling (without replacement) from the uniform distribution over all non-rejected tokens.

• 

SpecInfer satisfies the optimal transport property: For 
𝑘
=
1
, SpecInfer is identical to the original speculative decoding algorithm [24].

• 

SpecInfer does not satisfy the cover property: It is easy to see that SpecInfer does not satisfy the cover property, with the following counter-example. Let 
𝑄
=
[
0.5
,
0.5
]
 and 
𝑃
=
[
1.0
,
0
]
. We can see that the support of 
𝑄
 is of size 2 and contains the support of 
𝑃
. But with probability 25%, SpecInfer will sample the second token twice in a row, and will reject both of them.

• 

SpecTr satisfies the optimal transport property: For 
𝑘
=
1
, SpecTr is identical to the original speculative decoding algorithm [24], because 
𝛾
=
1
 by definition.

• 

SpecTr does not satisfies the cover property: We can show that SpecTr (in particular, the ‘
𝑘
-sequential selection’ algorithm from [40]) does not satisfy the cover property, with the following counter-example. Let 
𝑃
=
[
1
,
0
]
 and 
𝑄
=
[
0.5
,
0.5
]
. Then 
𝛽
𝑝
,
𝑞
⁢
(
𝛾
)
=
∑
𝑥
=
0
1
min
⁡
(
𝑄
⁢
(
𝑥
)
,
𝑃
⁢
(
𝑥
)
/
𝛾
)
=
min
⁡
(
0.5
,
1
/
𝛾
)
+
min
⁡
(
0.5
,
0
/
𝛾
)
=
0.5
 (because 
𝛾
∈
[
1
,
2
]
 by assumption). We know the acceptance rate of SpecTr is 
1
−
(
1
−
𝛽
𝑝
,
𝑞
⁢
(
𝛾
)
)
2
=
1
−
(
1
−
0.5
)
2
=
0.75
≠
1
. Thus, SpecTr does not satisfy the cover property.

• 

Top-
𝑘
 naive sampling does not satisfy the optimal transport property: Letting 
𝑄
=
[
0.6
,
0.4
]
 and 
𝑃
=
[
0.6
,
0.4
]
, we can see that top-
𝑘
 naive sampling will accept with probability 0.6, whereas 
1
−
‖
𝑃
−
𝑄
‖
/
2
=
1.0
.

• 

Top-
𝑘
 naive sampling satisfies the cover property: It’s easy to see that if the support of 
𝑄
 is of size 
𝑘
 and contains the support of 
𝑃
, then top-
𝑘
 naive sampling will always accept (because it will sample from the target model and accept if the sampled token is among the top-
𝑘
 tokens according to the draft model). Similarly, if 
𝑘
=
𝑉
, it must accept as well (because the top-
𝑉
 tokens must be the full vocabulary, and so any sample from the target model must accept).

∎

Appendix GAdditional Experiments
G.1Additional end-to-end speedup results

We provide additional end-to-end results comparing Sequoia to baselines, extending the results from Section 4.1. Here (Tables 3 and 4), we provide on-device results on A100 and L40 GPUs, for a more extended set of models, relative to the results in Table 1, but on different hardware.

Table 3:On-device results (A100): The optimal tree configuration and speedup for different pairs of draft and target models, and different temperatures, for Sequoia vs. SpecInfer. We specify the average number of generated tokens per decoding step in parentheses, next to the speedup factor. Sequoia attains up to 
4.04
×
 speedup on an A100. TBT refers to time between tokens.
Target LLM	Draft Model	T	Dataset	Tree Config.	Speedup	TBT	SpecInfer
(size, depth)	ms/token	
5
×
8

Llama2-7B	JF68M	0	C4	(128,10)	4.04 
×
(5.08)	6.0	3.45
×
(3.96)
Llama2-7B	JF68M	0.6	C4	(128,7)	3.18
×
(3.92)	7.6	2.47
×
(2.97)
Llama2-7B	JF68M	0	OpenWebText	(128,7)	3.22
×
(3.86)	7.5	2.79
×
(3.15)
Llama2-7B	JF68M	0.6	OpenWebText	(128,6)	2.71
×
(3.33)	8.9	2.10
×
(2.54)
Llama2-7B	JF68M	0	CNN Daily	(128,7)	3.41
×
(4.05)	7.1	2.95
×
(3.27)
Llama2-7B	JF68M	0.6	CNN Daily	(128,6)	2.83
×
(3.45)	8.5	2.11
×
(2.58)
Llama2-13B	JF68M	0	C4	(64,9)	3.73
×
(4.20)	8.4	3.30
×
(3.64)
Llama2-13B	JF68M	0.6	C4	(64,7)	3.19
×
(3.57)	9.8	2.48
×
(2.87)
Llama2-13B	JF68M	0	OpenWebText	(64,7)	3.18
×
(3.49)	9.8	2.77
×
(3.05)
Llama2-13B	JF68M	0.6	OpenWebText	(64,6)	2.77
×
(3.06)	11.3	2.17
×
(2.49)
Llama2-13B	JF68M	0	CNN Daily	(64,7)	3.33
×
(3.68)	9.4	2.95
×
(3.22)
Llama2-13B	JF68M	0.6	CNN Daily	(64,6)	2.88
×
(3.17)	10.8	2.17
×
(2.54)
Vicuna-33B	SL1.3B	0	C4	(64,6)	2.27
×
(4.28)	23.4	1.83
×
(3.86)
Vicuna-33B	SL1.3B	0.6	C4	(64,6)	2.19
×
(4.16)	24.3	1.64
×
(3.53)
Vicuna-33B	SL1.3B	0	OpenWebText	(64,5)	2.21
×
(3.93)	24.1	1.75
×
(3.70)
Vicuna-33B	SL1.3B	0.6	OpenWebText	(64,5)	2.13
×
(3.82)	25.0	1.57
×
(3.36)
Vicuna-33B	SL1.3B	0	CNN Daily	(64,5)	2.21
×
(3.93)	24.1	1.75
×
(3.71)
Vicuna-33B	SL1.3B	0.6	CNN Daily	(64,5)	2.16
×
(3.86)	24.6	1.58
×
(3.40)
Table 4:on-device results (L40): The optimal tree configuration and speedup for different pairs of draft and target models, and different temperatures, for Sequoia vs. SpecInfer. We specify the average number of generated tokens per decoding step in parentheses, next to the speedup factor. Sequoia attains up to 
3.95
×
 speedup on an L40.
Target LLM	Draft Model	T	Dataset	Tree Config.	Speedup	SpecInfer
(size, depth)	
5
×
8

Llama2-7B	JF68M	0	C4	(64,10)	3.95
×
(4.68)	3.50
×
(3.98)
Llama2-7B	JF68M	0.6	C4	(64,7)	3.10
×
(3.63)	2.28
×
(2.89)
Llama2-7B	JF68M	0	OpenWebText	(64,7)	3.12
×
(3.58)	2.79
×
(3.16)
Llama2-7B	JF68M	0.6	OpenWebText	(64,6)	2.68
×
(3.12)	2.08
×
(2.54)
Llama2-7B	JF68M	0	CNN Daily	(64,7)	3.30
×
(3.79)	2.89
×
(3.28)
Llama2-7B	JF68M	0.6	CNN Daily	(64,6)	2.81
×
(3.27)	2.09
×
(2.59)
Llama2-13B	JF68M	0	C4	(64,10)	3.15
×
(4.25)	2.76
×
(3.61)
Llama2-13B	JF68M	0.6	C4	(64,8)	2.62
×
(3.57)	2.06
×
 (2.81)
Llama2-13B	JF68M	0	OpenWebText	(64,8)	2.64
×
(3.52)	2.34
×
(3.05)
Llama2-13B	JF68M	0.6	OpenWebText	(64,6)	2.28
×
(3.07)	1.79
×
(2.44)
Llama2-13B	JF68M	0	CNN Daily	(64,7)	2.78
×
(3.68)	2.47
×
(3.21)
Llama2-13B	JF68M	0.6	CNN Daily	(64,7)	2.37
×
(3.22)	1.85
×
(2.51)
G.2More Comparisons with SpecInfer

To demonstrate the optimality of Sequoia’s tree construction, we provide a sweep of tree configurations and corresponding speedups of SpecInfer in Tables 5 and 6. Sequoia attains better speedups in both greedy decoding and stochastic decoding than all tree configurations of SpecInfer.

Table 5:A sweep of tree configurations and their corresponding speedups of SpecInfer [28] on A100. The draft model is JF68M, and the target model is Llama2-7B in greedy decoding. The evaluated dataset is C4. The default tree configuration in SpecInfer is 
5
×
8
, which brings 3.45
×
 speedup while Sequoia achieves 4.04
×
 speedup, surpassing all tree configurations below.
Width/Depth	1	2	4	8	16	32	64	128
1				3.09
×
	3.14
×
	2.75
×
	1.94
×
	1.19
×

2			2.95
×
	3.36
×
	3.46
×
	2.69
×
	1.74
×
	
4		2.4
×
	3.14
×
	3.46
×
	3.41
×
	2.47
×
		
8	1.88
×
	2.44
×
	3.14
×
	3.70
×
	3.03
×
			
16	2.00
×
	2.55
×
	3.27
×
	3.14
×
				
32	1.86
×
	2.57
×
	2.81
×
					
64	1.92
×
	2.22
×
						
128	1.68
×
							
Table 6:A sweep of tree configurations and their corresponding speedups of SpecInfer [28] on A100. The draft model is JF68M, and the target model is Llama2-7B in stochastic decoding. The evaluated dataset is C4. The default tree configuration in SpecInfer is 
5
×
8
, which brings 2.47
×
 speedup while Sequoia achieves 3.18
×
 speedup, surpassing all tree configurations below.
Width/Depth	1	2	4	8	16	32	64	128
1				2.08
×
	1.87
×
	1.48
×
	1.11
×
	0.69
×

2			2.14
×
	2.2
×
	1.89
×
	1.46
×
	1.07
×
	
4		1.99
×
	2.3
×
	2.28
×
	1.95
×
	1.53
×
		
8	1.73
×
	2.09
×
	2.42
×
	2.42
×
	2.14
×
			
16	1.78
×
	2.07
×
	2.41
×
	2.18
×
				
32	1.78
×
	2.08
×
	2.24
×
					
64	1.73
×
	2.04
×
						
128	1.61
×
							
Table 7:A sweep of tree configurations and their corresponding speedups of SpecInfer [28] on L40 offloading setting. The draft model is Llama2-7B-chat, and the target model is Llama2-70B-chat in stochastic decoding. The evaluated dataset is MT-Bench. Sequoia achieves 8.4
×
 speedup, surpassing all tree configurations below.
Tree Config.	(16,48)	(24,32)	(32,24)
Speedup	5.2
×
	5.3
×
	5.5
×
G.3Scalability Additional Results

Here we present additional results demonstrating the scalability of the Sequoia tree construction algorithm relative to baselines, for several Pythia draft and target model pairs on the WikiText-103 dataset:

Figure 6:Number of generated tokens vs. tree size: We plot the average number of tokens generated for different tree structures per decoding step of the target model, as a function of the tree size, for different draft and target model pairs. The number of generated tokens for Sequoia trees continues to grow with the tree size, while other tree structures asymptote.
G.4Robustness Additional Results

See Table 8.

Table 8:We compare the robustness of the Sequoia sampling and verification algorithm to the top-
𝑝
 hyperparameter, relative to SpecInfer and top-
𝑘
 sampling. We present total speedups on an A100 GPU for the different methods (number of generated tokens in parentheses). We hold the tree structure fixed across methods, use JF68M as the draft model, and Llama2-7B as the target model.
Top-
𝑝
	Sequoia (Ours)	SpecInfer	top-
𝑘
 sampling
0.8	
2.54
×
(3.18)	
2.35
×
(2.93)	
2.43
×
(2.90)
0.9	
2.61
×
(3.27)	
2.42
×
(3.01)	
2.27
×
(2.71)
1.0	
2.69
×
(3.26)	
2.55
×
(3.10)	
2.12
×
(2.44)
G.5Evaluation of Sequoia hardware-aware optimizer

In this section, we demonstrate the effectiveness of the Sequoia hardware-aware tree optimizer. We compare the speedups attained by the Sequoia trees of various sizes from Figure 4 (left) to the trees selected by the hardware-aware tree-optimizer. Because the tree optimizer is able to limit the tree depth to make speculation faster, it is able to attain larger end-to-end speedups than any of the Sequoia trees from Figure 4 (left), whose structures were chosen to maximize the expected number of generated tokens (not the speedup). The optimizer is also able to automatically find the tree size that produces the largest overall speedup.

Figure 7:We compare the wall-clock time speedup of Sequoia trees of various sizes (orange lines)—chosen to maximize the # generated tokens—with the speedup of the trees selected by the hardware-aware tree optimizer (horizontal green lines)—chosen to maximize speedup—on A100 and L40 GPUs. The optimizer can select the optimal tree size and depth for each type of hardware; by limiting the depth of the tree it can make speculation faster and thus attain larger speedups than the trees with unconstrained depth (orange lines).

As mentioned in Section 4.1, one of the inputs to the hardware aware optimizer is 
𝑡
⁢
(
𝑛
)
, which is the hardware-dependent amount of time it takes the target model to verify 
𝑛
 tokens divided by the time to verify 1 token. In Figure 8 we show the forward pass times for different models on different hardware, for different number of tokens 
𝑛
. As you can see, the forward pass times are roughly constant for low values of 
𝑛
, but then eventually start growing roughly linearly in 
𝑛
—the value of 
𝑛
 at which 
𝑡
⁢
(
𝑛
)
 begins to grow is model and hardware dependent. In general, this value of 
𝑛
 is lower for hardware that has a higher ratio of bandwidth (between GPU HBM and SRAM) to FLOPS, because it is less memory bound).

Figure 8:Forward pass times for different model/hardware combinations as a function of the number of tokens 
𝑛
 being processed. We use these values to choose the optimal tree.
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

