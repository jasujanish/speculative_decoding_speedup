Title: OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure

URL Source: https://arxiv.org/html/2406.17276

Published Time: Fri, 25 Apr 2025 00:34:30 GMT

Markdown Content:
Jikai Wang 1 1 1 footnotemark: 1, Yi Su 1, Juntao Li 1, 

Qingrong Xia 2, Zi Ye 2, Xinyu Duan 2, Zhefeng Wang 2, Min Zhang 1, 

1 Institute of Computer Science and Technology, Soochow University, China 

2 Huawei Cloud 

[risus254@gmail.com](mailto:risus254@gmail.com), [yisunlp@outlook.com](mailto:yisunlp@outlook.com)

[ljt@suda.edu.cn](mailto:ljt@suda.edu.cn)

###### Abstract

Autoregressive language models demonstrate excellent performance in various scenarios. However, the inference efficiency is limited by its one-step-one-word generation mode, which has become a pressing problem recently as the models become increasingly larger. Speculative decoding employs a "draft and then verify" mechanism to allow multiple tokens to be generated in one step, realizing lossless acceleration. Existing methods mainly adopt fixed heuristic draft structures, which fail to adapt to different situations to maximize the acceptance length during verification. To alleviate this dilemma, we proposed OPT-Tree, an algorithm to construct adaptive and scalable draft trees. It searches the optimal tree structure that maximizes the mathematical expectation of the acceptance length in each decoding step. Experimental results reveal that OPT-Tree outperforms the existing draft structures and achieves a speed-up ratio of up to 3.2 compared with autoregressive decoding. If the draft model is powerful enough and the node budget is sufficient, it can generate more than ten tokens in a single step. Our code is available at [https://github.com/Jikai0Wang/OPT-Tree](https://github.com/Jikai0Wang/OPT-Tree).

OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure

Jikai Wang 1 1 1 footnotemark: 1, Yi Su 1††thanks: Equal contribution., Juntao Li 1††thanks: Corresponding author.,Qingrong Xia 2, Zi Ye 2, Xinyu Duan 2, Zhefeng Wang 2, Min Zhang 1,1 Institute of Computer Science and Technology, Soochow University, China 2 Huawei Cloud[risus254@gmail.com](mailto:risus254@gmail.com), [yisunlp@outlook.com](mailto:yisunlp@outlook.com)[ljt@suda.edu.cn](mailto:ljt@suda.edu.cn)

1 Introduction
--------------

Large language models (LLMs) (Black et al., [2022](https://arxiv.org/html/2406.17276v4#bib.bib2); Touvron et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib16); Achiam et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib1); Zheng et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib22)) have achieved remarkable performance in various NLP scenarios. As models grow in size and complexity, the computational demands for inference increase significantly. Therefore, it is becoming increasingly important to accelerate decoding to save computing overhead.

Autoregressive models (Black et al., [2022](https://arxiv.org/html/2406.17276v4#bib.bib2); Zhang et al., [2022](https://arxiv.org/html/2406.17276v4#bib.bib21); Touvron et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib16)) usually generate one token in one decoding step, leading to limited decoding efficiency. In recent work, speculative decoding (Leviathan et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib11); He et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib9); Fu et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib8); Cai et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib3); Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12)) has shown great potential for lossless accelerated decoding. It applies a "draft and then verify" mechanism to maintain the original output distribution of the target model to be accelerated. Drafting is performed by a less-overhead drafting model. The generated draft is verified in parallel by the target model to generate multiple tokens in one decoding step, bringing promising acceleration.

![Image 1: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/intro.png)

Figure 1: Draft structures used in speculative decoding. Nodes in the same layer share the same position index. OPT-Tree various in each decoding step to achieve a larger acceptance length.

Existing work like EAGLE Li et al. ([2024](https://arxiv.org/html/2406.17276v4#bib.bib12)) has proposed methods for training small but effective draft models. To the best of our knowledge, previous work mainly adopts drafts with structures of Sequences or fixed trees. However, we argue that neither of them is the optimal draft structure under a limited node budget. Sequence-structured drafts (Stern et al., [2018](https://arxiv.org/html/2406.17276v4#bib.bib15); Leviathan et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib11); Xia et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib17); Yang et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib19); Zhang et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib20); Fu et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib8)) contain redundant nodes. For example, "A-B-C-D-E" and "A-B-C-F-G" have the same prefix "A-B-C", which is calculated twice during verification. Therefore, there are only 7 valid tokens among the 10 nodes of these two sequences. Drafts with tree structure (He et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib9); Cai et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib3); Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12); Jeon et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib10); Chen et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib5)) solved this problem. The same token can only appear once in the same tree layer. A corresponding tree attention mask is designed for parallel verification. The specific structure of the tree is usually heuristic and remains constant. However, given a node budget, the best structure that maximizes the acceptance length during verification would change according to different inputs in each decoding step.

This paper proposes an adaptive and scalable tree structure called OPT-Tree. It can be applied to any autoregressive draft model. As is shown in Figure [1](https://arxiv.org/html/2406.17276v4#S1.F1 "Figure 1 ‣ 1 Introduction ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"), the tree structure adaptively changes in each decoding step to maximize the mathematical expectation of the acceptance length. We apply a greedy algorithm to construct an OPT-Tree in each step. Details are elaborated in Section [3](https://arxiv.org/html/2406.17276v4#S3 "3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"). We conduct comprehensive experiments in Section [4](https://arxiv.org/html/2406.17276v4#S4 "4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") to evaluate the effectiveness of OPT-Tree. Experimental results demonstrate that OPT-Tree outperforms the baselines and can be up to 3.2 times faster than vanilla autoregressive decoding. The mathematical expectation of the acceptance length is generally positively correlated with the actual acceptance length in practice. Moreover, OPT-Tree performs well when the tree size scales up. Using LLaMA-2-7B as the draft model, LLaMA-2-7B can generate 10 tokens in a single decoding step with OPT-Tree when the number of nodes is over 500, which indicates its great potential for adapting to more powerful computation resources and more effective draft models in the future.

2 Preliminaries
---------------

We provide the necessary definitions in this section.

Inference. After inputting x=(x 1,x 2,…,x l)x subscript 𝑥 1 subscript 𝑥 2…subscript 𝑥 𝑙\textbf{x}=(x_{1},x_{2},...,x_{l})x = ( italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ), where l 𝑙 l italic_l is the current sequence length, the target model M 𝑀 M italic_M and the drafting model M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT return the next word distribution p⁢(y l+1|x 1,x 2,…,x l)𝑝 conditional superscript 𝑦 𝑙 1 subscript 𝑥 1 subscript 𝑥 2…subscript 𝑥 𝑙 p(y^{l+1}|x_{1},x_{2},...,x_{l})italic_p ( italic_y start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT | italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) and p d⁢(y^l+1|x 1,x 2,…,x l)subscript 𝑝 𝑑 conditional superscript^𝑦 𝑙 1 subscript 𝑥 1 subscript 𝑥 2…subscript 𝑥 𝑙 p_{d}(\hat{y}^{l+1}|x_{1},x_{2},...,x_{l})italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( over^ start_ARG italic_y end_ARG start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT | italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) respectively, where y l+1 superscript 𝑦 𝑙 1 y^{l+1}italic_y start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT and y^l+1 superscript^𝑦 𝑙 1\hat{y}^{l+1}over^ start_ARG italic_y end_ARG start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT are the sampled next words.

Speculative Decoding. In speculative decoding with tree-structured draft, M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT first infers d 𝑑 d italic_d steps to generate a draft tree T 𝑇 T italic_T of depth d 𝑑 d italic_d and then M 𝑀 M italic_M verify the draft. The verification depends on the sampling method. For greedy sampling, the ground truth is the sequence of tokens with the highest probability for each position output by M 𝑀 M italic_M. For all branches in the tree that contain the root node, the longest branch with the same prefix as the ground truth is accepted. Therefore, multiple tokens can be generated in one decoding step while ensuring that the generated sequences are consistent with the original ones.

3 OPT-Tree
----------

This section introduces OPT-Tree, an algorithm for constructing our defined optimal draft tree structure for any input sequence in speculative decoding with autoregressive draft models.

Draft tree T 𝑇 T italic_T is defined as follows:

T=(𝕍,𝔼)𝑇 𝕍 𝔼\displaystyle T=(\mathbb{V},\mathbb{E})italic_T = ( blackboard_V , blackboard_E )(1)
𝕍=𝕍 absent\displaystyle\mathbb{V}=blackboard_V =⋃i=l+1 l+d⋃j=1 n i{(y^j i,p^j i)},superscript subscript 𝑖 𝑙 1 𝑙 𝑑 superscript subscript 𝑗 1 subscript 𝑛 𝑖 subscript superscript^𝑦 𝑖 𝑗 subscript superscript^𝑝 𝑖 𝑗\displaystyle\bigcup_{i=l+1}^{l+d}\bigcup_{j=1}^{n_{i}}\left\{(\hat{y}^{i}_{j}% ,\hat{p}^{i}_{j})\right\},⋃ start_POSTSUBSCRIPT italic_i = italic_l + 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l + italic_d end_POSTSUPERSCRIPT ⋃ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUPERSCRIPT { ( over^ start_ARG italic_y end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) } ,

where 𝕍 𝕍\mathbb{V}blackboard_V and 𝔼 𝔼\mathbb{E}blackboard_E is the set of all nodes and edges. n i subscript 𝑛 𝑖 n_{i}italic_n start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT represents the number of sampled tokens in the i t⁢h subscript 𝑖 𝑡 ℎ i_{th}italic_i start_POSTSUBSCRIPT italic_t italic_h end_POSTSUBSCRIPT layer of T 𝑇 T italic_T. p^j i subscript superscript^𝑝 𝑖 𝑗\hat{p}^{i}_{j}over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is calculated by:

p^j i=∏y^∈ℙ⁢(y^j i)p d⁢(y^),subscript superscript^𝑝 𝑖 𝑗 subscript product^𝑦 ℙ superscript subscript^𝑦 𝑗 𝑖 subscript 𝑝 𝑑^𝑦\hat{p}^{i}_{j}=\prod_{\hat{y}\in\mathbb{P}(\hat{y}_{j}^{i})}p_{d}(\hat{y}),over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = ∏ start_POSTSUBSCRIPT over^ start_ARG italic_y end_ARG ∈ blackboard_P ( over^ start_ARG italic_y end_ARG start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( over^ start_ARG italic_y end_ARG ) ,(2)

where ℙ⁢(y^j i)ℙ superscript subscript^𝑦 𝑗 𝑖\mathbb{P}(\hat{y}_{j}^{i})blackboard_P ( over^ start_ARG italic_y end_ARG start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT ) is the set of all parent nodes of y^j i superscript subscript^𝑦 𝑗 𝑖\hat{y}_{j}^{i}over^ start_ARG italic_y end_ARG start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT (including itself). p^j i subscript superscript^𝑝 𝑖 𝑗\hat{p}^{i}_{j}over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT of the root node is regarded as positive infinity. For each node in T 𝑇 T italic_T, if it has k 𝑘 k italic_k children, they are k 𝑘 k italic_k tokens greedily sampled according to p d subscript 𝑝 𝑑 p_{d}italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT from its subsequent token distribution. The purpose of calculating p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG is to simplify subsequent operations.

###### Theorem 3.1.

For any two nodes v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT in the tree, if v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is a node in the subtree of v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT, then p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG of v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is less than p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG of v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT.

![Image 2: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/tree.png)

Figure 2: An example of a draft tree containing p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG in each node. The value of E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) is 2.07.

Considering a certain step in speculative decoding whose input is x, the draft model M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT generates a draft tree based on x and the given tree structure T 𝑇 T italic_T. Then, the target model inputs the draft tree and the corresponding tree attention mask and returns the next tokens of each token in T 𝑇 T italic_T. We get the longest accepted candidate with length A 𝐴 A italic_A by comparing the next tokens and the draft tree. Given M 𝑀 M italic_M, M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT and n 𝑛 n italic_n, for input x, an optimal tree structure T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT should maximize the mathematical expectation of the acceptance length E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ). Note that T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT changes as the input changes. Since the optimization goal of the draft model is to make its output distribution close to the target model distribution, for each node, p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG will be positively related to its probability of being accepted during verification when using an effective draft model for speculative decoding. Therefore, E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) can be approximately calculated by p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG:

E⁢(A)𝐸 𝐴\displaystyle E(A)italic_E ( italic_A )=∑(y^j i,p^j i)∈T∏y^∈ℙ⁢(y^j i)p d⁢(y^)absent subscript subscript superscript^𝑦 𝑖 𝑗 subscript superscript^𝑝 𝑖 𝑗 𝑇 subscript product^𝑦 ℙ superscript subscript^𝑦 𝑗 𝑖 subscript 𝑝 𝑑^𝑦\displaystyle=\sum_{(\hat{y}^{i}_{j},\hat{p}^{i}_{j})\in T}\prod_{\hat{y}\in% \mathbb{P}(\hat{y}_{j}^{i})}p_{d}(\hat{y})= ∑ start_POSTSUBSCRIPT ( over^ start_ARG italic_y end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) ∈ italic_T end_POSTSUBSCRIPT ∏ start_POSTSUBSCRIPT over^ start_ARG italic_y end_ARG ∈ blackboard_P ( over^ start_ARG italic_y end_ARG start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( over^ start_ARG italic_y end_ARG )(3)
=∑(y^j i,p^j i)∈T p^j i.absent subscript subscript superscript^𝑦 𝑖 𝑗 subscript superscript^𝑝 𝑖 𝑗 𝑇 subscript superscript^𝑝 𝑖 𝑗\displaystyle=\sum_{(\hat{y}^{i}_{j},\hat{p}^{i}_{j})\in T}\hat{p}^{i}_{j}.= ∑ start_POSTSUBSCRIPT ( over^ start_ARG italic_y end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) ∈ italic_T end_POSTSUBSCRIPT over^ start_ARG italic_p end_ARG start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT .

Figure [2](https://arxiv.org/html/2406.17276v4#S3.F2 "Figure 2 ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") shows a simple example of calculating p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG and E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ). E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) should positively correlate with the acceptance length. We discuss their correlation in Section [4](https://arxiv.org/html/2406.17276v4#S4.F4 "Figure 4 ‣ 4.2 Correlation between 𝐸⁢(𝐴) and 𝐴 ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure").

We use E s⁢u⁢b⁢(T,n)subscript 𝐸 𝑠 𝑢 𝑏 𝑇 𝑛 E_{sub}(T,n)italic_E start_POSTSUBSCRIPT italic_s italic_u italic_b end_POSTSUBSCRIPT ( italic_T , italic_n ) to represent the maximum value of E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) for all subtrees of T 𝑇 T italic_T that contain the root node and have n nodes. Note that the root node is not considered when calculating node trees and mathematical expectations.

Then, we propose Algorithm [1](https://arxiv.org/html/2406.17276v4#alg1 "Algorithm 1 ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") to construct T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT during the drafting phase for each decoding step. We initialize T 𝑇 T italic_T with a root node. At each drafting step, we greedily sample n tokens with the largest p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG in the next token distributions of nodes in the last layer of T 𝑇 T italic_T to construct the next layer. T 𝑇 T italic_T has d∗n 𝑑 𝑛 d*n italic_d ∗ italic_n nodes at this time. Finally, we select the n 𝑛 n italic_n nodes in T 𝑇 T italic_T with the largest p 𝑝 p italic_p. It is easy to prove that these n 𝑛 n italic_n nodes are a subtree of T 𝑇 T italic_T, which contains the root node:

###### Proof.

(1) If these nodes can not form a tree with the root, there is at least one node v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT whose parent node v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is not among these nodes. (2) According to Theorem [3.1](https://arxiv.org/html/2406.17276v4#S3.Thmtheorem1 "Theorem 3.1. ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"), p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG of v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is larger than p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG of v i subscript 𝑣 𝑖 v_{i}italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. Therefore, v j subscript 𝑣 𝑗 v_{j}italic_v start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is also selected. (1) and (2) are contradictory, so these nodes must be able to form a subtree of T 𝑇 T italic_T containing the root node. ∎

Algorithm 1 Construct an OPT-Tree T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT

0: Input sequence

x=(x 1,x 2,…,x l)x subscript 𝑥 1 subscript 𝑥 2…subscript 𝑥 𝑙\textbf{x}=(x_{1},x_{2},...,x_{l})x = ( italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT )
, draft model

M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
, number of nodes

n 𝑛 n italic_n
, threshold

δ 𝛿\delta italic_δ
.

0:A draft tree

T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT
.

Initialize a tree

T 𝑇 T italic_T
with root node

x l subscript 𝑥 𝑙 x_{l}italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT

E←0←𝐸 0 E\leftarrow 0 italic_E ← 0

Output distribution

P d⁢(T)←M d⁢(T)←subscript 𝑃 𝑑 𝑇 subscript 𝑀 𝑑 𝑇 P_{d}(T)\leftarrow M_{d}(T)italic_P start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T ) ← italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T )

T←t⁢o⁢p⁢k⁢(P d⁢(T),n)←𝑇 𝑡 𝑜 𝑝 𝑘 subscript 𝑃 𝑑 𝑇 𝑛 T\leftarrow topk(P_{d}(T),n)italic_T ← italic_t italic_o italic_p italic_k ( italic_P start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T ) , italic_n )

while Depth of tree

D⁢(T)<n 𝐷 𝑇 𝑛 D(T)<n italic_D ( italic_T ) < italic_n
and

E s⁢u⁢b⁢(T,n)−E>δ subscript 𝐸 𝑠 𝑢 𝑏 𝑇 𝑛 𝐸 𝛿 E_{sub}(T,n)-E>\delta italic_E start_POSTSUBSCRIPT italic_s italic_u italic_b end_POSTSUBSCRIPT ( italic_T , italic_n ) - italic_E > italic_δ
do

/⁣//// /
Drafting step

E←E s⁢u⁢b⁢(T,n)←𝐸 subscript 𝐸 𝑠 𝑢 𝑏 𝑇 𝑛 E\leftarrow E_{sub}(T,n)italic_E ← italic_E start_POSTSUBSCRIPT italic_s italic_u italic_b end_POSTSUBSCRIPT ( italic_T , italic_n )

Output distribution

P d⁢(T)←M d⁢(T)←subscript 𝑃 𝑑 𝑇 subscript 𝑀 𝑑 𝑇 P_{d}(T)\leftarrow M_{d}(T)italic_P start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T ) ← italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T )

T←t⁢o⁢p⁢k⁢(P d⁢(T),n)←𝑇 𝑡 𝑜 𝑝 𝑘 subscript 𝑃 𝑑 𝑇 𝑛 T\leftarrow topk(P_{d}(T),n)italic_T ← italic_t italic_o italic_p italic_k ( italic_P start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_T ) , italic_n )

end while

T o⁢p⁢t←←subscript 𝑇 𝑜 𝑝 𝑡 absent T_{opt}\leftarrow italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT ←
Select the

n 𝑛 n italic_n
nodes with the largest

p^^𝑝\hat{p}over^ start_ARG italic_p end_ARG
from

T 𝑇 T italic_T

###### Theorem 3.2.

As the drafting step increases, E s⁢u⁢b⁢(T,n)subscript 𝐸 𝑠 𝑢 𝑏 𝑇 𝑛 E_{sub}(T,n)italic_E start_POSTSUBSCRIPT italic_s italic_u italic_b end_POSTSUBSCRIPT ( italic_T , italic_n ) is monotonic non-decreasing.

Algorithm 2 Speculative Decoding with Adaptive Draft Tree Structure

0: Input sequence

x=(x 1,x 2,…,x l)x subscript 𝑥 1 subscript 𝑥 2…subscript 𝑥 𝑙\textbf{x}=(x_{1},x_{2},...,x_{l})x = ( italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT )
, target model

M 𝑀 M italic_M
, draft model

M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT
, number of nodes

n 𝑛 n italic_n
, threshold

δ 𝛿\delta italic_δ
.

0:New input sequence

x′=(x 1,x 2,…,\textbf{x}^{\prime}=(x_{1},x_{2},...,x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = ( italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ,x l+A)x_{l+A})italic_x start_POSTSUBSCRIPT italic_l + italic_A end_POSTSUBSCRIPT )

T o⁢p⁢t←←subscript 𝑇 𝑜 𝑝 𝑡 absent T_{opt}\leftarrow italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT ←
Construct the draft tree with

n 𝑛 n italic_n
nodes

m⁢a⁢s⁢k←←𝑚 𝑎 𝑠 𝑘 absent mask\leftarrow italic_m italic_a italic_s italic_k ←
Compute the corresponding tree attention mask

P←M⁢(T o⁢p⁢t,m⁢a⁢s⁢k)←𝑃 𝑀 subscript 𝑇 𝑜 𝑝 𝑡 𝑚 𝑎 𝑠 𝑘 P\leftarrow M(T_{opt},mask)italic_P ← italic_M ( italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT , italic_m italic_a italic_s italic_k )

(y l+1,y l+2,…,y l+A)←V⁢e⁢r⁢i⁢f⁢y⁢(T o⁢p⁢t,P)←superscript 𝑦 𝑙 1 superscript 𝑦 𝑙 2…superscript 𝑦 𝑙 𝐴 𝑉 𝑒 𝑟 𝑖 𝑓 𝑦 subscript 𝑇 𝑜 𝑝 𝑡 𝑃(y^{l+1},y^{l+2},...,y^{l+A})\leftarrow Verify(T_{opt},P)( italic_y start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT , italic_y start_POSTSUPERSCRIPT italic_l + 2 end_POSTSUPERSCRIPT , … , italic_y start_POSTSUPERSCRIPT italic_l + italic_A end_POSTSUPERSCRIPT ) ← italic_V italic_e italic_r italic_i italic_f italic_y ( italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT , italic_P )

/⁣//// /
Find the longest accepted candidate. If a sequence of length

A−1 𝐴 1 A-1 italic_A - 1
successfully hits, its next word will also be accepted. So, the total acceptance length is

A 𝐴 A italic_A
.

x′←C⁢o⁢n⁢c⁢a⁢t⁢(x,(y l+1,y l+2,…,y l+A))←superscript x′𝐶 𝑜 𝑛 𝑐 𝑎 𝑡 x superscript 𝑦 𝑙 1 superscript 𝑦 𝑙 2…superscript 𝑦 𝑙 𝐴\textbf{x}^{\prime}\leftarrow Concat(\textbf{x},(y^{l+1},y^{l+2},...,y^{l+A}))x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ← italic_C italic_o italic_n italic_c italic_a italic_t ( x , ( italic_y start_POSTSUPERSCRIPT italic_l + 1 end_POSTSUPERSCRIPT , italic_y start_POSTSUPERSCRIPT italic_l + 2 end_POSTSUPERSCRIPT , … , italic_y start_POSTSUPERSCRIPT italic_l + italic_A end_POSTSUPERSCRIPT ) )

M 𝑀 M italic_M M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT Tree MAL Tokens/s Speedup M 𝑀 M italic_M M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT Tree MAL Tokens/s Speedup
LLaMA-2-7B None-1.00 51.89 1.00 LLaMA-2-13B None-1.00 26.79 1.00
L-68M Binary 2.12 68.58 1.32 L-68M Binary 2.05 40.24 1.50
EAGLE 2.47 77.06 1.49 EAGLE 2.42 46.82 1.75
OPT-Tree 2.58 87.57 1.69 OPT-Tree 2.58 48.10 1.80
L-1B Binary 3.95 46.10 0.89 L-1B Binary 3.95 37.37 1.39
EAGLE 4.23 47.74 0.92 EAGLE 4.25 40.12 1.50
OPT-Tree 4.88 52.48 1.01 OPT-Tree 5.20 43.40 1.62
EAGLE Binary 3.40 107.91 2.08 EAGLE Binary 3.54 66.24 2.47
EAGLE 3.73 130.50 2.51 EAGLE 3.80 73.97 2.76
OPT-Tree 4.36 132.75 2.56 OPT-Tree 4.35 76.61 2.86
LLaMA-2-70B None-1.00 6.29 1.00 Vicuna-33B None-1.00 11.25 1.00
L-7B Binary 4.84 11.05 1.76 V-7B Binary 4.41 12.49 1.11
EAGLE 4.97 11.35 1.80 EAGLE 4.64 12.99 1.15
OPT-Tree 7.74 11.65 1.85 OPT-Tree 6.51 13.74 1.22
EAGLE Binary 3.39 17.02 2.71 EAGLE Binary 2.35 21.13 1.88
EAGLE 3.67 18.81 2.99 EAGLE 2.69 24.92 2.21
OPT-Tree 4.06 19.21 3.05 OPT-Tree 3.06 25.17 2.24

Table 1:  Experimental results on MT-Bench. M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT being None represents vanilla autoregressive decoding. "L" and "V" in M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT column represent "LLaMA-2" and "Vicuna". "MAL" indicates "Mean Acceptance Length".

According to Theorem [3.2](https://arxiv.org/html/2406.17276v4#S3.Thmtheorem2 "Theorem 3.2. ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"), we can get the desired T o⁢p⁢t subscript 𝑇 𝑜 𝑝 𝑡 T_{opt}italic_T start_POSTSUBSCRIPT italic_o italic_p italic_t end_POSTSUBSCRIPT in theory by stopping drafting when E⁢(T)𝐸 𝑇 E(T)italic_E ( italic_T ) no longer increases. However, the draft model brings additional overhead to the practice. For autoregressive draft models, the drafting overhead is proportional to the depth of the draft tree.

Taking this into consideration, we introduce a threshold δ 𝛿\delta italic_δ when setting the conditions for terminating drafting. The value of δ 𝛿\delta italic_δ should be controlled between μ 𝜇\mu italic_μ and 1, where μ 𝜇\mu italic_μ is the time of one drafting step divided by the time of one decoding step.

A complete decoding step of M 𝑀 M italic_M is shown in Algorithm [2](https://arxiv.org/html/2406.17276v4#alg2 "Algorithm 2 ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"). In practice, both M 𝑀 M italic_M and M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT use key and value cache to calculate attention. Thus, the actual input length of each drafting step is n 𝑛 n italic_n, which avoids computational bottlenecks in the inference of draft model under larger budgets of tree size.

4 Experiments
-------------

### 4.1 Main Results

Setup. We adopt LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B (Touvron et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib16)) and Vicuna-33B (Zheng et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib22)) as target models to verify the effectiveness of OPT-Tree. We use a single GeForce RTX 4090 GPU for LLaMA-2-7B, a single L20 GPU for LLaMA-2-13B and 4 A100-PCIE-40GB GPUs for LLaMA-2-70B and Vicuna-33B. We choose one or two smaller models in the same version as the draft model for each target model. Moreover, we adopt a corresponding EAGLE draft model for each target model. The temperature is set to zero. EAGLE (Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12)) is an effective speculation decoding method that trains additional autoregressive heads as draft models. It uses a well-designed heuristic draft tree structure with 25 nodes. In our experiments, we regard it as the EAGLE draft tree. EAGLE is certified by Xia et al. ([2024](https://arxiv.org/html/2406.17276v4#bib.bib18)) as the fastest speculative method in their experiments. For each target and draft model group, we perform speculative decoding with greedy sampling and compare OPT-Tree with the Binary tree and EAGLE tree.

![Image 3: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/forward.png)

Figure 3: The relationship between input length and the wall time for inference for models of different sizes on various GPUs.

M 𝑀 M italic_M M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT Tree MAL Tokens/s Speedup M 𝑀 M italic_M M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT Tree MAL Tokens/s Speedup
LLaMA-2-7B None-1.00 52.76 1.00 LLaMA-2-13B None-1.00 27.10 1.00
L-68M Binary 2.20 73.49 1.39 L-68M Binary 2.21 45.18 1.67
EAGLE 2.63 85.62 1.62 EAGLE 2.60 52.83 1.95
OPT-Tree 2.78 96.43 1.83 OPT-Tree 2.81 53.54 1.98
L-1B Binary 3.55 40.69 0.77 L-1B Binary 3.76 36.54 1.35
EAGLE 3.87 44.42 0.84 EAGLE 4.10 37.29 1.38
OPT-Tree 4.46 50.83 0.96 OPT-Tree 5.10 42.97 1.59
EAGLE Binary 3.52 118.15 2.24 EAGLE Binary 3.80 73.30 2.70
EAGLE 3.83 137.41 2.60 EAGLE 4.06 80.47 2.97
OPT-Tree 4.68 140.55 2.66 OPT-Tree 5.03 80.94 2.99
LLaMA-2-70B None-1.00 6.38 1.00 Vicuna-33B None-1.00 10.74 1.00
L-7B Binary 4.85 11.20 1.76 V-7B Binary 4.95 13.15 1.22
EAGLE 4.98 11.51 1.80 EAGLE 4.81 13.38 1.25
OPT-Tree 7.62 12.10 1.90 OPT-Tree 6.35 13.98 1.30
EAGLE Binary 3.62 18.63 2.92 EAGLE Binary 2.82 25.20 2.35
EAGLE 3.91 20.42 3.20 EAGLE 3.15 28.37 2.64
OPT-Tree 4.55 20.50 3.21 OPT-Tree 3.47 28.76 2.68

Table 2:  Experimental results on GSM8K. M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT being None represents vanilla autoregressive decoding. "L" and "V" in M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT column represent "LLaMA-2" and "Vicuna". "MAL" indicates "Mean Acceptance Length".

We compare the average acceptance length and number of tokens generated per second decoding with different tree structures. The speedup ratio is calculated according to generation speed. The node budget is determined by the target model and computational resource since the inference time generally remains the same within a certain input length. Figure [3](https://arxiv.org/html/2406.17276v4#S4.F3 "Figure 3 ‣ 4.1 Main Results ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") displays the inference time for input with various lengths for the 4 target models used in the experiments. The number of nodes needs to be controlled within a certain range to avoid excessive time consumption in the verification phase. It is treated as a hyperparameter chosen in [25,50,60]25 50 60[25,50,60][ 25 , 50 , 60 ] to maximize the speedup ratio according to different target models and GPU resources except for the EAGLE tree. We conduct evaluation on MT-Bench (Zheng et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib22)) and GSM8K (Cobbe et al., [2021](https://arxiv.org/html/2406.17276v4#bib.bib7)).

Results. Experimental results are shown in Table [1](https://arxiv.org/html/2406.17276v4#S3.T1 "Table 1 ‣ 3 OPT-Tree ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") and Table [2](https://arxiv.org/html/2406.17276v4#S4.T2 "Table 2 ‣ 4.1 Main Results ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"). Note that using LLaMA-2-1B as the draft model can hardly speed up decoding when the target model is LLaMA-2-7B because the difference in inference time between the two models is too small. EAGLE draft models achieve strong performance with fewer parameters, thus providing better acceleration than the small models in the same series with the target models. OPT-Tree outperforms other tree structures in terms of mean acceptance length in each group of experiments, especially when the performance of the draft model is close to the target model (e.g., LLaMA-2-70B combined with L-7B and Vicuna-33B combined with Vicuna-7B), indicating its high upper limit. Since OPT-Trees are usually deeper than binary trees and EAGLE trees, they incur more overhead when drafting. Therefore, from the perspective of tokens per second, the improvement is not as significant as that from the mean acceptance length. Tokens per second are affected by different hardware resources and random errors. In addition, some method-independent techniques can also be used to reduce computation time. For example, the unchanged part of the attention mask in the drafting phase can be initialized only once and called multiple times, thus saving the time of multiple initializations. In order to make a fairer comparison in our experiments, we avoid these tricks to be consistent with EAGLE’s practice. Overall, OPT-Tree outperforms the baselines. It can be up to about 3.2 times faster than vanilla autoregressive decoding. The similar performance on both datasets verifies the robustness of the proposed method.

### 4.2 Correlation between E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) and A 𝐴 A italic_A

![Image 4: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/e_similarity.png)

Figure 4: Correlation between E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) and A 𝐴 A italic_A. The horizontal axis represents E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ), and the vertical axis represents A 𝐴 A italic_A. Each square shows the number of times the corresponding situation occurs. The darker the color, the more times it indicates.

The theory of OPT-Tree is based on the premise that E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) is positively correlated with actual A 𝐴 A italic_A. We record the values of E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) and A 𝐴 A italic_A of OPT-Tree in about 8000 decoding steps for 4 groups of M 𝑀 M italic_M and M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT. Figure [4](https://arxiv.org/html/2406.17276v4#S4.F4 "Figure 4 ‣ 4.2 Correlation between 𝐸⁢(𝐴) and 𝐴 ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") shows the results. The value of E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) is rounded. The darker areas in the four images are basically distributed along the main diagonal line. When E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) of the tree is larger, it also tends to get a more considerable acceptance length after verification. A stronger draft model shifts the distribution to the lower right corner. These phenomena corroborate our theoretical analysis. In addition, in the LLaMA-2-70B+LLaMA-2-7B group, high values of E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) and A 𝐴 A italic_A (e.g., E⁢(A)=14,A=15 formulae-sequence 𝐸 𝐴 14 𝐴 15 E(A)=14,A=15 italic_E ( italic_A ) = 14 , italic_A = 15) are generally found, which demonstrates the potential of OPT-Tree to adapt to stronger draft models and larger draft tree sizes.

### 4.3 Scaling the Draft Tree Size

![Image 5: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/scale.png)

Figure 5: Mean acceptance length under different tree sizes under two sets of experiments. 

We conduct experiments to explore the changes in mean acceptance length with larger tree sizes. We compare OPT-Tree with Sequoia (Chen et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib5)) using LLaMA-2-7B and LLaMA-2-70B as target models. Sequoia is a scalable draft tree that uses dynamic programming to solve for the tree structure. It requires the target and draft models to be used in advance to infer some samples to determine the best structure. The tree structure is fixed when doing speculative decoding. We use 200 samples in C4 (Raffel et al., [2020](https://arxiv.org/html/2406.17276v4#bib.bib13)) to construct the Sequoia trees. Temperature is set to 0 in the experiments.

The experimental results are shown in Figure [5](https://arxiv.org/html/2406.17276v4#S4.F5 "Figure 5 ‣ 4.3 Scaling the Draft Tree Size ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"). OPT-Tree outperforms Sequoia under various tree sizes. For LLaMA-2-7B+LLaMA-2-68M, the mean acceptance length with both OPT-Tree and Sequoia proliferates when the number of nodes is smaller than 130. When the number of nodes exceeds 140, the mean acceptance length increases slowly. For LLaMA-2-70B+LLaMA-2-7B, the growth of mean acceptance length with Sequoia tends to be flat when the number of nodes exceeds 150. However, OPT-Tree can continue to improve the mean acceptance length even if the number of nodes exceeds 500. Since LLaMA-2-7B is a strong draft model for LLaMA-2-70B, the mean acceptance length can achieve 10 with an OPT-Tree of 500 nodes. A tree with 500 nodes costs a large amount of computation time for LLaMA-2-70B with A100-PCIE-40GB GPUs, thus being unable to speed up decoding in our practice. However, this cost may be acceptable if more powerful computational resources are equipped in the future.

### 4.4 Impact of the Threshold

![Image 6: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/threshold.png)

Figure 6: The two figures on the left and right are the mean acceptance length and tokens/s under different thresholds on MT-Bench. The target model is LLaMA-2-7B. The blue and orange dashed lines in the right figure represent the values of μ 𝜇\mu italic_μ with LLaMA-2-68M and EAGLE as the draft model, respectively.

Considering the overhead of the draft model is proportional to the depth of the tree, the tree that maximizes the acceptance length does not necessarily have the highest speed-up ratio. Therefore, we experiment to study the mean acceptance length and tokens/s under different thresholds.

Figure [6](https://arxiv.org/html/2406.17276v4#S4.F6 "Figure 6 ‣ 4.4 Impact of the Threshold ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") shows the experimental results on LLaMA-2-7B. The mean acceptance length drops as the threshold grows when using LLaMA-2-68M as the draft model. However, there is a slight fluctuation for the EAGLE draft model. This is because E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) and A 𝐴 A italic_A are not completely equivalent. We calculate μ 𝜇\mu italic_μ for each group of models, which is the time of one drafting step divided by the time of one decoding step. A threshold that is too large will reduce the tree’s depth, thus reducing the value of A 𝐴 A italic_A. On the other hand, a threshold that is too small may make the tree too deep and increase the cost of drafting. When the depth of the tree increases by one but the increment of the E⁢(A)𝐸 𝐴 E(A)italic_E ( italic_A ) does not exceed μ 𝜇\mu italic_μ, it is not worth increasing the depth. So, we set a threshold between μ 𝜇\mu italic_μ and 1 in practice. LLaMA-2-68M and EAGLE achieve the highest acceleration when δ=0.2 𝛿 0.2\delta=0.2 italic_δ = 0.2 and δ=0.8 𝛿 0.8\delta=0.8 italic_δ = 0.8, respectively.

### 4.5 Performance on Non-greedy Settings

M 𝑀 M italic_M M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT MAL Tokens/s Speedup
LLaMA-2-7B L-68M 2.72 88.90 1.71
L-1B 5.25 49.76 0.96
EAGLE 4.07 125.79 2.42
LLaMA-2-13B L-68M 2.26 43.45 1.62
L-1B 4.23 37.84 1.41
EAGLE 4.13 69.27 2.21
LLaMA-2-70B L-7B 7.17 11.87 1.89
EAGLE 4.09 18.92 3.01
Vicuna-33B V-7B 4.91 13.48 1.20
EAGLE 2.89 25.31 2.25

Table 3: Performance of OPT-Tree on MT-Bench with the temperature set to 1. "L" and "V" in M d subscript 𝑀 𝑑 M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT column represents "LLaMA-2" and "Vicuna". "MAL" indicates "Mean Acceptance Length".

![Image 7: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/temperature.png)

Figure 7: The two figures on the left and right are the mean acceptance length and tokens/s with OPT-Tree with different temperatures on MT-Bench. The target model is LLaMA-2-7B.

In the decoding setting of non-greedy sampling (random sampling), we only modify the acceptable tokens during the verification phase. We conduct experiments to evaluate OPT-Tree on these non-greedy settings, where the temperature exceeds 0.

We perform speculative decoding with OPT-Tree on the MT-Bench dataset for all groups of models in [4.1](https://arxiv.org/html/2406.17276v4#S4.SS1 "4.1 Main Results ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") with the temperature set to 1. Table [3](https://arxiv.org/html/2406.17276v4#S4.T3 "Table 3 ‣ 4.5 Performance on Non-greedy Settings ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") displays the experimental results. The mean acceptance length and the speedup ratio of speculative decoding with OPT-Tree are slightly lower when the temperature is set to 1 than when the temperature is set to 0. Since the draft tree greedily samples tokens with higher probability, the positive correlation between E(A) and A will be weakened in the decoding of random sampling. Therefore, it is typical for the acceleration of speculative decoding to drop when the temperature is greater than 0. Figure [7](https://arxiv.org/html/2406.17276v4#S4.F7 "Figure 7 ‣ 4.5 Performance on Non-greedy Settings ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure") shows specific changes in mean acceptance length and tokens/s with different temperature values. Both metrics drop as the temperature rises in general. But even when the temperature is set to 1, opt-tree can still provide high speedup compared to vanilla autoregressive decoding.

### 4.6 Case Study

![Image 8: Refer to caption](https://arxiv.org/html/2406.17276v4/extracted/6385995/images/case.png)

Figure 8: An example of speculative decoding with OPT-Tree on LLaMA-2-70B. Text on a blue background is the input prompt. Blue text represents drafts generated by LLaMA-2-7B and accepted by LLaMA-2-70B. Red text represents the next token for each accepted draft, which is generated by LLaMA-2-70B during the verification.

We show an example of speculative decoding with an OPT-Tree of 50 nodes on LLaMA-2-70B with LLaMA-2-7B as the draft model in Figure [8](https://arxiv.org/html/2406.17276v4#S4.F8 "Figure 8 ‣ 4.6 Case Study ‣ 4 Experiments ‣ OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure"). The threshold is 0.7, and the temperature is 0. The mean acceptance length is 9.34, and the generation speed is 12.07 tokens per second. Most words (blue text) are generated by the draft model and then verified by the target model. Each couple of red words and the continuous blue text in front of it is generated in a single decoding step of the target model. The appearance of red words is either because the depth of the draft tree is limited or because none of the candidates for this position hits the target. Prepositions (e.g., in, for and with), conjunctions (e.g., and and or), articles (e.g., a and the), punctuation and other words which have no apparent practical meanings are prone to miss in the draft. In addition, the beginning of new sentences in drafts tends to be rejected because it has no solid sequential association with the previous word.

5 Related Work
--------------

Speculative decoding (Stern et al., [2018](https://arxiv.org/html/2406.17276v4#bib.bib15); Xia et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib17); Leviathan et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib11); Chen et al., [2023a](https://arxiv.org/html/2406.17276v4#bib.bib4)) accelerates autoregressive decoding by drafting and then verifying while ensuring consistent output. Drafting methods are mainly divided into independent drafting and self-drafting. Independent drafting leverages an external low-cost model. SpecDec (Xia et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib17)) trains a non-autoregressive model for drafting while others (Leviathan et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib11); Chen et al., [2023a](https://arxiv.org/html/2406.17276v4#bib.bib4); Spector and Re, [2023](https://arxiv.org/html/2406.17276v4#bib.bib14); Chen et al., [2023b](https://arxiv.org/html/2406.17276v4#bib.bib6), [2024](https://arxiv.org/html/2406.17276v4#bib.bib5)) directly utilize a smaller version of the target model. In addition, REST (He et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib9)) proposed a retrieval-based drafting method. Self-drafting uses the original information of the target model to draft. Yang et al. ([2023](https://arxiv.org/html/2406.17276v4#bib.bib19)) adopt an early-exiting mechanism for drafting. Similarly, Zhang et al. ([2023](https://arxiv.org/html/2406.17276v4#bib.bib20)) performs adaptive layer skipping in the drafting phase. Lookahead Decoding (Fu et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib8)) designed an algorithm for parallel drafting and verification. MEDUSA (Cai et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib3)) trains multiple decoding heads to obtain candidates for multiple steps from original features in parallel. Considering that different sampling results at each step in drafting will affect the distribution of subsequent outputs, EAGLE (Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12)) designed an autoregressive head, which introduced the embedding of each word in the drafting stage.

The verification method has evolved from sequence-structured verification to tree-structured verification. Early work (Stern et al., [2018](https://arxiv.org/html/2406.17276v4#bib.bib15); Leviathan et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib11); Xia et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib17); Yang et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib19); Zhang et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib20); Fu et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib8)) verifies drafts in the form of one or several sequences. However, as the number of verification tokens increases, there are a large number of prefix duplications between sequences, resulting in redundant calculations. To alleviate this problem, recent work (He et al., [2023](https://arxiv.org/html/2406.17276v4#bib.bib9); Cai et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib3); Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12); Jeon et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib10)) uses heuristic tree-structured drafts and designs corresponding attention masks for parallel verification. Chen et al. ([2024](https://arxiv.org/html/2406.17276v4#bib.bib5)) proposed Sequoia, an algorithm for constructing draft trees, which performs well as the tree size scales up.

6 Conclusion
------------

In this paper, we propose a novel and effective method called OPT-Tree to construct adaptive draft tree structures for speculative decoding. OPT-Tree maximizes the mathematical expectation of the acceptance length under any limited draft tree size. Experimental results with ten groups of target models and draft models on two datasets show that opt-tree outperforms existing draft structures. It achieves a lossless acceleration of up to 3.2 times compared to vanilla autoregressive decoding and shows robustness on different datasets and with different temperatures. Additionally, if equipped with a strong draft model, the mean acceptance length with OPT-Tree continues to grow even if the number of nodes is over 500, demonstrating its great potential for adapting to scenarios with more powerful computational resources.

Limitations
-----------

Different hardware resources and environments will affect the throughput speed reported in the experiments in this article. The experiments in this paper adopt the same decoding framework as EAGLE (Li et al., [2024](https://arxiv.org/html/2406.17276v4#bib.bib12)) for fair comparison. In practice, the decoding algorithm can be optimized from other perspectives to further improve the decoding speed, which is not explored in this paper.

References
----------

*   Achiam et al. (2023) Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_. 
*   Black et al. (2022) Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. 2022. [GPT-NeoX-20B: An open-source autoregressive language model](https://doi.org/10.18653/v1/2022.bigscience-1.9). In _Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models_, pages 95–136, virtual+Dublin. Association for Computational Linguistics. 
*   Cai et al. (2024) Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple llm inference acceleration framework with multiple decoding heads. _arXiv preprint arXiv:2401.10774_. 
*   Chen et al. (2023a) Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023a. Accelerating large language model decoding with speculative sampling. _arXiv preprint arXiv:2302.01318_. 
*   Chen et al. (2024) Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. 2024. Sequoia: Scalable, robust, and hardware-aware speculative decoding. _arXiv preprint arXiv:2402.12374_. 
*   Chen et al. (2023b) Ziyi Chen, Xiaocong Yang, Jiacheng Lin, Chenkai Sun, Jie Huang, and Kevin Chen-Chuan Chang. 2023b. Cascade speculative drafting for even faster llm inference. _arXiv preprint arXiv:2312.11462_. 
*   Cobbe et al. (2021) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_. 
*   Fu et al. (2024) Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of llm inference using lookahead decoding. _arXiv preprint arXiv:2402.02057_. 
*   He et al. (2023) Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He. 2023. Rest: Retrieval-based speculative decoding. _arXiv preprint arXiv:2311.08252_. 
*   Jeon et al. (2024) Wonseok Jeon, Mukul Gagrani, Raghavv Goel, Junyoung Park, Mingu Lee, and Christopher Lott. 2024. [Recursive speculative decoding: Accelerating LLM inference via sampling without replacement](https://openreview.net/forum?id=RdKYAHZPxg). In _ICLR 2024 Workshop on Large Language Model (LLM) Agents_. 
*   Leviathan et al. (2023) Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning_, pages 19274–19286. PMLR. 
*   Li et al. (2024) Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024. Eagle: Speculative sampling requires rethinking feature uncertainty. _arXiv preprint arXiv:2401.15077_. 
*   Raffel et al. (2020) Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_, 21(140):1–67. 
*   Spector and Re (2023) Benjamin Spector and Chris Re. 2023. Accelerating llm inference with staged speculative decoding. _arXiv preprint arXiv:2308.04623_. 
*   Stern et al. (2018) Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. _Advances in Neural Information Processing Systems_, 31. 
*   Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_. 
*   Xia et al. (2023) Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. 2023. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In _Findings of the Association for Computational Linguistics: EMNLP 2023_, pages 3909–3925. 
*   Xia et al. (2024) Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. 2024. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. _arXiv preprint arXiv:2401.07851_. 
*   Yang et al. (2023) Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, and Kangwook Lee. 2023. Predictive pipelined decoding: A compute-latency trade-off for exact llm decoding. _arXiv preprint arXiv:2307.05908_. 
*   Zhang et al. (2023) Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2023. Draft & verify: Lossless large language model acceleration via self-speculative decoding. _arXiv preprint arXiv:2309.08168_. 
*   Zhang et al. (2022) Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_. 
*   Zheng et al. (2024) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in Neural Information Processing Systems_, 36.

