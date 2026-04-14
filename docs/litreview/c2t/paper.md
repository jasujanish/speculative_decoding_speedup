C2T: A Classifier-Based Tree Construction Method in Speculative Decoding

1 Introduction

2 Background

2.1 Speculative Decoding

2.2 Tree Attention and EAGLE-2

3 Motivation

4 Methodology

4.1 Classifier

4.2 Tree Construction

4.2.1 First Pruning based on Confidence

4.2.2 Second Pruning based on Topk

5 Experiments

5.1 Feature Ablation

5.2 Dataset Transferability

5.3 Model Transferability

5.4 Speed up in Larger LLMs

5.5 Benefits in Chain Mode

6 Conclusion

A EAGLE-1 and EAGLE-2

B Classifier

B.1 Dataset

B.2 Training and Evaluation

B.3 Structure

B.4 Fine-tuning

B.5 Cross Validation

B.6 Use TopK for Second Pruning

C Proof of Simplified Calculation

D Confidence

E Additional Overhead

E.1 Experimental Analysis

E.2 Quantitative Analysis

F Combo with MTP-style Layer

C2T: A Classifier-Based Tree Construction Method in Speculative Decoding

Feiye Huo1,2,*,
Jianchao Tan2,*,
Kefeng Zhang2,
Xunliang Cai2,
Shengli Sun1,+

1Peking University,
2Meituan

Correspondence: phiyeh@stu.pku.edu.cn,
{tanjianchao02, zhangkefeng, caixunliang}@meituan.com, sunshengli@pku.edu.cn

Abstract

The growing scale of Large Language Models (LLMs) has exacerbated inference latency and computational costs. Speculative decoding methods, which aim to mitigate these issues, often face inefficiencies in the construction of token trees and the verification of candidate tokens. Existing strategies, including chain mode, static tree, and dynamic tree approaches, have limitations in accurately preparing candidate token trees for verification. We propose a novel method named C2T that adopts a lightweight classifier to generate and prune token trees dynamically. Our classifier considers additional feature variables beyond the commonly used joint probability to predict the confidence score for each draft token to determine whether it is the candidate token for verification. This method outperforms state-of-the-art (SOTA) methods such as EAGLE-2 on multiple benchmarks, by reducing the total number of candidate tokens by 25%, while maintaining or even improving the acceptance length.

C2T: A Classifier-Based Tree Construction Method in Speculative Decoding

Feiye Huo1,2,*,
Jianchao Tan2,*,
Kefeng Zhang2,
Xunliang Cai2,
Shengli Sun1,+

1Peking University,
2Meituan

Correspondence: phiyeh@stu.pku.edu.cn,
{tanjianchao02, zhangkefeng, caixunliang}@meituan.com, sunshengli@pku.edu.cn

Figure 1: Two illustrations of EAGLE-2 for verifying with 4 candidate tokens. Blue represents the chosen candidate tokens, red represents the tokens that were not chosen, bold text represents the correct answers, numbers on the arrows represent the generation probabilities, and C represents confidence, which in EAGLE-2 refers to joint probability.

1 Introduction

Large Language Models (LLMs) Achiam et al. (2023); Touvron et al. (2023); Chiang et al. (2023) have shown remarkable abilities in various fields, but face significant bottlenecks in autoregressive token generation due to high memory bandwidth demands and underutilized GPU resources Patterson (2004); Shazeer (2019), as each token requires access to all model parameters Radford et al. (2019); Brown et al. (2020). To address this issue, Speculative Decoding (SD) Chen et al. (2023); Leviathan et al. (2023) has been developed, which quickly generates multiple draft tokens and verifies them all at once using the target model to maximize GPU computational capacity, and it has been applied in the latest influential LLMs Liu et al. (2024).

Vanilla SD employs a chain structure for the draft tokens, and the verification process follows a topological order Chen et al. (2023); Leviathan et al. (2023). If a token is rejected, all subsequent tokens are also discarded. To overcome inefficiency, tree-structured draft tokens have been proposed Miao et al. (2024); Sun et al. (2024), which integrate multiple chains. Static tree methods, such as EAGLE-1 Li et al. (2024b) and Medusa Cai et al. (2024), use preset tree structures that bias the sampling rate of specific positions Chen et al. (2024), while dynamic methods, such as EAGLE-2 Li et al. (2024a), rely on contextual information.

For dynamic tree methods, most are designed to build and prune trees on the basis of confidence scores. The most straightforward approach is to use the joint probability as confidence Li et al. (2024a); Wang et al. (2024); Brown et al. (2024); Qin et al. (2024). However, directly using joint probability as confidence is not enough for complex situations, leading to misjudgments. As shown in Figure 1, which will be explained in detail in 3. From the perspective of tree data structure, whether a token is accepted or not, depends not only on its own property but also other nodes’ properties. This means that incorporating more variables together is necessary in the confidence score calculation. Therefore, we propose a tree construction method based on a designed tiny classifier to perform this calculation. Our contributions are summarized below:

•

We propose a classifier-based method named C2T for the dynamic construction of the token tree. And it can be easily integrated into any confidence-based SD system.

•

Our classifier demonstrates strong transferability across different datasets and within the same model family, thus being plug-and-play.

•

Compared to the SOTA method EAGLE-2, C2T reduces the number of candidate tokens by 25%percent2525\%25 % while maintaining or improving the acceptance length on multiple benchmarks.

2 Background

2.1 Speculative Decoding

Speculative decoding (SD) Chen et al. (2023); Leviathan et al. (2023) is an algorithm designed to speed up model reasoning by leveraging the parallel computing capabilities of attention mechanisms. It consists of two main stages. The first stage is the draft phase, where a smaller model, known as the draft model Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT, generates draft tokens. The second stage is verification, where the draft tokens are verified all at once by the larger model, known as the target model Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT. At the same time, we call the draft tokens chosen to be verified as candidate tokens.

2.2 Tree Attention and EAGLE-2

Vanilla SD uses a chain-like structure, where if a token is rejected, all subsequent tokens are also discarded. SpecInfer’s Miao et al. (2024) Tree Attention achieves the integration of multiple speculations at a minimal cost and has been widely adopted in other SD methods He et al. (2023); Sun et al. (2024); Cai et al. (2024); Li et al. (2024b); Chen et al. (2024); Svirschevski et al. (2024). In the pioneer SD works, the token tree was static, with a preset tree shape, and the draft model generated tokens layer by layer and filled in the corresponding positions. This static method is undoubtedly rough, so various heuristic dynamic generation methods appeared later Li et al. (2024a); Wang et al. (2024); Brown et al. (2024); Qin et al. (2024); Huang et al. (2024). Several trainable dynamic methods have been also proposed Mamou et al. (2024); Huang et al. (2024); Zhang et al. (2024), but due to design redundancies, they have only been applied to early stopping in chain-like drafting, which significantly limits their applicability.

The current SOTA dynamic tree construction method EAGLE-2 Li et al. (2024a) introduces the joint probability of each node as contextual information, and divides the sampling process into two stages: expand and rerank. The former is to build the tree and the latter is for post-pruning. We can denote the token tree after expansion as T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, and the token tree reranked as T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. There are three important parameters in EAGLE-2:

1.

K𝐾Kitalic_K: During expanding, each token in current layer generates TopK𝐾Kitalic_K tokens. Then, among all the generated tokens, the TopK𝐾Kitalic_K tokens are selected to generate the next tree layer.

2.

dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT: The rounds of the drafting in the expansion phase.

3.

N𝑁Nitalic_N: Use joint probability to rerank all nodes in T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, and then take the TopN𝑁Nitalic_N nodes as T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. Since the joint probability of the parent node is always greater than that of the child nodes, T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT must be a valid subtree of T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT. Therefore, TopN𝑁Nitalic_N determines the size of T2subscript𝑇2T_{2}italic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, which directly determines the number of candidate tokens.

However, to achieve the benefits claimed by EAGLE-2, the target model must verify nearly twice as many tokens, which is not a fair comparison. When aligning for the size of the token tree, the benefit of EAGLE-2 in terms of accept length is only 11%. Moreover, due to the additional latency introduced by the dynamic strategy, the wall-clock time of EAGLE-2 is more. Details in Appendix A.

(a) Average Probability Heatmap

(b) Accept Rate Heatmap

(c) Bias Heatmap

Figure 2: The coordinates in the three heatmaps have the same meanings: The y-axis represents the entropy value interval in which the current probability distribution lies. From top to bottom, these intervals are 0∼similar-to\sim∼1, 1∼similar-to\sim∼2, 2∼similar-to\sim∼3, 3∼similar-to\sim∼4, 4∼similar-to\sim∼5, 5∼similar-to\sim∼6, and >>>6. The x-axis shows the top 20 probabilities within the distribution, decreasing from left to right. Each square in the heatmap indicates the value corresponding to the probability rank within the respective entropy interval. In Figure 2(a), the value represents the average probability at each position, smoothed using a logarithm. In Figure 2(b), the value represents the accept rate at each position, also smoothed with a logarithm. In Figure 2(c), the value shows the bias between probability and accept rate, with red indicating probabilities higher than accept rates and blue indicating the opposite.

3 Motivation

As shown in Figure 2, we use the EAGLE-2’s LLaMA-2 7B model pair to reason on the MT-bench. We performed a single forward to calculate the entropy and divided it into seven intervals. We then computed the average probability of different ranks within these intervals to create the heatmap in Figure 2(a) which shows that lower entropy corresponds to more concentrated distributions.

We also tested the acceptance rate of tokens with different ranks in these intervals and generated the heatmap in Figure 2(b). The similarity between 2(a) and 2(b) confirms the correlation between probability and acceptance rate.

However, Figure 2(b) is less stable than Figure 2(a), indicating that there is a discrepancy. To quantify this, we subtracted the data matrix of Figure 2(a) from that of Figure 2(b) to create Figure 2(c), which shows the bias between probability and acceptance rate.

As shown in Figure 2(c), there are two main deviations between probability and acceptance rate:

•

When the entropy is high, the node with the highest probability in the distribution is underestimated in terms of probability;

•

When the entropy is low, the node with the highest probability in the distribution is overestimated in terms of probability.

These two deviations may lead to misjudgments as shown in Figure 1. For the "all" and "great", although the probabilities of their parent are greater than those of the parent of "a" and "to", due to their higher entropy in the probability distribution, the normalized generation probabilities are not high. As a result, they are outperformed in joint probability by the "a" from a distribution with lower entropy.

In addition, due to the nature of the token tree, if a token is rejected, all its subsequent child nodes will also be rejected. Therefore, the acceptance rate is a value that decreases with depth. Nodes at shallower levels should be given greater confidence. If only joint probability is considered, the depth factor will become inactive in some cases. As shown in Figure 1, although the "the" and the "has" are both the smaller ones in their respective distributions and the "has" is shallower, the joint probability of the former is still greater than that of the latter because the probability of the former’s parent node is too large (our previous experiments can also prove that in distributions with lower entropy, the node with the highest probability is overestimated). Consequently, in subsequent recalls, the "has" node is also unlikely to be selected, even though it is on the second level.

Therefore, directly using joint probability as confidence cannot handle complex situations and will inevitably lead to misjudgments. This necessitates that we consider more features like entropy and depth when designing the confidence function. However, introducing more features makes the function design more challenging, and low-dimensional functions are not robust. Naturally, we thought of introducing a learnable neural network to fit an efficient confidence function for us. To this end, we propose an efficient method using Classifier to build Trees named C2T in speculative decoding. Please refer to Appendix D for how the classifier addresses the two situations mentioned above.

Figure 3: This figure provides further details on C2T. The classifier is a two-layer FFN, represented by the rounded rectangle labeled "Cls". It uses the joint probability P𝑃Pitalic_P, the entropy H𝐻Hitalic_H, and its depth d𝑑ditalic_d as features, which are depicted as the larger yellow circles, where bold text represents the correct answers. The classifier outputs a logit as the confidence score C𝐶Citalic_C, shown as the smaller circle, where blue represents the candidate tokens, and red represents the tokens that were not recalled. Then a threshold β𝛽\betaitalic_β = 0.5 and TopK𝐾Kitalic_K = 2 is used for pre-pruning. Tokens above β𝛽\betaitalic_β will be used as input for the next round of draft model Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT to generate the next tree layer. The features for the next layer of tokens can be obtained from the generation probability p𝑝pitalic_p output by the draft model Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT.

4 Methodology

Our primary goal is to maintain the accept length with fewer candidate tokens using a lightweight classifier. To enhance its versatility, the training features should be based solely on the token tree’s properties, without relying on model or dataset information. This classifier will serve as a predictor for pre-pruning during token tree construction, helping create a concise tree efficiently.

4.1 Classifier

The ultimate goal of our method is to reduce latency, so the classifier must be as lightweight as possible. At the same time, we aim for a plug-and-play solution where a model, trained on one dataset, remains effective when applied to others, avoiding features that are strongly tied to specific datasets or models, such as hidden states. Instead, we should use only the mathematical properties of nodes as features for the classifier. Additionally, these features should be easy to compute and minimal in number.

We adopt a two-layer Feed-Forward Network (FFN) as the classifier to achieve our goal, using ReLU as the activation function and sigmoid for normalizing the final logits. This classifier is extremely simple, with minimal parameters and computational load. For features, we start with the joint probability of each node, as methods like EAGLE-2 have shown their effectiveness as a confidence measure. Next, we select the entropy of the probability distribution in which each node resides as the third feature, as lower entropy indicates a higher acceptance rate. Finally, we include the depth of each node, since shallower nodes have a higher likelihood of acceptance. To simplify calculations, we only consider the top 1000 probabilities of each node’s child nodes when calculating entropy. This simplification is necessary and almost lossless, and the detailed proof can be found in Appendix C. By selecting the top 1000 based on probability, we can almost certainly include the topK of the confidence scores (K is typically much smaller than 1000).

The introduction of entropy and depth is not only because they are related to the acceptance rate, but also because the dimensionality-increased node information is more conducive to solving the problems raised by only using joint probability in the motivation. More details in Appendix D.

With these features, a complete classifier structure is obtained. More details of training settings are shown in Appendix B.

4.2 Tree Construction

C2T is a pre-pruning approach, which can be divided into two steps: the first pruning based on confidence and the second pruning based on topK.

4.2.1 First Pruning based on Confidence

As shown in Figure 3, after obtaining the classifier, we can use it to build the token tree. The tree construction based on the classifier is a layer-by-layer construction process. The draft model can obtain the generation probability of each node during each forward pass. Let a node be i𝑖iitalic_i, and the generation probability of this node is pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, then the joint probability of this node is denoted as

Pi=∏j∈P⁢a⁢t⁢h⁢(r⁢o⁢o⁢t,i)pjsubscript𝑃𝑖subscriptproduct𝑗𝑃𝑎𝑡ℎ𝑟𝑜𝑜𝑡𝑖subscript𝑝𝑗P_{i}=\prod_{j\in Path(root,i)}p_{j}italic_P start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = ∏ start_POSTSUBSCRIPT italic_j ∈ italic_P italic_a italic_t italic_h ( italic_r italic_o italic_o italic_t , italic_i ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

(1)

Where the P⁢a⁢t⁢h⁢(r⁢o⁢o⁢t,ti)𝑃𝑎𝑡ℎ𝑟𝑜𝑜𝑡subscript𝑡𝑖Path(root,t_{i})italic_P italic_a italic_t italic_h ( italic_r italic_o italic_o italic_t , italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) represents the set of all nodes on the path from the root node to node i𝑖iitalic_i.

At this point, we can also determine the entropy of node i𝑖iitalic_i’s probability distribution.

Hisubscript𝐻𝑖\displaystyle H_{i}italic_H start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
=−∑j∈S⁢(i)pj⁢log⁡pjabsentsubscript𝑗𝑆𝑖subscript𝑝𝑗subscript𝑝𝑗\displaystyle=-\sum_{j\in S(i)}p_{j}\log p_{j}= - ∑ start_POSTSUBSCRIPT italic_j ∈ italic_S ( italic_i ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_log italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

(2)

≈−∑j∈S1000⁢(i)pj⁢log⁡pjabsentsubscript𝑗subscript𝑆1000𝑖subscript𝑝𝑗subscript𝑝𝑗\displaystyle\approx-\sum_{j\in S_{1000}(i)}p_{j}\log p_{j}≈ - ∑ start_POSTSUBSCRIPT italic_j ∈ italic_S start_POSTSUBSCRIPT 1000 end_POSTSUBSCRIPT ( italic_i ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_log italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

(3)

Where S⁢(i)𝑆𝑖S(i)italic_S ( italic_i ) represents the probability distribution of i𝑖iitalic_i, and S1000⁢(i)subscript𝑆1000𝑖S_{1000}(i)italic_S start_POSTSUBSCRIPT 1000 end_POSTSUBSCRIPT ( italic_i ) represents the set of nodes with top 1000 probabilities.

We can also easily record the depth of node disubscript𝑑𝑖d_{i}italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

After obtaining these three features, we can use the trained classifier to obtain the confidence of node i𝑖iitalic_i, denoted as

Ci=F⁢(Pi,Hi,di)subscript𝐶𝑖𝐹subscript𝑃𝑖subscript𝐻𝑖subscript𝑑𝑖C_{i}=F(P_{i},H_{i},d_{i})italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_F ( italic_P start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )

(4)

Input: draft model Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT, root node r𝑟ritalic_r, maximum depth dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT, threshold β𝛽\betaitalic_β

Result: token tree T𝑇Titalic_T

1
current depth d←0←𝑑0d\leftarrow 0italic_d ← 0

2
confidence set C←{Cr=1}←𝐶subscript𝐶𝑟1C\leftarrow\{C_{r}=1\}italic_C ← { italic_C start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT = 1 }

3
token tree T←{r}←𝑇𝑟T\leftarrow\{r\}italic_T ← { italic_r }

4
current tree layer node set N←{r}←𝑁𝑟N\leftarrow\{r\}italic_N ← { italic_r }

5
while d<dm⁢a⁢x𝑑subscript𝑑𝑚𝑎𝑥d<d_{max}italic_d < italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT and exists Ci>βsubscript𝐶𝑖𝛽C_{i}>\betaitalic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT > italic_β in C𝐶Citalic_C do

6
N←t⁢o⁢p⁢K⁢(Md⁢(N))←𝑁𝑡𝑜𝑝𝐾subscript𝑀𝑑𝑁N\leftarrow topK(M_{d}(N))italic_N ← italic_t italic_o italic_p italic_K ( italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT ( italic_N ) )

7
C←{}←𝐶
C\leftarrow\{\}italic_C ← { }

8
for i𝑖iitalic_i in N𝑁Nitalic_N do

9
Pi←∏j∈P⁢a⁢t⁢h⁢(r⁢o⁢o⁢t,i)pj←subscript𝑃𝑖subscriptproduct𝑗𝑃𝑎𝑡ℎ𝑟𝑜𝑜𝑡𝑖subscript𝑝𝑗P_{i}\leftarrow\prod_{j\in Path(root,i)}p_{j}italic_P start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← ∏ start_POSTSUBSCRIPT italic_j ∈ italic_P italic_a italic_t italic_h ( italic_r italic_o italic_o italic_t , italic_i ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

10
di←d←subscript𝑑𝑖𝑑d_{i}\leftarrow ditalic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_d

11
Hi←−∑j∈S1000⁢(i−1)pj⁢log⁡pj←subscript𝐻𝑖subscript𝑗subscript𝑆1000𝑖1subscript𝑝𝑗subscript𝑝𝑗H_{i}\leftarrow-\sum_{j\in S_{1000}(i-1)}p_{j}\log p_{j}italic_H start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← - ∑ start_POSTSUBSCRIPT italic_j ∈ italic_S start_POSTSUBSCRIPT 1000 end_POSTSUBSCRIPT ( italic_i - 1 ) end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_log italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

12
Ci←F⁢(Pi,Hi,di)←subscript𝐶𝑖𝐹subscript𝑃𝑖subscript𝐻𝑖subscript𝑑𝑖C_{i}\leftarrow F(P_{i},H_{i},d_{i})italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_F ( italic_P start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )

13
Append Cisubscript𝐶𝑖C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT to C𝐶Citalic_C

14
if Ci<βsubscript𝐶𝑖𝛽C_{i}<\betaitalic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT < italic_β then

15
Remove node i𝑖iitalic_i from N𝑁Nitalic_N

16

17             end if

18

19       end for

20      d←d+1←𝑑
𝑑1d\leftarrow d+1italic_d ← italic_d + 1

21
extend N𝑁Nitalic_N or t⁢o⁢p⁢K⁢(N)𝑡𝑜𝑝𝐾𝑁topK(N)italic_t italic_o italic_p italic_K ( italic_N ) to T𝑇Titalic_T

22

23 end while

24return T𝑇Titalic_T

Algorithm 1 C2T

Where F⁢(∗)𝐹F(*)italic_F ( ∗ ) represents the classifier’s output. Since Cisubscript𝐶𝑖C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is normalized, we set a threshold β𝛽\betaitalic_β between 0 and 1 to determine which tokens participate in generating the next tree layer. This screening-generation process repeats at each tree layer until the current depth disubscript𝑑𝑖d_{i}italic_d start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT reaches the maximum depth dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT we set, or until no nodes in the current tree layer have confidence greater than β𝛽\betaitalic_β.

4.2.2 Second Pruning based on Topk

In our experiments, setting β𝛽\betaitalic_β appropriately ensures that the classifier does not qualify too many tokens at each tree layer, making the naive generation method sufficient. However, in practical applications, we often need to set TopK𝐾Kitalic_K to manage tree generation. TopK𝐾Kitalic_K serves two main purposes:

1.

To reduce the classifier’s computational cost, we calculate confidence only for the TopK𝐾Kitalic_K tokens with the highest generation probability.

2.

To prevent excessive tree expansion, we limit the number of tokens participating in the next tree layer’s generation. After identifying the tokens that pass the classifier test in the current tree layer, we select only the TopK𝐾Kitalic_K tokens with the highest confidence as the final candidates.

Appendix B.6 shows that when β𝛽\betaitalic_β is sufficiently large, the use of TopK𝐾Kitalic_K does not impact the method’s effectiveness. However, when β𝛽\betaitalic_β is less strict, TopK𝐾Kitalic_K helps prevent the number of candidate tokens from becoming too large, though it may also limit the algorithm’s maximum performance. Generally, the first strategy is essential for reducing classifier costs proven in Appendix C Proof-2, while the second can be applied as needed, with the size of TopK𝐾Kitalic_K adjusted based on GPU’s capabilities or omitted if unnecessary.

5 Experiments

Models: We used LLaMA-2-Chat 7B, 13B, 70B Touvron et al. (2023) and Vicuna 7B, 13B, 33B Chiang et al. (2023) as target model Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, and the corresponding draft model Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT is from EAGLE Li et al. (2024b).

Tasks: To compare with EAGLE-2 Li et al. (2024a), we aligned with it on the dataset. For tasks such as multi-round dialogue, code generation, mathematical reasoning, instruction following, summarization, and Q&A, we selected the MT-bench Zheng et al. (2023), HumanEval Chen et al. (2021), GSM8K Cobbe et al. (2021), Alpaca Taori et al. (2023), CNN/Daily Mail Nallapati et al. (2016), and Natural Questions Kwiatkowski et al. (2019), respectively.

Metrics: The ultimate goal of this work is to let the Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT verify as few candidate tokens as possible while maintaining the hit rate unchanged or even better, so we mainly focus on the following two device-independent indicators:

•

The number of candidate tokens γ𝛾\gammaitalic_γ: The total number of tokens verified by Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT.

•

Accept length τ𝜏\tauitalic_τ: The average length accepted by Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT for each generation, and this indicator in this paper does not include the initial token generated by the target model itself, so it needs to be added 1 compared to some papers.

In this paper, the experiment with the LLaMA-2 7B model focuses on precision rather than inference time, as the method’s main advantage is obtaining a more accurate token tree at minimal cost. For powerful GPUs and lightweight LLMs, if the GPU’s parallel computing capability threshold is not reached, this optimization may not improve time performance. So we will demonstrate the efficiency advantage of this method on the LLaMA-2 70B model. When analyzing inference time, there will be the following two indicators:

•

Draft time: The total time measured from the completion of the last verification to the generation of the next verification input on benchmark data.

•

Verify time: The total time used for verification on benchmark data.

Comparison: This work will mainly compare with the SOTA dynamic strategy, EAGLE-2 Li et al. (2024a). The N𝑁Nitalic_N will be our variable to control γ𝛾\gammaitalic_γ and τ𝜏\tauitalic_τ. To make the comparison fairer, we will set EAGLE-2’s K=15𝐾15K=15italic_K = 15 and dm⁢a⁢x=10subscript𝑑𝑚𝑎𝑥10d_{max}=10italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 10. This parameter configuration nearly reaches the limit of EAGLE-2’s Mdsubscript𝑀𝑑M_{d}italic_M start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT capabilities.

Figure 4: The scatter plot uses the LLaMA-2 7B model, with the acceptance length τ𝜏\tauitalic_τ on the x-axis and the number of candidate tokens γ𝛾\gammaitalic_γ on the y-axis. Cls represents our classifier-based method C2T, E2 represents EAGLE-2, w/o represents not using topK secondary pruning, and wK𝐾Kitalic_K represents the use of TopK𝐾Kitalic_K secondary pruning with K𝐾Kitalic_K values of 15, 20, 25, and 30.

5.1 Feature Ablation

We conducted ablation studies on three features: joint probability, entropy, and depth. The model trained with all three features served as the baseline, while other combinations were used as control groups. As shown in Table 1, the joint probability is the most critical factor for performance improvement. In contrast, entropy and depth alone lead to significant performance degradation. Entropy and depth primarily serve as corrective factors, while joint probability is essential for confidence scoring. Combining joint probability with either entropy or depth slightly outperforms EAGLE-2. However, using all three features together significantly enhances performance, demonstrating their strong complementarity.

Method
Setting
Feature
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

E-2
N=100𝑁100N=100italic_N = 100
/
//
3.89
1.0M

C2T
β=0.42𝛽0.42\beta=0.42italic_β = 0.42
P+H
3.92
0.9M

β=0.7𝛽0.7\beta=0.7italic_β = 0.7
P+d
3.92
0.9M

β=0.5𝛽0.5\beta=0.5italic_β = 0.5
H+d
0.98
3.3M

β=0.5𝛽0.5\beta=0.5italic_β = 0.5
P+H+d
3.92
0.7M

Table 1: Regarding the ablation experiments of the classifier features, E-2 represents EAGLE-2. For both methods, we set the default dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT to 10 and topK to 15. All experiments were conducted using LLaMA-2 7B on MT-bench. In the features, P stands for joint probability, H for entropy, and d for depth. P+H+d indicates training the classifier using joint probability, entropy, and depth as features, and so on. M represents a million. τ𝜏\tauitalic_τ represents accept length and γ𝛾\gammaitalic_γ represents the number of candidate tokens.

5.2 Dataset Transferability

EAGLE-2, being a post-pruning method, directly controls γ𝛾\gammaitalic_γ by adjusting TopN𝑁Nitalic_N. In contrast, our pre-pruning approach controls γ𝛾\gammaitalic_γ by setting a pruning threshold β𝛽\betaitalic_β during tree generation. Thus, a direct step-by-step comparison is not feasible. Instead, we compare the precision of both methods using a scatter plot of τ𝜏\tauitalic_τ versus γ𝛾\gammaitalic_γ. To further validate transferability and data independence, we transferred the classifier’s parameters to other datasets without fine-tuning.

As shown in Figure 4, C2T consistently achieves lower γ𝛾\gammaitalic_γ for similar τ𝜏\tauitalic_τ compared to EAGLE-2, indicating superior precision. Moreover, C2T’s curve is more gradual, showing its increasing advantage as τ𝜏\tauitalic_τ grows. Cross-validation is in Appendix B.5.

Model
Method
Setting
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

L2 7B
E-2
N=290𝑁290N=290italic_N = 290
4.40
2.6M

C2T
β=0.3𝛽0.3\beta=0.3italic_β = 0.3
4.41
2.0M

L2 13B
E-2
N=300𝑁300N=300italic_N = 300
4.58
2.6M

C2T
β=0.3𝛽0.3\beta=0.3italic_β = 0.3
4.59
1.9M

L2 70B
E-2
N=260𝑁260N=260italic_N = 260
4.08
2.5M

C2T
β=0.3𝛽0.3\beta=0.3italic_β = 0.3
4.08
2.0M

V 7B
E-2
N=180𝑁180N=180italic_N = 180
5.25
1.5M

C2T
β=0.35𝛽0.35\beta=0.35italic_β = 0.35
5.30
1.3M

C2T*
β=0.25𝛽0.25\beta=0.25italic_β = 0.25
5.32
1.1M

V 13B
E-2
N=200𝑁200N=200italic_N = 200
4.02
1.4M

C2T
β=0.35𝛽0.35\beta=0.35italic_β = 0.35
4.03
1.2M

C2T*
β=0.25𝛽0.25\beta=0.25italic_β = 0.25
4.03
1.1M

V 33B
E-2
N=210𝑁210N=210italic_N = 210
3.70
2.0M

C2T
β=0.3𝛽0.3\beta=0.3italic_β = 0.3
3.72
2.0M

C2T*
β=0.20𝛽0.20\beta=0.20italic_β = 0.20
3.74
1.8M

Table 2: Comparison of the inference performance of Eagle-2 and our naive method without second TopK pruning on different models using MT-bench. L2 represents LLaMA-2, V represents Vicuna, E-2 represents EAGLE-2, C2T represents our naive method using the Classifier trained on the token trees inferred by LLaMA-2 7B, and C2T* represents using the Classifier fine-tuned on the token trees inferred by Vicuna-7B. M represents a million.

5.3 Model Transferability

We trained the classifier on LLaMA-2 7B’s token tree and tested its transferability to other models (LLaMA-2 13B, 70B, Vicuna 7B, 13B, and 33B) by freezing its parameters. Results in Table 2 show that C2T performs well on LLaMA-2 models that stably utilizes 75% to 80% of γ𝛾\gammaitalic_γ while keeping τ𝜏\tauitalic_τ unchanged. But the ability has declined on Vicuna models, though still better than EAGLE-2. When the classifier is fine-tuned on Vicuna’s token trees, its performance comes back to the level achieved on the LLaMA-2 model family. The cost of this fine-tuning is minimal. For details, please refer to Appendix B.4.

In summary, C2T has strong transferability within the same model family to still obtain good performance but may require fine-tuning for optimal performance when transferring to a different model family.

(a) Relationship Graph between Verify Tokens and Time

(b) Comparison Graph of EAGLE-2 and Ours

Figure 5: The graph shows the relationship between candidate tokens number γ𝛾\gammaitalic_γ and total wall-clock time on MT-bench using LLaMA-2 70B, with topK=15 and dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT=10. Total time is split into draft time (D/Time) and verify time (V/Time), shown as bar chart subparts on the left axis, while γ𝛾\gammaitalic_γ (V/Tokens) are shown as a line chart on the right axis. The first figure uses topN as the x-axis, and the second figure uses accept length τ𝜏\tauitalic_τ to align the methods.

5.4 Speed up in Larger LLMs

To evaluate the time efficiency of C2T under GPU limits, we tested the LLaMA-2 70B model on 2 * A100 (80G) GPUs. Figure 5(a) shows that with EAGLE-2, as TopN𝑁Nitalic_N increases, γ𝛾\gammaitalic_γ grows linearly but verify time increases in steps, not linearly.

We compared C2T with EAGLE-2 by matching τ𝜏\tauitalic_τ. Figure 5(b) shows that C2T has a slightly higher draft time, but lower γ𝛾\gammaitalic_γ resulting in a time advantage. When the parallel computing capability of GPUs is pushed to the limit, our method reduces the time by 18% compared to EAGLE-2 under the same τ𝜏\tauitalic_τ. For detailed latency analysis, see Appendix E.1.

5.5 Benefits in Chain Mode

Method
Avg length
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

EAGLE 1/2
5
1.98
112231

6
2.06
125992

7
2.10
140238

8
2.09
155420

9
2.13
184140

DyMax
6.10
2.06
127018

DyJoint
6.33
2.12
129876

C2T
5.46
2.12
116694

Table 3: The comparison between other methods and C2T with β=0.85𝛽0.85\beta=0.85italic_β = 0.85 in chain mode using the LLaMA-2 7B model on the MT-bench. DyMax and DyJoint represent the dynamic methods using the maximum probability with threshold=0.3 and joint probability with threshold=0.08 as the criterion for early stopping respectively. The maximum depth for all the dynamic methods is 10. Avg length represents the average generation length without the initial token.

Experiments in chain mode are meaningful because current dynamic tree construction does not support batch sizes greater than 1. This is due to the inability to have different attention masks within the same batch. In contrast, a chained token tree is always compatible with multi-batch scenarios.

In chain mode, C2T degrades as an early exit strategy and EAGLE-2 is rendered ineffective, essentially reverting to EAGLE-1. We varied the maximum draft length from 5 to 9 tokens for EAGLE-1/2. Additionally, we compared dynamic methods using the maximum probability and joint probability as early stopping criteria. The results, shown in Table 3, demonstrate that C2T retains an advantage in chain mode.

6 Conclusion

In this paper, we propose C2T to address the limitations of previous tree construction methods that rely solely on joint probability. By training a classifier with additional features, we improved the precision of tree construction. We conducted extensive evaluations on multiple benchmarks and various LLMs to compare with the SOTA method EAGLE-1/2. Our method achieved superior results in all experiments and demonstrated strong transferability and applicability. C2T can construct a more precise tree using 75% to 80% of the tokens while maintaining the acceptance length. When the parallel computing capability of GPUs is pushed to the limit, C2T reduces the time by 18% compared to EAGLE-2 under the same acceptance length.

Limitation

C2T, similar to other dynamic tree construction approaches, currently does not support batch sizes greater than 1 (bs > 1) due to the use of different tree masks within the same batch, which is not supported by existing engineering implementations. However, C2T supports early stopping in chain mode, which is compatible with bs > 1. In practical industry use, bs > 1 is typically used in conjunction with chain mode Liu et al. (2024). And the preliminary combo experiments of the MTP-style layer in DeepSeek-V3 and C2T are in Appendix F.

Compared to methods using joint probability directly, C2T adds minimal overhead. This overhead is negligible when verification time is significant. And the quantitative analysis shows that the additional FLOPs are negligible and imply potential for optimization in engineering implementation. Please refer to Appendix E.2 for details.

Ethics Statement

Our research adheres to the ACL Code of Ethics. We have ensured that our work respects user privacy and does not include any personal information in the datasets used. The datasets are publicly available and were labeled through interactions with English-speaking users. The tools and models used in this study are utilized in compliance with their intended purposes and are accessible under permissive licenses. We are committed to upholding the ethical standards of the ACL and promoting responsible research practices within the NLP community.

References

Achiam et al. (2023)

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023.

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774.

Brown et al. (2024)

Oscar Brown, Zhengjie Wang, Andrea Do, Nikhil Mathew, and Cheng Yu. 2024.

Dynamic depth decoding: Faster speculative decoding for llms.

arXiv preprint arXiv:2409.00142.

Brown et al. (2020)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.

Language models are few-shot learners.

Advances in neural information processing systems, 33:1877–1901.

Cai et al. (2024)

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. 2024.

Medusa: Simple llm inference acceleration framework with multiple decoding heads.

arXiv preprint arXiv:2401.10774.

Chen et al. (2023)

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023.

Accelerating large language model decoding with speculative sampling.

arXiv preprint arXiv:2302.01318.

Chen et al. (2021)

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021.

Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374.

Chen et al. (2024)

Zhuoming Chen, Avner May, Ruslan Svirschevski, Yu-Hsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. 2024.

Sequoia: Scalable and robust speculative decoding.

In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Chiang et al. (2023)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023.

Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality.

See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6.

Cobbe et al. (2021)

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021.

Training verifiers to solve math word problems.

arXiv preprint arXiv:2110.14168.

He et al. (2023)

Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He. 2023.

Rest: Retrieval-based speculative decoding.

arXiv preprint arXiv:2311.08252.

Huang et al. (2024)

Kaixuan Huang, Xudong Guo, and Mengdi Wang. 2024.

Specdec++: Boosting speculative decoding via adaptive candidate lengths.

arXiv preprint arXiv:2405.19715.

Kwiatkowski et al. (2019)

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019.

Natural questions: a benchmark for question answering research.

Transactions of the Association for Computational Linguistics, 7:453–466.

Leviathan et al. (2023)

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023.

Fast inference from transformers via speculative decoding.

In International Conference on Machine Learning, pages 19274–19286. PMLR.

Li et al. (2024a)

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a.

EAGLE-2: Faster inference of language models with dynamic draft trees.

In Empirical Methods in Natural Language Processing.

Li et al. (2024b)

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b.

EAGLE: Speculative sampling requires rethinking feature uncertainty.

In International Conference on Machine Learning.

Liu et al. (2024)

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024.

Deepseek-v3 technical report.

arXiv preprint arXiv:2412.19437.

Mamou et al. (2024)

Jonathan Mamou, Oren Pereg, Daniel Korat, Moshe Berchansky, Nadav Timor, Moshe Wasserblat, and Roy Schwartz. 2024.

Dynamic speculation lookahead accelerates speculative decoding of large language models.

arXiv preprint arXiv:2405.04304.

Miao et al. (2024)

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. 2024.

Specinfer: Accelerating large language model serving with tree-based speculative inference and verification.

In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pages 932–949.

Nallapati et al. (2016)

Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. 2016.

Abstractive text summarization using sequence-to-sequence rnns and beyond.

arXiv preprint arXiv:1602.06023.

Patterson (2004)

David A Patterson. 2004.

Latency lags bandwith.

Communications of the ACM, 47(10):71–75.

Qin et al. (2024)

Zongyue Qin, Zifan He, Neha Prakriya, Jason Cong, and Yizhou Sun. 2024.

Dynamic-width speculative beam decoding for efficient llm inference.

arXiv preprint arXiv:2409.16560.

Radford et al. (2019)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019.

Language models are unsupervised multitask learners.

OpenAI blog, 1(8):9.

Shazeer (2019)

Noam Shazeer. 2019.

Fast transformer decoding: One write-head is all you need.

arXiv preprint arXiv:1911.02150.

Sun et al. (2024)

Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix Yu. 2024.

Spectr: Fast speculative decoding via optimal transport.

Advances in Neural Information Processing Systems, 36.

Svirschevski et al. (2024)

Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin. 2024.

Specexec: Massively parallel speculative decoding for interactive llm inference on consumer devices.

arXiv preprint arXiv:2406.02532.

Taori et al. (2023)

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023.

Stanford alpaca: An instruction-following llama model.

Touvron et al. (2023)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.

Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971.

Wang et al. (2024)

Jikai Wang, Yi Su, Juntao Li, Qinrong Xia, Zi Ye, Xinyu Duan, Zhefeng Wang, and Min Zhang. 2024.

Opt-tree: Speculative decoding with adaptive draft tree structure.

Preprint, arXiv:2406.17276.

Zhang and Sennrich (2019)

Biao Zhang and Rico Sennrich. 2019.

Root mean square layer normalization.

Advances in Neural Information Processing Systems, 32.

Zhang et al. (2024)

Situo Zhang, Hankun Wang, Da Ma, Zichen Zhu, Lu Chen, Kunyao Lan, and Kai Yu. 2024.

Adaeagle: Optimizing speculative decoding via explicit modeling of adaptive draft structures.

arXiv preprint arXiv:2412.18910.

Zheng et al. (2023)

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023.

Judging llm-as-a-judge with mt-bench and chatbot arena.

Advances in Neural Information Processing Systems, 36:46595–46623.

Appendix A EAGLE-1 and EAGLE-2

We conducted our experiments on LLaMA-2 7B and MT-bench. According to Table 1 of EAGLE-2 Li et al. (2024a), the accept length of EAGLE-1 Li et al. (2024b) is 3.62, and that of EAGLE-2 is 4.70. We also obtained similar results in our reproduction. However, during the experiment, we found that this comparison is not fair. Based on Appendix A of EAGLE-1 paper and EAGLE-2 paper, we can obtain the shapes of their respective token trees, where the size of the token tree in EAGLE-1 is 26, while that in EAGLE-2 is 60. Therefore, we further aligned the sizes of the token trees of both models and introduced our method C2T. The experimental results are shown in Figure 4.

method
setting
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

E-1
TopN𝑁Nitalic_N=26
2.66
340054

E-2
TopN𝑁Nitalic_N=26
2.95
317668

TopN𝑁Nitalic_N=60
3.65
628980

C2T

β𝛽\betaitalic_β=0.85
3.06
226569

β𝛽\betaitalic_β=0.65
3.66
531947

Table 4: The experiments on LLaMA-2 7B and MT-bench regarding the capability comparison between EAGLE-1, EAGLE-2, and C2T. TopN represents the size of the token tree, τ𝜏\tauitalic_τ represents the accept length, and γ𝛾\gammaitalic_γ represents the total number of candidate tokens to be verified. For EAGLE-2 and C2T, TopK𝐾Kitalic_K=10 and dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT=6. It should be noted that our τ𝜏\tauitalic_τ does not include the initial token generated by the target model, which is always accepted. Therefore, our τ𝜏\tauitalic_τ is 1 smaller than that reported in the EAGLE-2 paper.

After aligning the tree sizes, EAGLE-2’s performance falls short of expectations, with only an 11% improvement in accept length. In practice, dynamic methods incur additional costs, leading to worse wall-clock times (on 2 A100 80G GPUs). While dynamic methods generate more accurate token trees than static methods, the extra computational cost means the tree size must be increased to achieve a speedup. Essentially, dynamic methods optimize GPU utilization further, as manually designing larger trees is extremely difficult and impractical for complex scenarios. However, given GPU limitations, increasing tree size also increases the verification burden on the target model. Thus, C2T’s ability to generate more compact trees while maintaining the same accept length is particularly valuable.

Appendix B Classifier

Figure 6: The training process of FFNs with different structures, where HSn𝑛nitalic_n represents a two-layer FFN with hidden state size n𝑛nitalic_n.

Figure 7: The fine-tuning process of the classifier with the benchmark.

B.1 Dataset

The classifier used in this paper, if not specifically mentioned, is trained on the token tree generated by LLaMA-2 7B on the MT-bench using the EAGLE-2 strategy. The settings are dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 11 (excluding the root generated by Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT), TopK𝐾Kitalic_K = 10, and the TopN𝑁Nitalic_N = 1011 (meaning no recall is performed during the rerank stage, and the complete tree is used as the training dataset). Each data entry uses joint probability, depth, and entropy as features, and whether it is accepted as the label. We simply clean the data by dropping entries containing NA values, resulting in 8880 token trees, each containing 1011 nodes, for a total of 8880 * 1011 = 8,977,680 training data entries.

B.2 Training and Evaluation

We split the dataset in a ratio of 0.95:0.05. Since the dataset has sparse positive samples, for example, each token tree has 1011 nodes, however, only 3.5 tokens are accepted by the Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, so we perform negative sampling on this sparse dataset. During training, we set the batch size to 1024, and during evaluation, to align with the token tree verification configuration, we set the batch size to 1011. We use Adam as the optimizer with a learning rate (lr) of 1×10−31superscript1031\times 10^{-3}1 × 10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT and train for 10 epochs, and use BCE as the criterion.

For evaluation, we focus more on recall, meaning the classifier should try to recall all tokens that are ultimately accepted by Mtsubscript𝑀𝑡M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT. At the same time, we should also pay attention to the positive rate, which is the probability that the classifier predicts a token as positive, denoted as θ𝜃\thetaitalic_θ. This value corresponds to the ratio of TopN𝑁Nitalic_N to the size of T1subscript𝑇1T_{1}italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT in EAGLE-2 and is positively correlated with the final γ𝛾\gammaitalic_γ. When selecting the classifier, priority should be given to models with significantly higher recall. Among models with similar recall, choose the one with a smaller θ𝜃\thetaitalic_θ.

B.3 Structure

In this paper, we also briefly explored the effects of classifiers with different FFN structures. We mainly discussed the performance of FFNs with two layers and different hidden states, setting the hidden state of the classifier to 2, 6, 12, 24, 36, and 48, respectively. We trained various FFNs according to the training configuration in B.2, and the training process is shown in Figure 6:

Furthermore, we applied these classifiers to the C2T inference of LLaMA-2 7B on MT-bench, aligning the threshold β=0.5𝛽0.5\beta=0.5italic_β = 0.5, It can be observed that, with similar γ𝛾\gammaitalic_γ, the τ𝜏\tauitalic_τ for the six classifier structures are 3.91, 3.97, 4.02, 4.03, 4.02, and 4.02, respectively. Therefore, it can be concluded that for our task, a hidden state of 12 to 48 is more appropriate for the classifier. In our other experiments, this value is set to 48 by default.

B.4 Fine-tuning

In this paper, we also explored the fine-tuning, using Adam as the optimizer with a lr of 1×10−41superscript1041\times 10^{-4}1 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT and training for 10 epochs using 10% of the data. In addition to fine-tuning in the Experiment 5.3 testing model transferability, we also attempted fine-tuning between different token trees inferred on different benchmarks. The fine-tuning process is shown in Figure 7. As shown in the figure, although fine-tuning improves the classifier’s recall, it also increases the positive rate. When we applied the fine-tuned classifier to inference, we found that the distribution relationship between candidate tokens number γ𝛾\gammaitalic_γ and accept length τ𝜏\tauitalic_τ remained almost unchanged. The only difference is that the fine-tuned classifier requires a larger β𝛽\betaitalic_β to achieve the same γ𝛾\gammaitalic_γ and τ𝜏\tauitalic_τ as before. This means that the fine-tuned classifier becomes more confident, but there is no significant improvement in precision. This also indirectly proves the data-free characteristic of our classifier.

B.5 Cross Validation

Since our previous experiments involved inferring token trees on MT-bench to train the classifier, and then transferring the classifier to other datasets for speculative decoding, we now cross-validate the effectiveness of C2T on MT-bench. To do this, we train the classifier from scratch using token trees generated from other datasets and then apply it to inference on MT-bench. This experiment was conducted on LLaMA-2 7B. The results are shown in Figure 8. Classifiers trained on other datasets and those trained directly on MT-bench show nearly identical distributions in the scatter plots when used for inference with C2T. This cross-validates the feasibility of C2T on MT-bench.

Figure 8: Scatter plots of candidate tokens to γ𝛾\gammaitalic_γ and accept length τ𝜏\tauitalic_τ obtained by C2T on MT-bench using classifiers trained on different datasets.

B.6 Use TopK for Second Pruning

β𝛽\betaitalic_β

K𝐾Kitalic_K

τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

0.3
15
3.98
1026454

20
4.15
1235635

25
4.21
1413190

30
4.30
1568173

/
//
4.41
2030064

0.5
15
3.92
781806

20
4.01
878722

25
4.02
931429

30
4.02
971486

/
//
4.02
983491

Table 5: The topK secondary pruning experiment of LLaMA-2 7B on MT-bench, where K represents the value of topK, /
// represents the naive method without secondary pruning.

In the methodology section, we discussed the improvement of further constraining the tree shape using TopK𝐾Kitalic_K in pre-pruning. In the methodology, we mentioned that using TopK𝐾Kitalic_K to simplify the computation process is necessary, but the latter TopK𝐾Kitalic_K pruning after obtaining the confidence scores is optional. In this experiment, we will conduct a variable analysis for the latter case. This approach results in a smaller and more stable token tree, and by varying TopK𝐾Kitalic_K, we can generate multiple scatter plots. As shown in 4, it is observed that after introducing TopK𝐾Kitalic_K for second pruning, the method maintains a similar distribution on the graph as the original method, and in some threshold values, it even yields slightly better results. Regardless of fluctuations, it consistently outperforms EAGLE-2. As shown in Table 5, under a high β𝛽\betaitalic_β, secondary pruning can get gains in γ𝛾\gammaitalic_γ with minimal loss of τ𝜏\tauitalic_τ, but it also limits the maximum capability under a low β𝛽\betaitalic_β. In our other experiments, we default to using TopK𝐾Kitalic_K=15 for two complete rounds of constrained pruning.

Appendix C Proof of Simplified Calculation

The method proposed in this paper, when calculating entropy, requires first obtaining the topM𝑀Mitalic_M probabilities and then calculating the entropy of these M𝑀Mitalic_M probabilities. Let the vocabulary size be V𝑉Vitalic_V, and it is known that the implementation of torch.topk is based on the quickselect algorithm.

If we consider only the calculation of entropy, obtaining the topM𝑀Mitalic_M probabilities first and then calculating the entropy is more complex than directly calculating the entropy.

Proof-1.

The FLOPs for directly calculating the entropy is F1=2∗Vsubscript𝐹12𝑉F_{1}=2*Vitalic_F start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 2 ∗ italic_V. In contrast, obtaining the topM𝑀Mitalic_M probabilities involves F2=C1∗Vsubscript𝐹2subscript𝐶1𝑉F_{2}=C_{1}*Vitalic_F start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∗ italic_V, after calculating the entropy of M𝑀Mitalic_M probabilities involves F3=F2+2∗M=C1∗V+2∗Msubscript𝐹3
subscript𝐹22𝑀
subscript𝐶1𝑉2𝑀F_{3}=F_{2}+2*M=C_{1}*V+2*Mitalic_F start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT = italic_F start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT + 2 ∗ italic_M = italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∗ italic_V + 2 ∗ italic_M, where C1>2subscript𝐶12C_{1}>2italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT > 2 in most cases. So 2∗V<C1∗V+2∗M2𝑉
subscript𝐶1𝑉2𝑀2*V<C_{1}*V+2*M2 ∗ italic_V < italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∗ italic_V + 2 ∗ italic_M, which means F1<F3subscript𝐹1subscript𝐹3F_{1}<F_{3}italic_F start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT < italic_F start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT. In summary, selecting first and then calculating the entropy is more complex than directly calculating the entropy in most cases.
∎

However, we need to take into account the impact of the joint probability calculation step. Both C2T and EAGLE-2 only need to calculate the joint probabilities of the TopK𝐾Kitalic_K when computing joint probabilities. We first argue the necessity of this step.

Proof-2.

If we were to fully calculate the joint probabilities and then select, since each tree layer has at most K𝐾Kitalic_K nodes, considering the parallel computing capability of GPUs, the overall complexity for calculating the joint probabilities is O⁢(V)𝑂𝑉O(V)italic_O ( italic_V ). There would be K∗V𝐾𝑉K*Vitalic_K ∗ italic_V probabilities in total, and selecting the TopK𝐾Kitalic_K from them would involve O⁢(K∗V)𝑂𝐾𝑉O(K*V)italic_O ( italic_K ∗ italic_V ). Therefore, the total complexity would be O⁢(V+K∗V)𝑂
𝑉𝐾𝑉O(V+K*V)italic_O ( italic_V + italic_K ∗ italic_V ). In contrast, by only taking the TopK𝐾Kitalic_K for each node, due to the parallel computing nature of GPUs, the total complexity for selection is O⁢(V)𝑂𝑉O(V)italic_O ( italic_V ), resulting in K2superscript𝐾2K^{2}italic_K start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT probabilities. The complexity for selecting the TopK𝐾Kitalic_K from these probabilities is O⁢(K2)𝑂superscript𝐾2O(K^{2})italic_O ( italic_K start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ), so the overall complexity is O⁢(V+K2)𝑂
𝑉superscript𝐾2O(V+K^{2})italic_O ( italic_V + italic_K start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ). Since K2<<Vmuch-less-thansuperscript𝐾2𝑉K^{2}<<Vitalic_K start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT < < italic_V, therefore O⁢(V+K∗V)>O⁢(V+K2)𝑂
𝑉𝐾𝑉𝑂
𝑉superscript𝐾2O(V+K*V)>O(V+K^{2})italic_O ( italic_V + italic_K ∗ italic_V ) > italic_O ( italic_V + italic_K start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ). In summary, it is necessary to first select and then calculate the joint probabilities.
∎

Therefore, what we are actually comparing are the complexities of the following two scenarios:

•

Directly calculating the entropy and then selecting the TopK𝐾Kitalic_K probabilities.

•

First selecting the topM𝑀Mitalic_M probabilities, then calculating the entropy of these M𝑀Mitalic_M probabilities, and finally selecting the TopK𝐾Kitalic_K probabilities from these M𝑀Mitalic_M probabilities.

Proof-3.

From Proof-1, we know that for the first scenario, the FLOPs before taking the TopK𝐾Kitalic_K is F1=2∗Vsubscript𝐹12𝑉F_{1}=2*Vitalic_F start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 2 ∗ italic_V, and the FLOPs for taking the TopK𝐾Kitalic_K is F4=C2∗Vsubscript𝐹4subscript𝐶2𝑉F_{4}=C_{2}*Vitalic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT = italic_C start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ∗ italic_V. Therefore, the total FLOPs is F5=(2+C2)∗Vsubscript𝐹5
2subscript𝐶2𝑉F_{5}=(2+C_{2})*Vitalic_F start_POSTSUBSCRIPT 5 end_POSTSUBSCRIPT = ( 2 + italic_C start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) ∗ italic_V. For the second scenario, the FLOPs before taking the TopK𝐾Kitalic_K is F3=C1∗V+2∗Msubscript𝐹3
subscript𝐶1𝑉2𝑀F_{3}=C_{1}*V+2*Mitalic_F start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT = italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∗ italic_V + 2 ∗ italic_M, and the FLOPs for taking the TopK𝐾Kitalic_K from the M𝑀Mitalic_M probabilities is F5=C3∗Msubscript𝐹5subscript𝐶3𝑀F_{5}=C_{3}*Mitalic_F start_POSTSUBSCRIPT 5 end_POSTSUBSCRIPT = italic_C start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ∗ italic_M. Therefore, the total FLOPs is F6=C1∗V+(2+C3)∗Msubscript𝐹6
subscript𝐶1𝑉
2subscript𝐶3𝑀F_{6}=C_{1}*V+(2+C_{3})*Mitalic_F start_POSTSUBSCRIPT 6 end_POSTSUBSCRIPT = italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ∗ italic_V + ( 2 + italic_C start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ) ∗ italic_M. Since for the quickselect algorithm, the final number of computations is independent of the number of elements to be selected, C1≈C2subscript𝐶1subscript𝐶2C_{1}\approx C_{2}italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ≈ italic_C start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. Also, since V>>Mmuch-greater-than𝑉𝑀V>>Mitalic_V > > italic_M, we have F4−F6=(C2−C1)∗V+2∗V−(2+C3)∗M≈2∗V−(2+C3)∗M>0subscript𝐹4subscript𝐹6
subscript𝐶2subscript𝐶1𝑉2𝑉
2subscript𝐶3𝑀2𝑉
2subscript𝐶3𝑀0F_{4}-F_{6}=(C_{2}-C_{1})*V+2*V-(2+C_{3})*M\approx 2*V-(2+C_{3})*M>0italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT - italic_F start_POSTSUBSCRIPT 6 end_POSTSUBSCRIPT = ( italic_C start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT - italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ) ∗ italic_V + 2 ∗ italic_V - ( 2 + italic_C start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ) ∗ italic_M ≈ 2 ∗ italic_V - ( 2 + italic_C start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ) ∗ italic_M > 0, which means F4>F6subscript𝐹4subscript𝐹6F_{4}>F_{6}italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT > italic_F start_POSTSUBSCRIPT 6 end_POSTSUBSCRIPT. In summary, considering the computation of joint probabilities, the FLOPs of the first scenario are more than the second scenario.
∎

We have demonstrated the necessity of simplifying the calculation of entropy, and Proof-3 implies that, from the perspective of reducing FLOPs, M𝑀Mitalic_M should be as small as possible. However, an excessively small M𝑀Mitalic_M may lead to the long-tail effect. Therefore, we conducted experiments on M𝑀Mitalic_M using LLaMA-2 7B on the MT-bench with β=0.5,t⁢o⁢p⁢K=15formulae-sequence𝛽0.5𝑡𝑜𝑝𝐾15\beta=0.5,topK=15italic_β = 0.5 , italic_t italic_o italic_p italic_K = 15, and the results are shown in Table 6. The results indicate that when M=1000𝑀1000M=1000italic_M = 1000, the impact of the long-tail effect is almost completely eliminated.

M𝑀Mitalic_M
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

100
3.70
815365

500
3.89
796981

1000
3.92
781613

10000
3.92
782104

/
//
3.92
781806

Table 6: The experiments on the values of M𝑀Mitalic_M using C2T with LLaMA-2 7B on MT-bench, with β=0.5,t⁢o⁢p⁢K=15formulae-sequence𝛽0.5𝑡𝑜𝑝𝐾15\beta=0.5,topK=15italic_β = 0.5 , italic_t italic_o italic_p italic_K = 15, where /
// indicates no use of topM𝑀Mitalic_M for simplified calculation.

Appendix D Confidence

Figure 9: Input-output 3D-heatmap, where the color becomes lighter as the outputs increase.

Figure 10: Interpolation plots of input-output with some parameters fixed, where the z-axis represents the outputs of the classifier.

Since the features of our classifier are a triplet and the output is a scalar, it is highly suitable for statistical analysis. First, we created a 3D heatmap for the overall input-output by randomly generating 1000 data points, where the joint probability is a float in [0,1], entropy is a float in [0,10], and depth is an integer in [0,10]. To highlight the outputs, the output is not normalized using sigmoid but rather using min-max normalization. The experimental results are shown in Figure 9. Among these three features, the joint probability has the greatest impact on the confidence score, and this fitting result aligns with our intuition. The impact of entropy and depth is not as easily captured, hence we conducted further quantitative analysis.

Furthermore, we fixed one parameter and varied the other two to generate 10,000 data points. The interpolated results are shown in Figure 10. Figure 10 shows that the three surfaces are widely spaced, indicating the significant impact of joint probability on the confidence score. The steeper slope with lower joint probability suggests greater influence from the other features. Figure 10 shows that surfaces are closely adjacent at low entropy (1 or 2), indicating minimal impact on confidence scores. However, at high entropy, surfaces rise above those with lower entropy at low joint probability, showing positive confidence corrections for less confident distributions, addressing the case in Figure 1. Figure 10 shows that surfaces coincide and flatten at high joint probability, indicating minimal effect of depth. Conversely, at low joint probability, surfaces steepen and diverge, with shallower depths yielding higher confidence scores, providing additional opportunities for subsequent node generation, addressing the case in Figure 1.

(a) EAGLE-2 Draft Latency

(b) Ours Draft Latency

Figure 11: Draft latency analysis using the LLaMA-2 model family on MT-bench. The top row is for EAGLE-2, and the bottom row is for Ours. From left to right are the results on LLaMA-2 7B, 13B, and 70B, respectively.

Appendix E Additional Overhead

E.1 Experimental Analysis

Speculative decoding can be divided into two stages: Draft and Verify. Our code is modified based on EAGLE, without changing the Verify part, but only optimizing the Draft part. Therefore, we mainly focus on the latency analysis of the Draft stage. In C2T, for the Draft stage, we can further divide it into several cyclical sub-stages:

1.

Classify: The classifier accepts input features and makes predictions.

2.

Pre-processing: Process the predictions of the classifier and obtain the input for the draft model.

3.

Generate: The draft model accepts input and generates the tokens.

4.

Post-processing: Process the output of the draft model and obtain the input features for the classifier.

For EAGLE-2, we can align the granularity and divide its Draft into three stages:

1.

Generate: The draft model accepts input and produces output.

2.

Post-processing: Process the output of the draft model and obtain the input for the next step.

3.

Rerank: rerank the generated token tree based on joint probability and recall the TopN𝑁Nitalic_N nodes.

We conducted latency analysis for the Draft stage of the LLaMA-2 7B, 13B, and 70B models on 2 * A100 (80G) GPUs to infer MT-bench using C2T with t⁢o⁢p⁢K=15𝑡𝑜𝑝𝐾15topK=15italic_t italic_o italic_p italic_K = 15 for secondary pruning and a threshold β=0.5𝛽0.5\beta=0.5italic_β = 0.5 and EAGLE-2 with t⁢o⁢p⁢N=80𝑡𝑜𝑝𝑁80topN=80italic_t italic_o italic_p italic_N = 80, as shown in Figure 11.

From Figure 11(a), it can be seen that in EAGLE-2, the proportion of Rerank Time significantly increases with the growth of model size. This is because when the GPU’s computational capacity is pushed to its limit, the more complex rerank-recall operations also take longer to complete. From Figure 11(b), in C2T, the proportion of Postprocess Time significantly increases with the model size. Compared to EAGLE-2’s Postprocess Time, we need to additionally calculate entropy and depth, since the latter can be obtained directly, the main impact is caused by the former. Moreover, to prevent the high complexity of calculating entropy due to a large vocabulary, we decompose the topK operation into two stages: top1000 and then topK. We use the 1000 probabilities selected in the first stage to calculate entropy, which introduces additional latency.

Since C2T and EAGLE-1/2 share the same draft model, the Generate Time is aligned between the two methods. The proportion of this item in each method indicates that C2T’s Draft Time is slightly greater than EAGLE-2, but this gap is gradually decreasing with the increase in model size.

E.2 Quantitative Analysis

From the perspective of quantitative analysis, the additional computational overhead introduced by C2T compared to EAGLE-2 primarily consists of the calculation of entropy, the computation of depth, and the output of the classifier. Since the computation of joint probability is inherently coupled with the EAGLE-2’s generation process, and depth can be directly obtained during the construction of the tree structure, both can be considered negligible. Therefore, the focus of our analysis is on the computational complexity of entropy calculation and the forward pass of the classifier. In addition, when calculating the entropy, we first select the topM𝑀Mitalic_M values before computing the entropy. Similarly, for EAGLE-2, the engineering implementation of calculating joint probability also requires selecting the TopK𝐾Kitalic_K values first. According to Appendix C Proof-3, the costs of these two parts can offset each other. Therefore, we only need to consider the complexity introduced by calculating the entropy over M𝑀Mitalic_M values which is O⁢(M)𝑂𝑀O(M)italic_O ( italic_M ).

Then, considering the worst-case scenario, where K𝐾Kitalic_K tokens participate in tree construction at each tree layer with a maximum depth of dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT, the total complexity for entropy calculation is O⁢(K∗dm⁢a⁢x∗M)𝑂𝐾subscript𝑑𝑚𝑎𝑥𝑀O(K*d_{max}*M)italic_O ( italic_K ∗ italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT ∗ italic_M ).

Given that our classifier consists of two layers with a hidden layer size of hℎhitalic_h:

•

the complexity for matrix multiplication from the input to the hidden layer is O⁢(3⁢h)𝑂3ℎO(3h)italic_O ( 3 italic_h ).

•

the activation function computation in the hidden layer is O⁢(h)𝑂ℎO(h)italic_O ( italic_h ).

•

the matrix multiplication from the hidden layer to the output layer is O⁢(h)𝑂ℎO(h)italic_O ( italic_h )

Under the worst-case scenario, the total classification complexity is O⁢(K∗dm⁢a⁢x∗5⁢h)𝑂𝐾subscript𝑑𝑚𝑎𝑥5ℎO(K*d_{max}*5h)italic_O ( italic_K ∗ italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT ∗ 5 italic_h ).

In summary, the overall additional complexity introduced by C2T is O⁢(K∗dm⁢a⁢x∗(M+5⁢h))𝑂𝐾subscript𝑑𝑚𝑎𝑥
𝑀5ℎO(K*d_{max}*(M+5h))italic_O ( italic_K ∗ italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT ∗ ( italic_M + 5 italic_h ) ).

In practical scenarios, with M=1000𝑀1000M=1000italic_M = 1000, K=15𝐾15K=15italic_K = 15, dm⁢a⁢x=10subscript𝑑𝑚𝑎𝑥10d_{max}=10italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 10, h=48ℎ48h=48italic_h = 48, the complexity for entropy calculation is O⁢(150,000)𝑂150000O(150,000)italic_O ( 150 , 000 ), and the complexity for the classifier is O⁢(36,000)𝑂36000O(36,000)italic_O ( 36 , 000 ). The total additional overhead complexity is O⁢(186,000)𝑂186000O(186,000)italic_O ( 186 , 000 ).

This complexity is almost negligible compared to the complexity of a single forward pass of LLMs. Compared with our experimental results, it implies a great space for engineering optimization.

Appendix F Combo with MTP-style Layer

In the latest industry application of speculative decoding, DeepSeek-V3 Liu et al. (2024), inspired by the training method of EAGLE’s draft model, a Multi-Token Prediction (MTP) approach is proposed. Specifically, based on EAGLE, the embeddings of input tokens and the hidden states are firstly normalized by RMSNorm Zhang and Sennrich (2019), then concatenated, and finally dimensionally reduced through the linear head. We froze the parameters of the previously trained classifier and directly transferred it to the inference of the EAGLE layer trained in an MTP-style manner. Preliminary experiment is in Table 7.

model
strategy
setting
τ𝜏\tauitalic_τ
γ𝛾\gammaitalic_γ

EAGLE
C
/
//
1.96
161020

C + C2T

β𝛽\betaitalic_β=0.9
1.97
104141

E-2
TopN𝑁Nitalic_N=90
3.68
935280

C2T

β𝛽\betaitalic_β=0.5
3.70
800848

MTP
C
/
//
2.12
156040

C + C2T

β𝛽\betaitalic_β=0.8
2.10
98141

E-2
TopN𝑁Nitalic_N=90
3.86
900090

C2T

β𝛽\betaitalic_β=0.5
3.84
724913

Table 7: The MTP combo experiments on LLaMA-2 7B and MT-bench. C represents chain-like drafting, C + C2T represents using C2T to early stopping in chain-like drafting, E-2 represent EAGLE-2. The default parameters are TopK𝐾Kitalic_K=15 and dm⁢a⁢xsubscript𝑑𝑚𝑎𝑥d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT=10.

It should be noted that, in order to align with the MTP experimental configuration, the EAGLE layer here is retrained by us, hence there is some data deviation from previous experiments that directly used the model provided by EAGLE.

Generated on Wed Feb 19 10:39:53 2025 by LaTeXML