# AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures

Situo Zhang and Hankun Wang and Da Ma and Zichen Zhu  
Lu Chen\* and Kunyao Lan and Kai Yu\*

X-LANCE Lab, Department of Computer Science and Engineering  
MoE Key Lab of Artificial Intelligence, SJTU AI Institute  
Shanghai Jiao Tong University, Shanghai, China  
{situozhang, wanghankun, chenlusz, kai.yu}@sjtu.edu.cn

## Abstract

Speculative Decoding (SD) is a popular lossless technique for accelerating the inference of Large Language Models (LLMs). We show that the decoding speed of SD frameworks with static draft structures can be significantly improved by incorporating context-aware adaptive draft structures. However, current studies on adaptive draft structures are limited by their performance, modeling approaches, and applicability. In this paper, we introduce AdaEAGLE, the first SD framework that explicitly models adaptive draft structures. AdaEAGLE leverages the Lightweight Draft Length Predictor (LDLP) module to explicitly predict the optimal number of draft tokens during inference to guide the draft model. It achieves comparable speedup results without manual thresholds and allows for deeper, more specialized optimizations. Moreover, together with threshold-based strategies, AdaEAGLE achieves a  $1.62\times$  speedup over the vanilla AR decoding and outperforms fixed-length SotA baseline while maintaining output quality.

## 1 Introduction

Auto-regressive (AR) models are effective in language modeling but face latency issues due to their sequential token generation, particularly in Large Language Models (LLMs) (Brown et al., 2020; Ouyang et al., 2024; Touvron et al., 2023a,b). Speculative decoding (SD) mitigates this by dividing decoding into two stages: a draft stage and a verification stage (Chen et al., 2023; Leviathan et al., 2023). A smaller draft model predicts multiple tokens, which are verified by the larger model in one forward pass. By iterating these two stages, SD accelerates LLM while maintaining the original model’s output distribution (Chen et al., 2023).

Most existing methods generate draft tokens using *static* structures, such as fixed-length sequences (Chen et al., 2023; Leviathan et al., 2023)

\*The corresponding authors are Lu Chen and Kai Yu.

The diagram illustrates the motivation for finding the optimal draft stopping point. It shows a 'Draft' stage (1) generating tokens  $x_1, x_2, x_3, x_4, x_5, x_6$ . These tokens are compared against a 'Target' (2) to find the optimal stopping point at  $x_5$  (Perfect). Other options like  $x_3$  (Too short) and  $x_7$  (Too Long) are shown as less optimal. The diagram also shows a 'Stop drafting at various positions' option with a red 'END' marker.

Figure 1: Motivation: finding the optimal draft stopping point to reduce target forward passes and minimize wasted computation.

or static trees (Cai et al., 2024; Li et al., 2024b). However, studies have shown that the acceptance length of draft tokens strongly depends on the generated context (Li et al., 2024a). This implies that draft models with static draft structures cannot always exactly generate the optimal number of tokens, thus introducing more costly target model’s forward passes and draft computation waste, as shown in Figure 1. Figure 2 further demonstrates the big difference in acceptance length of different draft iterations during the inference of a sample. Therefore, to optimize decoding efficiency, an ideal draft model should use *context-aware adaptive* draft structures, rather than using static structures. In fact, in our preliminary experiments (§ 2.4), we further show that a draft length oracle (telling the draft model optimal draft length before generation) can improve the decoding throughput of fixed-length EAGLE (Li et al., 2024b) by 29%. Thus, realizing adaptive draft structures is meaningful and rewarding.

Though multiple efforts have explored adaptive draft structures in different ways, as listed in Table 1, they lack a comprehensive analysis and systematic approach. Their limitations can be summarized as: (1) **Unsatisfied performance**. Current methods are mostly based on inherent outputs and<table border="1">
<thead>
<tr>
<th>Method Name</th>
<th>Speculation Framework</th>
<th>Draft Structure Modeling</th>
<th>Manual Thresholds</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Draft &amp; Verify</b> (Zhang et al., 2024)</td>
<td>Self-Speculative</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>SpecDec++</b> (Huang et al., 2024)</td>
<td>Standalone Draft Model</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>DISCO</b> (Mamou et al., 2024)</td>
<td>Standalone Draft Model</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>PEARL</b> (Liu et al., 2024b)</td>
<td>Standalone Draft Model</td>
<td>None</td>
<td>/</td>
</tr>
<tr>
<td><b>EAGLE-2</b> (Li et al., 2024a)</td>
<td>EAGLE (Li et al., 2024b)</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>DDD</b> (Brown et al., 2024)</td>
<td>EAGLE</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>OPT-Tree</b> (Wang et al., 2024)</td>
<td>EAGLE</td>
<td>Implicit</td>
<td>Required</td>
</tr>
<tr>
<td><b>AdaEAGLE (Ours)</b></td>
<td>EAGLE</td>
<td>Explicit</td>
<td>Not Required</td>
</tr>
</tbody>
</table>

Table 1: Comparison of speculative decoding methods with adaptive draft lengths, in terms of base framework, draft boundary modeling, and manual threshold requirements.

Figure 2: Maximum draft lengths accepted at different draft iterations via EAGLE.

threshold control and fail to effectively model the optimal draft length. Experiments also show that there remains a significant optimization gap when compared to the topline performance of EAGLE-Oracle. (2) **Hard to optimize.** The *implicit* approach to modeling adaptive draft structures relies on the LM head output of the draft model, which is naturally trained on the original token sequences. This output cannot be further optimized for the specific goal of adaptive draft generation. Also, predefined fixed thresholds cannot fit all datasets and draft starting points, challenging the generalization capability. (3) **Obsolete applicability.** Earlier methods were not suitable for the current SotA framework, EAGLE (Li et al., 2024b; Zhou et al., 2024). For example, SpecDec++ (Huang et al., 2024) assumes draft generation follows a Markov Decision Process (MDP), which EAGLE does not conform. PEARL (Liu et al., 2024b) is only applicable when the draft model’s inference cost per step is in the same order as the target model’s. (4) **Complex.** Methods requiring training involve intricate data construction and complex training processes (Huang et al., 2024; Mamou et al., 2024).

In this paper, we propose a novel method, AdaEAGLE, the first SD framework realizing explicit modeling of adaptive draft structures via a Lightweight Draft Length Prophet module (LDLP). The advantages of AdaEAGLE include: (1) **Ex-**

**plicit modeling** of adaptive draft lengths. We use LDLP with simple inputs to predict the optimal draft length before generating the draft. Benefiting from this, LDLP neither relies on output logits nor needs manual set thresholds. (2) **Built on SotA** SD framework, EAGLE. Inspired by EAGLE and other works (Wu et al., 2024), we facilitate hidden states as well as output tokens to predict the draft length. (3) **Simple and easy.** Simplified data construction and highly parallelized training, making it easily integrated into existing SD frameworks. Our results show that compared to vanilla AR decoding, AdaEAGLE achieves a speedup ratio of 1.61; it outperforms fixed-length EAGLE decoding with a speed improvement of 2% and beats implicit-modeling methods (throughput 65.20 vs 64.43).

In Summary, the main contributions of our work are: (1) Provides a systematic analysis of the huge potential gain of realizing adaptive draft length. (2) Proposes a simple but effective SotA-based method, AdaEAGLE, to demonstrate that explicitly modeling of draft length is feasible and efficient in adaptive draft generation control. (3) Explicit modeling facilitates deeper and more delicate optimization and paves the way for improving adaptive draft structures in more complex scenarios, such as tree decoding and non-greedy decoding.

## 2 Analysis of Adaptive Draft Length

In this section, we provide a systematic analysis of the huge potential gain of integrating adaptive draft length control into SD. We start with introducing SD and its SotA variation, EAGLE (Li et al., 2024b). Then we analyze the optimal draft length under the EAGLE framework. Finally, based on pilot experiments, we discuss the significant benefit of adaptive drafts.<table border="1">
<thead>
<tr>
<th>Models</th>
<th>Draft Len.</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>T_{\text{total}}</math></th>
<th><math>T_{\text{draft}}</math></th>
<th><math>T_{\text{target}}</math></th>
<th><math>N_{\text{draft}}</math></th>
<th><math>N_{\text{target}}</math></th>
<th><math>N_{\text{waste}}</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Vanilla AR</td>
<td>-</td>
<td>1.00</td>
<td>39.36</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">EAGLE (Li et al., 2024b)</td>
<td>2</td>
<td>2.35</td>
<td>58.89</td>
<td>692.93</td>
<td>60.60</td>
<td>576.93</td>
<td>36590</td>
<td>18295</td>
<td>11968</td>
</tr>
<tr>
<td>3</td>
<td>2.71</td>
<td>64.66</td>
<td>631.41</td>
<td>73.21</td>
<td>504.19</td>
<td>47535</td>
<td>15845</td>
<td>20459</td>
</tr>
<tr>
<td>4</td>
<td>2.95</td>
<td>63.84</td>
<td>638.77</td>
<td>85.32</td>
<td>499.93</td>
<td>58192</td>
<td>14548</td>
<td>29806</td>
</tr>
<tr>
<td>5</td>
<td>3.11</td>
<td>64.25</td>
<td>636.05</td>
<td>98.14</td>
<td>484.80</td>
<td>69035</td>
<td>13807</td>
<td>39896</td>
</tr>
<tr>
<td>6</td>
<td>3.22</td>
<td>61.98</td>
<td>656.57</td>
<td>110.89</td>
<td>492.21</td>
<td>80112</td>
<td>13352</td>
<td>50503</td>
</tr>
<tr>
<td>EAGLE-Oracle</td>
<td>Dyn.</td>
<td>3.41</td>
<td>83.46</td>
<td>483.64</td>
<td>48.74</td>
<td>383.66</td>
<td>33313</td>
<td>13952</td>
<td>0</td>
</tr>
<tr>
<td>AdaEAGLE</td>
<td>Dyn.</td>
<td>3.03</td>
<td>66.35</td>
<td>615.57</td>
<td>85.94</td>
<td>476.86</td>
<td>55649</td>
<td>14170</td>
<td>26871</td>
</tr>
</tbody>
</table>

Table 2: Performance comparison between the EAGLE model with various fixed draft lengths (ranging from 2 to 6) and adaptive-length draft models.  $\tau$  and Tok/s are the acceptance length and throughput averaged over samples, respectively.  $T_{\text{total}}$  represents the overall inference time, while  $T_{\text{draft}}$  and  $T_{\text{target}}$  represent the time spent on the draft model and target model, respectively.  $N_{\text{draft}}$  and  $N_{\text{target}}$  denote the number of tokens generated during the decoding process, and  $N_{\text{waste}}$  indicates the number of draft tokens rejected by the target model during verification.

## 2.1 Speculative Decoding

Let  $T_{1:j}$  denote the sequence of tokens generated so far  $t_1, t_2, \dots, t_j$  by an AR model (typically a LLM), and  $p(t_j)$  be the probability assigned to token  $t_j$  by the target model. Speculative decoding introduces a smaller draft model that guesses a sequence of candidate tokens  $\hat{T}_{j+1:j+k}$  along with their probabilities  $\hat{p}(\hat{t}_{j+i})$  for  $i \in [1, k]$ . The target model then computes the actual probabilities  $p(\hat{t}_{j+i})$  for these tokens in parallel and decides the acceptance of the tokens in order. The acceptance of each draft token  $\hat{t}_{j+i}$  is determined by:  $p_{\text{accept}}(\hat{t}_{j+i}) = \min\left(1, \frac{p(\hat{t}_{j+i})}{\hat{p}(\hat{t}_{j+i})}\right)$ . If token  $\hat{t}_{j+i}$  is rejected, all successor tokens  $\hat{T}_{j+i+1:j+k}$  are discarded, and a new token is resampled from the distribution:  $\text{norm}(\max(0, p(\hat{t}_{j+i}) - \hat{p}(\hat{t}_{j+i})))$ . This process allows SD to validate multiple tokens in one pass, significantly reducing the number of sequential steps required in AR decoding. This method ensures that the output distribution of the SD is consistent with vanilla AR decoding of the target LLM (Chen et al., 2023).

## 2.2 EAGLE

To improve draft quality, instead of training a separate small model, EAGLE (Extrapolation Algorithm for Greater Language Model Efficiency) reuses the target model’s input embedding and LM head, adding a trainable draft head in between (Li et al., 2024b). EAGLE’s first version used a static tree for draft validation in one forward pass, while the second version dynamically selected candidates based on confidence values (Li et al., 2024a). DDD (Brown et al., 2024) refined this method but still relied on confidence-based acceptance with manually set thresholds, as discussed in § 1.

## 2.3 Draft-Length Oracle for EAGLE

To facilitate the discussion, unless otherwise stated, the following paper is based on sequential draft generation and greedy decoding (sample temperature is set as 0). We start by illustrating a new EAGLE-Oracle model with ideal adaptability via a draft-length oracle. The EAGLE-Oracle model uses the same architecture and parameters as the original EAGLE model, but during inference, a draft-length oracle module can tell the optimal draft length. In the context of greedy decoding, the optimal draft length is defined as the length that minimizes the number of forward passes required by the target model. In the context of greedy decoding and EAGLE, the optimal draft length is exactly the acceptance length (need an additional assumption, see the proof in Appendix A). By telling the draft model the optimal length before generating, EAGLE-Oracle ensures that the draft model outputs the necessary number of tokens, minimizing waste and maximizing efficiency.

## 2.4 Benefit of Adaptive Draft Lengths

By using the draft length oracle, EAGLE-Oracle achieves the theoretically optimal speedup under the given model parameters. We evaluated the acceleration effects of both fixed-length EAGLEs and EAGLE-Oracle. The experiment settings follow § 4.1, with results shown in Table 2. The first five rows represent EAGLE with exactly 2 to 6 fixed-length draft token sequences, respectively, while the sixth row corresponds to EAGLE-Oracle.

For EAGLE-Oracle,  $N_{\text{waste}} = 0$  since it generates exactly the optimal number of tokens at every draft starting position. As a result, the average acceptance length and the throughput of EAGLE-Oracle are much higher than those of EAGLE mod-Figure 3 consists of three parts: (a) AdaEAGLE Architecture, (b) LDLP Architecture, and (c) LDLP Training Data Collection.

- **(a) AdaEAGLE Architecture:** Shows a Target model and a Draft model. The Target model takes input tokens  $x_1, x_2, x_3, x_4$  and produces a prediction  $2.8 \approx 3$ . The Draft model takes input tokens  $x_1, x_2, x_3, x_4, x_5, x_6$  and produces a prediction  $3$ . The LDLP module takes the Target's prediction and the Draft's prediction as input and outputs a draft length prediction. The process involves generating more tokens (Draft 3 more tokens) and comparing the Target and Draft models.
- **(b) LDLP Architecture:** A detailed view of the LDLP module. It consists of an Embedding layer, Transformer layers, and a Final norm layer. The input tokens  $\hat{x}_1, \hat{x}_2, \hat{x}_3, \hat{x}_4$  are embedded and passed through the layers to produce the output  $e_3, f_3$ .
- **(c) LDLP Training Data Collection:** Illustrates the process of collecting paired training data. It shows the Target model generating data by the target model (Step 1), the Draft model generating data by the draft model (Step 2), and obtaining paired data by comparison (Step 3).

Figure 3: Illustration of AdaEAGLE framework. (a) The architecture of AdaEAGLE. The draft length are predicted by an LDLP module. (b) A closer look of LDLP. (c) An example of collecting paired training data of LDLP.

els with fixed-length drafts (improved by up to 42%), demonstrating the huge potential reward of pursuing perfect adaptive draft lengths.

### 3 AdaEAGLE

In this section, we start by introducing the overall architecture of AdaEAGLE, the first SD framework that explicitly models the adaptive draft structure (§ 3.1). To achieve this, we carefully design a Lightweight Draft Length Predictor (LDLP) which functions like a prophet by directly estimating the optimal draft length in advance during inference. More details of the LDLP are presented in § 3.2. Finally, we demonstrate how to train the LDLP (§ 3.3).

#### 3.1 SD with Adaptive Draft Length

Denote the target model as  $\mathcal{M}_T$  and the draft model as  $\mathcal{M}_D$ . The draft model is responsible for autoregressively generating drafts, while the target model verifies the correctness of all tokens in the draft in a single forward pass. The validated prefixes<sup>1</sup> from the generated drafts are fed back into the draft model to continue decoding until the entire decoding process is complete.

Formally, for the interaction between  $\mathcal{M}_T$  and  $\mathcal{M}_D$  in the  $r$ -th iteration, let  $T^r = (t_1, t_2, \dots, t_j, \hat{t}_{j+1}, \dots, \hat{t}_{j+k})$  denote the input sequence of  $\mathcal{M}_T$ , where  $T_{1:j}^r$  is the generated formal sequence till last iteration,  $T_{j+1:j+k}^r$  is the draft at current iteration, and  $k \in \mathbb{N}^+$  is the draft length.

<sup>1</sup>A validated prefix is defined as the longest prefix in a draft that contains no incorrect tokens.

The target model  $\mathcal{M}_T$  verifies the draft via

$$\begin{aligned} k_r^\circ, t_{j+k_r^\circ+1} &= \mathcal{M}_T(T^r, k), \\ \bar{T} &= T_{1:j+k_r^\circ}^r + (t_{j+k_r^\circ+1}, \dots) \end{aligned} \quad (1)$$

where  $k_r^\circ$  is the number of correct tokens in the draft and  $t_{j+k_r^\circ+1}$  is an extra predicted bonus token during the verification (Figure 3-(a)-(1)-(2)). Then,  $\mathcal{M}_D$  generates the draft for the  $(r+1)$ -th iteration by

$$\begin{aligned} (\hat{t}_{j+k_r^\circ+2}, \dots, \hat{t}_{j+k_r^\circ+1+k}) &= \mathcal{M}_D(\bar{T}, k), \\ T^{r+1} &= \bar{T} + (\hat{t}_{j+k_r^\circ+2}, \dots, \hat{t}_{j+k_r^\circ+1+k}), \end{aligned} \quad (2)$$

where  $+$  represents the concatenation operation (Figure 3-(a)-(4)).

As we see in Equation 2, in standard speculative decoding, the length of the draft generated by  $\mathcal{M}_D$  is a hyperparameter  $k$  that is fixed in advance. According to the aforementioned analysis, during each iteration of interaction between  $\mathcal{M}_T$  and  $\mathcal{M}_D$ , a significant gap between  $k$  and the number of correct tokens in the draft  $k_r^\circ$  can slow down the decoding speed of the model. Motivated by this, we design a Lightweight Draft Length Predictor (LDLP) which functions like a prophet by estimating the optimal draft length in advance (Figure 3-(a)-(3)), i.e., predicting a value of  $k_r \in \mathbb{N}^+$  closing to the optimal draft length  $k_r^\circ$ . Mathematically, the  $k$  in Equation 1 and 2 is replaced into  $k_r$ . We illustrate the details in Algorithm 1.

#### 3.2 Lightweight Draft Length Predictor

In designing LDLP, we follow the “KISS” (Keep It Simple and Stupid) principle. our goal is to ensure that its output closely approximates the numberof correct tokens in the draft while keeping it as lightweight as possible. A heavy LDLP would increase computational overhead and data transfer between CPU and GPU, thereby negating the intended speedup of the speculative decoding.

Based on this design principle, we implement a three-layer Multi-Layer Perceptron (MLP) as the model structure for our LDLP. Prior studies on LLM future token prediction (Pal et al., 2023; Hernandez et al., 2024; Wu et al., 2024) point out that the hidden state of the last generated token contains the “plan” of generating subsequent future tokens. Inspired by this, for the  $r$ -th iteration, LDLP simply takes as input the embedding  $e_{j+k_r^\circ}$  of the last token  $\hat{t}_{j+k_r^\circ}$  in the validated prefix of the draft, along with its last hidden state  $f_{j+k_r^\circ}$  after the final layer normalization in  $\mathcal{M}_T$  (see Figure 3-(b)). Meanwhile, LDLP outputs a scalar in  $\mathbb{R}$ . Mathematically,

$$\begin{aligned} \bar{k}_{r+1} &= \text{Round}(\text{MLP}(e_{j+k_r^\circ}, f_{j+k_r^\circ})), \\ k_{r+1} &= \min\{k_{\max}, \max\{0, \bar{k}_{r+1}\}\}, \end{aligned} \quad (3)$$

where the  $k_{\max}$  is a very slack hyperparameter that controls the upper bound of possible draft length.

### 3.3 Training of LDLP

The training of LDLP involves two main problems: training data collection and optimization criteria design.

**Training data collection** As discussed in § 3.2, the training data for LDLP consists of pairs of  $([e_j; f_j], k^j)$ , where  $e_j$  and  $f_j$  represent the embedding and last hidden state after final layer normalization in the target model for the  $j$ -th token, respectively. Here,  $k^j$  denotes the optimal draft length starting from the  $j$ -th token. For a given prompt  $T_{\text{in}}$  with  $m$  tokens, we collect such pairs in the following three steps:

1. 1) generate the output sequence  $T^{\text{out}}$  ( $n$  tokens) by the target model  $\mathcal{M}_T$  based on  $T_{\text{in}}$  (Figure 3-(c)-①)
2. 2) continue drafting  $k_{\max}$  tokens (denoted as  $\hat{T}^{\text{out}}$ ) by the draft model  $\mathcal{M}_D$  based on  $T_{\text{in}} + T_{:j}^{\text{out}}$  (for any  $1 \leq j \leq n$ , see Figure 3-(c)-②)<sup>2</sup>
3. 3) compare  $\hat{T}^{\text{out}}$  to  $T^{\text{out}}$  and calculate the length  $k^j$  of the longest common prefix between them (Figure 3-(c)-③)

<sup>2</sup>The  $k_{\max}$  can be different from the value in Equation 3.

---

### Algorithm 1 Adaptive-Length Draft Decoding with LDLP

---

**Require:** Prompt  $T_{\text{in}}$

**Ensure:** Response  $T_{\text{out}}$

1. 1:  $T \leftarrow T_{\text{in}}$  ▷  $T^r$  in Equation 1
2. 2:  $k \leftarrow 0$  ▷ the draft length  $k_r$
3. 3: **while** not generating a terminator **do**
4. 4:   Calculate  $k_r^\circ$  and  $\bar{T}$  ▷ Equation 1
5. 5:   Predict the draft length  $k_{r+1}$  ▷ Equation 3
6. 6:   Calculate  $T^{r+1}$  ▷ Equation 2
7. 7:    $k \leftarrow k_{r+1}$ ;  $T \leftarrow T^{r+1}$
8. 8: **end while**
9. 9:  $T_{\text{out}} \leftarrow \text{trim the prefix } T_{\text{in}} \text{ of } T$

---

**Optimization criteria design** Let  $k^\circ$  and  $\bar{k}$  denote the ground truth and the prediction of LDLP, respectively. On the one hand, we desire  $\bar{k}$  and  $k^\circ$  to be as close as possible and we adopt  $L_1$  loss to achieve this. On the other hand, we observe that the required time for a single forward pass of the target model is approximately 20 times that of the draft model. Therefore, we hope that  $\bar{k}$  is not less than  $k^\circ$  as much as possible. To achieve this, we introduce a penalty coefficient  $\lambda > 1$  to scale up the  $L_1$  loss when  $\bar{k} < k^\circ$ . Formally, the criteria is defined as

$$\mathcal{L} = \begin{cases} \lambda \cdot |\bar{k} - k^\circ| & \bar{k} < k^\circ \\ |\bar{k} - k^\circ| & \text{otherwise} \end{cases} \quad (4)$$

### 3.4 Incorporate AdaEAGLE with Other Adaptive Draft Techniques

Our explicit draft length modeling via LDLP can be easily combined with other approaches that utilize inherent model outputs (e.g., logits, entropy) for draft boundary modeling. By evaluating the per-step condition, the draft model can terminate generation based on both the predicted length and inherent threshold criteria. In experiments, we discuss the combination of AdaEAGLE and DDD (Brown et al., 2024), referred to as AdaEAGLE-DDD.

## 4 Experiments

### 4.1 Experimental Setups

**Model** We conduct the experiments with Vicuna-7B-v1.3 (Chiang et al., 2023) and the paired EAGLE (Li et al., 2024b) draft model. Our LDLP model is a three-layer MLP with residual connections, and the output hidden states are mapped to the length scalar through a linear transform.<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Draft Len.</th>
<th colspan="2">MT-Bench</th>
<th colspan="2">Alpaca</th>
<th colspan="2">HumanEval</th>
<th colspan="2">GSM8K</th>
<th colspan="2">CNN/DM</th>
<th colspan="2">Natural Ques.</th>
<th colspan="2">Avg.</th>
</tr>
<tr>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
<th><math>\tau</math></th>
<th>Tok/s</th>
</tr>
</thead>
<tbody>
<tr>
<td>Vanilla AR</td>
<td>-</td>
<td>1.00</td>
<td>39.36</td>
<td>1.00</td>
<td>43.42</td>
<td>1.00</td>
<td>41.44</td>
<td>1.00</td>
<td>42.62</td>
<td>1.00</td>
<td>32.65</td>
<td>1.00</td>
<td>43.85</td>
<td>1.00</td>
<td>50.56</td>
</tr>
<tr>
<td rowspan="5">EAGLE (Li et al., 2024b)</td>
<td>2</td>
<td>2.35</td>
<td>58.89</td>
<td>2.27</td>
<td>59.33</td>
<td>2.48</td>
<td>67.01</td>
<td>2.42</td>
<td>63.16</td>
<td>2.21</td>
<td>47.41</td>
<td>2.07</td>
<td>54.34</td>
<td>2.30</td>
<td>58.36</td>
</tr>
<tr>
<td>3</td>
<td>2.71</td>
<td>64.66</td>
<td>2.61</td>
<td>65.01</td>
<td>2.92</td>
<td>75.50</td>
<td>2.80</td>
<td>69.36</td>
<td>2.47</td>
<td>50.32</td>
<td>2.29</td>
<td><b>57.44</b></td>
<td>2.63</td>
<td>63.72</td>
</tr>
<tr>
<td>4</td>
<td>2.95</td>
<td>63.84</td>
<td>2.82</td>
<td>62.94</td>
<td>3.24</td>
<td>76.11</td>
<td>3.08</td>
<td>69.25</td>
<td>2.61</td>
<td>48.89</td>
<td>2.43</td>
<td>55.25</td>
<td>2.86</td>
<td>62.71</td>
</tr>
<tr>
<td>5</td>
<td>3.11</td>
<td>64.25</td>
<td>2.95</td>
<td>62.88</td>
<td>3.44</td>
<td>77.95</td>
<td>3.27</td>
<td>69.83</td>
<td>2.69</td>
<td>48.36</td>
<td>2.51</td>
<td>54.00</td>
<td>3.00</td>
<td>62.88</td>
</tr>
<tr>
<td>6</td>
<td>3.22</td>
<td>61.98</td>
<td>3.04</td>
<td>60.18</td>
<td>3.57</td>
<td>75.05</td>
<td>3.37</td>
<td>66.77</td>
<td>2.73</td>
<td>46.16</td>
<td>2.56</td>
<td>51.45</td>
<td>3.08</td>
<td>60.27</td>
</tr>
<tr>
<td>DDD (Brown et al., 2024)</td>
<td>Dyn.</td>
<td>3.21</td>
<td>66.33</td>
<td>3.02</td>
<td>65.41</td>
<td>3.59</td>
<td>77.71</td>
<td>3.41</td>
<td>69.39</td>
<td>2.65</td>
<td><b>50.53</b></td>
<td>2.49</td>
<td>57.18</td>
<td>3.06</td>
<td>64.43</td>
</tr>
<tr>
<td>AdaEAGLE</td>
<td>Dyn.</td>
<td>3.03</td>
<td>66.35</td>
<td>2.86</td>
<td>64.67</td>
<td>3.41</td>
<td>77.55</td>
<td>3.23</td>
<td>70.68</td>
<td>2.60</td>
<td>49.99</td>
<td>2.43</td>
<td>57.41</td>
<td>2.93</td>
<td>64.44</td>
</tr>
<tr>
<td>AdaEAGLE-DDD</td>
<td>Dyn.</td>
<td>3.12</td>
<td><b>66.86</b></td>
<td>2.93</td>
<td><b>66.16</b></td>
<td>3.53</td>
<td><b>79.07</b></td>
<td>3.34</td>
<td><b>71.20</b></td>
<td>2.63</td>
<td>50.52</td>
<td>2.45</td>
<td>57.36</td>
<td>3.00</td>
<td><b>65.20</b></td>
</tr>
</tbody>
</table>

Table 3: Average accepted tokens  $\tau$  and throughput Tok/s of different methods on six benchmarks. We include standard EAGLE speculative decoding method with fixed draft length ranging from 2 to 6 tokens.

**Training** For a fair comparison, we use the ShareGPT dataset with 68000 dialog samples to train our length prediction model. We first ask Vicuna to auto-regressively generate responses and give the prompts for each dialog sample. Then we follow the data collection process detailed in § 3.3. For the training of LDLP, we keep the base target model and corresponding Eagle draft model frozen. We set the start learning rate to  $5e-5$  and use a cosine learning rate scheduler. LDLP is trained with batch size 128 for 5 epochs trained within 30 minutes on 8 A800 80GB GPUs, making AdaEAGLE training-efficient.

**Benchmarks** Following the evaluation setup of EAGLE (Li et al., 2024a), we select six representative text generation benchmarks across diverse domains to evaluate our method. These benchmarks include the multi-turn conversation task MT-Bench (Zheng et al., 2024), the instruction-following task Alpaca (Taori et al., 2024), the code generation task HumanEval (Chen et al., 2021), the mathematical reasoning task GSM8K (Cobbe et al., 2021), the summarization task CNN/Daily Mail (Nallapati et al., 2016), and the question-answering task Natural Questions (Kwiatkowski et al., 2019). All experiments are conducted on a single A800-80G GPU with a batch size of 1, using greedy decoding (temperature set to 0.0) and FP32 precision.

**Metrics** Since we do not modify the target model or the draft model, the generation quality remains consistent with EAGLE and prior speculative decoding methods that achieve lossless acceleration. Therefore, we do not report quality metrics for the benchmarks. Instead, we use the following two metrics to evaluate acceleration performance: (1) average acceptance length ( $\tau$ ), defined as the number of accepted tokens per decoding step, and (2)

throughput (Tok/s), defined as the number of tokens generated per second.

## 4.2 Main Results

**Baselines** We select the original EAGLE method with a fixed draft length as our baseline. During each decoding step, EAGLE autoregressively generates a fixed number of draft tokens (ranging from 2 to 6). We also compare our method with the recently proposed adaptive draft length approach, DDD (Brown et al., 2024), applied to EAGLE. DDD is a threshold-based method with implicit draft structure modeling that controls the termination of draft generation. Specifically, DDD uses the probability of a draft sequence, which is calculated as the product of the probabilities of all tokens generated up to that point. If the product exceeds a given threshold, the draft model continues generation; otherwise, it terminates. We set the threshold to the value that performed best on MT-Bench and test it on other benchmarks.

**AdaEAGLE-DDD** We also incorporate our draft length prediction method with the threshold-based method of DDD, resulting in AdaEAGLE-DDD. The draft model exit generation iff. the draft step exceeds the predicted length and the probability of the draft sequence is lower than the threshold.

Table 3 compares our method with other baseline methods in terms of average accepted tokens and throughput (Tok/s). On average, AdaEAGLE achieves the highest throughput, outperforming all variations of the fixed draft length with the standard EAGLE decoding method. Fixed draft length methods can only achieve the best performance on specific benchmarks, for example, EAGLE with fixed draft length 3 achieves the best throughput on Natural Questions. This is because the response to this question-answering task is more dynamic and<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">MT-Bench</th>
<th colspan="2">Alpaca</th>
</tr>
<tr>
<th><math>\tau</math></th>
<th>Toks/s</th>
<th><math>\tau</math></th>
<th>Toks/s</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Penalty</td>
<td>2.81</td>
<td>64.62</td>
<td>2.62</td>
<td>63.08</td>
</tr>
<tr>
<td>w/ Cls</td>
<td>2.87</td>
<td>64.27</td>
<td>2.72</td>
<td>63.64</td>
</tr>
<tr>
<td>AdaEAGLE</td>
<td><b>3.03</b></td>
<td><b>66.35</b></td>
<td><b>2.86</b></td>
<td><b>64.67</b></td>
</tr>
</tbody>
</table>

Table 4: Ablation experiment results on MT-Bench and Alpaca about the training loss design. "w/o Penalty" indicate that the model is trained without L1 loss penalty. "w/ Cls" indicate that the model is trained with classification loss.

the draft model cannot effectively match the output of the target model, hence resulting in a short average accepted token. In this case, a short fixed length like three would save waste tokens. However on the code generation benchmark HumanEval, where the output is more predictable due to the structure of code and draft models have more acceptable tokens, a larger fixed draft length like five would reduce the number of target forward times. While AdaEAGLE with adaptive draft length demonstrates its adaptability across different tasks.

Table 2 provides a breakdown analysis of the factors contributing to acceleration on MT-Bench, showing the inference time spent on the draft and target models for each draft setting, along with the number of tokens generated by each. For EAGLE with a fixed draft length, as the draft length increases, the number of tokens generated by the draft model ( $N_{\text{draft}}$ ) grows from 47,525 to 80,112, resulting in fewer target model forward passes ( $N_{\text{target}}$ ). However, this also leads to a surge in the number of wasted tokens ( $N_{\text{waste}}$ ) due to the drastic fluctuation in maximum draft lengths at different positions, as shown in Figure 2. This trend is directly reflected in the time spent on each component: as  $N_{\text{draft}}$  increases,  $T_{\text{draft}}$  also increases, while  $T_{\text{target}}$  decreases due to the reduction in  $N_{\text{target}}$ , presenting a challenging trade-off. Our adaptive draft length prediction method achieves a better balance. Specifically, compared to EAGLE with a fixed draft length of 4, our method results in fewer wasted draft tokens and fewer target model forward passes, reducing both  $T_{\text{draft}}$  and  $T_{\text{target}}$ , and ultimately enhancing overall throughput to 66.35 tok/s.

We also test the recently proposed acceptance rate modeling method DDD, which is based on the model-generated token probability. As shown in Table 3 AdaEAGLE is comparable to DDD in the average acceleration performance, with a throughput of 64.44 tok/s and 64.43 tok/s respectively. DDD

Figure 4: Comparison of draft length distributions with and without loss penalty.

has more average accepted tokens, and this is possible because our model is relatively conservative in predicting the draft length, and DDD is more aggressive, generating more draft tokens and more waste tokens.

Our AdaEAGLE-DDD approach achieves an additional gain in decoding throughput, reaching an average of 65.20 tokens per second. AdaEAGLE-DDD also achieves the highest throughput on four out of six benchmarks. Notably, on the code generation task HumanEval, AdaEAGLE-DDD demonstrates the best acceleration performance. This is because we retain the longer of the two lengths, reducing the probability of premature draft exit and increasing the average acceptance length.

### 4.3 Ablation Study

In this section we explore the effectiveness of different designing choices in our method.

#### 4.3.1 Loss Penalty

We train AdaEAGLE using L1 regression loss to predict the adaptive draft length logits. As discussed in Section 3.3, shorter lengths are given greater tolerance compared to longer predictions. To address this, we apply a penalized L1 loss during training, which penalizes predictions that are shorter than the ground truth label. Table 4 compares the impact of incorporating the penalized loss, showing that it improves both the average accepted length and throughput. Figure 4 further illustrates the draft length distribution with and without the loss penalty, highlighting that the loss penalty encourages longer draft lengths.<table border="1">
<thead>
<tr>
<th>Threshold</th>
<th>-0.2</th>
<th>-0.4</th>
<th>-0.6</th>
<th>-0.8</th>
<th>-1.0</th>
</tr>
</thead>
<tbody>
<tr>
<td>Tok/s</td>
<td>66.13</td>
<td>66.85</td>
<td>66.86</td>
<td>66.86</td>
<td>66.4</td>
</tr>
</tbody>
</table>

Table 5: Throughput results (Tok/s) for different threshold values. The thresholds correspond to the log probabilities, resulting in negative values. The ablation is conducted on the MT-Bench dataset.

### 4.3.2 Regression v.s. Classification

We also explore alternative approaches for draft length prediction, including regression and classification. In the regression approach, the model maps hidden states to a continuous score, whereas in the classification approach, it maps hidden states to a predefined set of classes. As shown in Table 4, regression outperforms classification. This is because regression can naturally model the ordinal relationships between different labels.

### 4.3.3 AdaEAGLE-DDD thresholds

Table 5 presents the performance for various threshold values for the DDD component of AdaEAGLE-DDD, showing that the throughput remains stable and insensitive across different settings. Based on these results, we select a threshold value of -0.6 for the experiments reported in Table 3.

## 5 Related Works

LLM is popular and capable. However, its AR inference is slow and costly. Significant methods for efficient LLM inference have been proposed (Zhou et al., 2024). This section focuses on the technique of speculative decoding (SD) and adaptive draft structure for SD.

**Speculative Decoding** Speculative decoding (SD) uses a draft-verify paradigm to realize lossless acceleration. SD frameworks can be roughly divided by the type of draft models (Xia et al., 2024): (1) Independent draft models. SpecDec (Xia et al., 2023) introduced a non-AR Transformer to generate multiple tokens simultaneously. Alternatively, many works (Leviathan et al., 2023; Chen et al., 2023; Spector and Re, 2023; Sun et al., 2023) propose using smaller pre-trained models from the same LLM series (e.g., Llama2-7B and Llama2-70B (Touvron et al., 2023b)) to accelerate inference, which avoids extra training and preserves alignment in prediction behaviors due to shared tokenizers and training corpora. (2) Self-drafting eliminates the need for external draft models by leveraging the target LLM itself for drafting (Stern et al., 2018; Santilli et al., 2023; Cai et al., 2024; Li et al.,

2024b). Recent techniques employ lightweight FFN heads to enable parallel (Cai et al., 2024) or AR (Li et al., 2024b) token generation, reducing computational overhead. Other works (Yang et al., 2023; Zhang et al., 2024) explore early exiting and layer skipping within the target model to improve drafting efficiency. However, most works use static draft structures, leaving the rewarding problem of adaptive draft generation unsolved.

**Adaptive Draft Structure** An adaptive draft structure means that the draft model can adjust its generation structure (e.g., the length of the draft sequence, or the depth/width/shape of the draft tree) dynamically based on the speculation **context**. Although some studies have recognized the need to make the draft structure dynamic, these methods still lack specialization and systematic approaches, as listed in Table 1. Regarding the modeling approach of draft boundaries, some works use inherent approaches by leveraging the output distribution’s confidence to determine the stopping point, as seen in (Zhang et al., 2024; Li et al., 2024a; Liu et al., 2024a; Brown et al., 2024; Wang et al., 2024), while others (Huang et al., 2024; Mamou et al., 2024) employ extra modules such as additional trainable heads to determine whether to stop generating drafts.

There are also some works orthogonal to the above, such as Sequoia (Chen et al., 2024), which is hardware-aware adaptive rather than context-aware adaptive. In summary, current models supporting adaptive draft structure lack performance, timeliness, and simplicity. The method proposed in this paper is the first SD framework that directly models the draft structure and achieves optimal performance based on EAGLE.

## 6 Conclusion

In this paper, we present a novel speculative decoding framework, AdaEAGLE, which is the first to implement adaptive draft control by explicitly modeling the optimal draft structure. By incorporating the Lightweight Draft Length Prophet (LDLP) module, AdaEAGLE predicts the optimal draft length to guide token generation, minimizing the number of forward passes for the large target model and maximizing overall decoding speed. Experimental results demonstrate that AdaEAGLE achieves decoding throughput closer to the oracle topline than existing SD methods. It provides a lossless speedup of 1.61% over vanilla AR de-coding, surpassing fixed-length SotA of 2%, and shows generalizability and robustness across various test sets. Our approach opens the door for modeling adaptive draft structures in an explicit and dedicated “end-to-end” manner and can be integrated into a broad range of SD frameworks and sampling strategies.

## Limitations

Unlike EAGLE (Li et al., 2024b) and EAGLE-2 (Li et al., 2024a) which use tree-based decoding, the method presented in this paper is currently limited to a sequential draft structure, so we only provide results under a greedy (top-1) decoding strategy. While a sequential structure allows us to clearly demonstrate our core idea with a single objective (length), tree-based structures require more complex and fine-grained control, such as the width and depth of each sub-tree, which we leave for future work.

Additionally, further exploration could be conducted on how LDLP can improve accuracy. Although this paper presents improvements based on EAGLE, the concept of directly modeling adaptive draft structures can, in fact, be applied to many other frameworks, such as (Zhang et al., 2024). Therefore, future research is needed to optimize LDLP, evaluate different SD frameworks, and examine the interactions between these elements.

## References

Oscar Brown, Zhengjie Wang, Andrea Do, Nikhil Mathew, and Cheng Yu. 2024. [Dynamic depth decoding: Faster speculative decoding for llms](#). *Preprint*, arXiv:2409.00142.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In *Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS '20*, Red Hook, NY, USA. Curran Associates Inc.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. [Medusa: Simple LLM inference acceleration framework with multiple decoding heads](#). In *Proceedings*

*of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 5209–5235. PMLR.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. [Accelerating large language model decoding with speculative sampling](#). *Preprint*, arXiv:2302.01318.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. [Evaluating large language models trained on code](#). *Preprint*, arXiv:2107.03374.

Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. 2024. [Sequoia: Scalable, robust, and hardware-aware speculative decoding](#). *Preprint*, arXiv:2402.12374.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. [Vicuna: An open-source chatbot impressing gpt-4 with 90%\\* chatgpt quality](#).

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training verifiers to solve math word problems](#). *Preprint*, arXiv:2110.14168.

Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, and David Bau. 2024. [Linearity of relation decoding in transformer language models](#). In *The Twelfth International Conference on Learning Representations*.

Kaixuan Huang, Xudong Guo, and Mengdi Wang. 2024. [Specdec++: Boosting speculative decoding via adaptive candidate lengths](#). *Preprint*, arXiv:2405.19715.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. [Natural questions: A benchmark for question answering research](#). *Transactions of the Association for Computational Linguistics*, 7:452–466.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. [Fast inference from transformers via speculative decoding](#). In *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 19274–19286. PMLR.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. [Eagle-2: Faster inference of language models with dynamic draft trees](#). *Preprint*, arXiv:2406.16858.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. [EAGLE: Speculative sampling requires rethinking feature uncertainty](#). In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 28935–28948. PMLR.

Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Kai Han, and Yunhe Wang. 2024a. [Kangaroo: Lossless self-speculative decoding via double early exiting](#). *Preprint*, arXiv:2404.18911.

Tianyu Liu, Yun Li, Qitan Lv, Kai Liu, Jianchen Zhu, and Winston Hu. 2024b. [Parallel speculative decoding with adaptive draft length](#). *Preprint*, arXiv:2408.11850.

Jonathan Mamou, Oren Pereg, Daniel Korat, Moshe Berchansky, Nadav Timor, Moshe Wasserblat, and Roy Schwartz. 2024. [Dynamic speculation lookahead accelerates speculative decoding of large language models](#). *Preprint*, arXiv:2405.04304.

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Çağlar Gülçehre, and Bing Xiang. 2016. [Abstractive text summarization using sequence-to-sequence RNNs and beyond](#). In *Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning*, pages 280–290, Berlin, Germany. Association for Computational Linguistics.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2024. Training language models to follow instructions with human feedback. In *Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS '22*, Red Hook, NY, USA. Curran Associates Inc.

Koyena Pal, Jiuding Sun, Andrew Yuan, Byron Wallace, and David Bau. 2023. [Future lens: Anticipating subsequent tokens from a single hidden state](#). In *Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL)*, pages 548–560, Singapore. Association for Computational Linguistics.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodola. 2023. [Accelerating transformer inference for translation via parallel decoding](#). In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12336–12355, Toronto, Canada. Association for Computational Linguistics.

Benjamin Spector and Chris Re. 2023. [Accelerating llm inference with staged speculative decoding](#). *Preprint*, arXiv:2308.04623.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. [Blockwise parallel decoding for deep autoregressive models](#). In *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 10107–10116.

Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix X. Yu. 2023. [Spectr: Fast speculative decoding via optimal transport](#). In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2024. [Stanford CRFM](#).

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. [Llama: Open and efficient foundation language models](#). *Preprint*, arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcín Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan,Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. [Llama 2: Open foundation and fine-tuned chat models](#). *Preprint*, arXiv:2307.09288.

Jikai Wang, Yi Su, Juntao Li, Qingrong Xia, Zi Ye, Xinyu Duan, Zhefeng Wang, and Min Zhang. 2024. [Opt-tree: Speculative decoding with adaptive draft tree structure](#). *Preprint*, arXiv:2406.17276.

Wilson Wu, John X. Morris, and Lionel Levine. 2024. [Do language models plan ahead for future tokens?](#) *Preprint*, arXiv:2404.00859.

Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. 2023. [Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation](#). In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 3909–3925, Singapore. Association for Computational Linguistics.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. 2024. [Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding](#). In *Findings of the Association for Computational Linguistics ACL 2024*, pages 7655–7671, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papiliopoulos, and Kangwook Lee. 2023. [Predictive pipelined decoding: A compute-latency trade-off for exact LLM decoding](#). In *Workshop on Efficient Systems for Foundation Models @ ICML2023*.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2024. [Draft& verify: Lossless large language model acceleration via self-speculative decoding](#). In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 11263–11282, Bangkok, Thailand. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2024. [Judging llm-as-a-judge with mt-bench and chatbot arena](#). In *Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS '23*, Red Hook, NY, USA. Curran Associates Inc.

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. 2024. [A survey on efficient inference for large language models](#). *Preprint*, arXiv:2404.14294.

## A Oracle-Guided Analysis of Optimal Draft Length

Assume that when the input is token sequence  $T_{a:b}$ , output hidden states from the target model’s second-to-top layer and the draft model’s AR head are  $F(T_{a:b}) = \{f(t_a), f(t_{j+1}) \dots, f(t_b)\}$  and  $\hat{F}(T_{a:b}) = \{\hat{f}(t_a), \hat{f}(t_{j+1}) \dots, \hat{f}(t_b)\}$ , respectively. Let the generated token sequence so far be denoted as  $T_{1:j}$  and the draft token sequence based on  $T_{1:j}$  be  $\hat{T}_{j+1:j+k|j}$ . Under the greedy decoding strategy, given the sequence history  $T_{1:j}$ , no matter what draft model outputs, the target model’s output results are deterministic. We refer to the token sequence generated by the original target model (without drafts) as the *formal* token sequence.

In SpecDec++ (Huang et al., 2024), the optimal draft length  $k_j^* = \text{optK}(j)$  is defined as the length such that minimizes the number of forward passes required by the target model. It can be inferred that  $\hat{t}_{j+k_j^*|j}$  is accepted by the target model, while  $\hat{t}_{j+k_j^*+1|j}$  is rejected. This occurs because in the SpecDec++ framework, generating a new draft token  $\hat{t}_{j+i+1|j}$  depends only on the token history  $T_{1:j}, \hat{T}_{j+1:j+i|j}$ . Generating more than  $k_j^*$  candidates results in unnecessary work by the draft model, while generating fewer than  $k_j^*$  is sub-optimal as the rejected token  $\hat{t}_{j+k_j^*+1|j}$  would still be generated in the next draft iteration, necessitating an additional forward pass from the target model.

However, in the EAGLE framework, draft generation depends not only on the token history but also on the hidden states produced by both the target and draft models, namely  $F(T_{1:j})$  and  $\hat{F}(\hat{T}_{j+1:j+i})$ . If fewer than  $k_j^*$ , say  $k'$  draft candidates are generated, the hidden history changes in the next draft iteration, meaning that a different draft token  $\hat{t}_{j+k_j^*+1|j+k'+1}$  instead of the rejected  $\hat{t}_{j+k_j^*+1|j}$  may be produced at the same position. That token, may equal  $T_{j+k_j^*+1}$ , and then be accepted by the target model, thus providing a chance of reducing the target model’s forward passes. This implies that, even if we knew the next draft token could be accepted, generating it greedily is no longer guaranteed optimal.

To address this counter-intuitive situation, this paper introduces a reasonable assumption to resolve the dilemma.

**Assumption 1.** *Let  $j$  and  $j'$  ( $j' < j$ ) be the length*of the generated formal token history, and currently the draft model is generating a draft token at position  $m$  ( $m > j$ ). Under EAGLE framework, we state that

$$p_{\text{accept}}(\hat{t}_{m|j}) \geq p_{\text{accept}}(\hat{t}_{m|j'}). \quad (5)$$

To summarize, the more hidden states from the target, the more accurate guesses from the draft. This is reasonable because the draft hidden states and tokens are estimations of the target. Having a longer history from the target model allows the draft to generate output distributions that more closely align with those of the target model. Through this assumption, under EAGLE's framework, we obtain the following lemma:

**Lemma 1.** *Let the generated formal tokens be  $T_{1:j}$ . Under EAGLE framework, the optimal draft length is still  $k_j^* = \text{optK}(j)$  such that  $\hat{t}_{j+k_j^*|j}$  is accepted by the target model, while  $\hat{t}_{j+k_j^*+1|j}$  is rejected.*

**Prove** It is obvious that a draft length larger than  $k_j^*$  is still not optimal. Let  $j_A = j + k_j^* + 1$ ,  $j_{A\text{End}} = j_A + \text{optK}(j_A)$ . In the next draft iteration, the draft model starts with  $\hat{t}_{j_A+1|j_A}$ , and ends with  $\hat{t}_{j_{A\text{End}}|j_A}$ , i.e.,  $p_{\text{accept}}(\hat{t}_{j_{A\text{End}}+1|j_A}) = 0$ . Denote this draft scheme as scheme A.

When less than  $k_j^*$  candidates, say  $k'$  tokens are generated, let  $j_B = j + k' + 1$ . Denote this draft scheme as scheme B. Since  $j_B < j_A$ , according to Assumption 1,

$$p_{\text{accept}}(\hat{t}_{j_{A\text{End}}+1|j_B}) \leq p_{\text{accept}}(\hat{t}_{j_{A\text{End}}+1|j_A}) = 0,$$

which means in the next draft iteration, the decoding progress for scheme B cannot exceed that of scheme A. Similarly, after the same number of draft iterations, scheme B is still left behind by scheme A. Equivalently, when generating the same number of formal tokens, scheme A needs fewer target model's forward passes than scheme B.  $\square$

Lemma 1 can also be easily extended to non-greedy decoding in terms of expectation.

