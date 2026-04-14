Title: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

URL Source: https://arxiv.org/html/2401.15077

Markdown Content:
###### Abstract

Autoregressive decoding makes the inference of Large Language Models (LLMs) time-consuming. In this paper, we reconsider speculative sampling and derive two key observations. Firstly, autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level. Secondly, the inherent uncertainty in feature (second-to-top-layer) level autoregression constrains its performance. Based on these insights, we introduce EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), a simple yet highly efficient speculative sampling framework. By incorporating a token sequence advanced by one time step, EAGLE effectively resolves the uncertainty, enabling precise second-to-top-layer feature prediction with minimal overhead. We conducted comprehensive evaluations of EAGLE, including all models from the Vicuna and LLaMA2-Chat series, the MoE model Mixtral 8x7B Instruct, and tasks in dialogue, code generation, mathematical reasoning, and instruction following. For LLaMA2-Chat 70B, EAGLE achieved a latency speedup ratio of 2.7x-3.5x, doubled throughput, while maintaining the distribution of the generated text.

Machine Learning, ICML

Yuhui Li♠Fangyun Wei‡Chao Zhang♠Hongyang Zhang♣†

♠Peking University ‡Microsoft Research ♣University of Waterloo †Vector Institute 

hongyang.zhang@uwaterloo.ca

[https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

![Image 1: [Uncaptioned image]](https://arxiv.org/html/2401.15077v3/x1.png)

Figure 1: Speedup ratio of Vicuna and LLaMA2-Chat inference latency on the MT-bench for greedy (temperature=0) settings. Speedup ratio of Medusa and Lookahead are copied from their original technical reports. With speculative sampling, there is a lack of suitable draft models to accelerate the 7B model. Employing a 7B model as the draft model for a 13B model results in slow speeds due to the high overhead of the 7B model, rendering it less efficient than vanilla autoregressive decoding. These scenarios are marked as N/A. _In this paper, we only compare with speculative sampling based methods that do not need to finetune the backbone models, ensuring the output text distribution remains constant._

1 Introduction
--------------

Autoregressive decoding, the de facto standard for large language models (LLMs), generates tokens sequentially, leading to slow and costly generation. Speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17); Chen et al., [2023a](https://arxiv.org/html/2401.15077v3#bib.bib2)) based methods address this by dividing the process into a low-cost draft stage and a _parallelized_ verification stage over the drafted tokens, allowing for multiple tokens to be validated in a single LLM pass. These approaches accelerate generation by producing multiple tokens per pass. More importantly, the verification stage ensures that the text distribution aligns precisely with the decoding results of the original LLM, maintaining the integrity of the generated content.

Applying speculative sampling hinges on finding a draft model that mirrors the original LLM’s functionality but with reduced latency, often involving a lower-parameter version from the same LLM series. For instance, in the LLaMA2 (Touvron et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib30)) series which includes models with 7B, 13B, and 70B parameters, using the 7B model as a draft model of the 70B model is valid, while finding a suitable draft model for the smallest 7B variant is tricky. An alternative could be to use TinyLLaMA (Zhang et al., [2024](https://arxiv.org/html/2401.15077v3#bib.bib39)), but it is not feasible for instruct-tuned models due to the inconsistency in instruction templates between LLaMA2-Chat and TinyLLaMA-Chat. Despite the 7B model’s potential as a draft model, its high overhead diminishes acceleration gains. Training a new, appropriately sized draft model specifically for speculative sampling is not an ideal solution either due to the high cost: TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens.

The key to enhancing acceleration in speculative sampling lies in reducing the time overhead and improving the acceptance rate of the draft by the original LLM(Chen et al., [2023b](https://arxiv.org/html/2401.15077v3#bib.bib4); Xia et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib33); Santilli et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib23)). Numerous approaches focus on reducing the overhead of the drafting phase. Lookahead (Fu et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib6)) employs n-gram and Jacobi iteration, while Medusa (Cai et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib1)) utilizes a set of MLPs that predict tokens based on the second-to-top-layer feature of the original LLM. These strategies significantly decrease the latency in generating drafts, leading to improved acceleration. However, their effectiveness is limited by the lower accuracy of the resulting drafts, with Medusa achieving an accuracy of about 0.6, and Lookahead even lower. In contrast, our method attains an accuracy of approximately 0.8.

![Image 2: Refer to caption](https://arxiv.org/html/2401.15077v3/x2.png)

Figure 2: Speedup ratio on the MT-bench for non-greedy (temperature=1) settings. Lookahead is confined to greedy decoding, and the non-greedy generation of Medusa does not guarantee lossless performance. Therefore, EAGLE is not compared with these methods.

To overcome these limitations, we introduce EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), an efficient speculative sampling method, grounded in the following two observations.

Firstly, autoregression at the feature level is simpler than at the token level. In this paper, “features” refer to the second-to-top-layer features of the original LLM, located before the LM head. Compared to token sequences, which are simple transformations of natural language, feature sequences exhibit more regularity. Autoregressively processing at the feature level and then deriving tokens using the LM head of the original LLM yields better results than directly autoregressively predicting tokens. As illustrated in Figure [4](https://arxiv.org/html/2401.15077v3#S1.F4 "Figure 4 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), autoregressively predicting features yields better performance, demonstrated by a higher speedup ratio of 1.9x compared to 1.5x.

Secondly, the uncertainty inherent in the sampling process significantly constrains the performance of predicting the next feature. In text generation, the target LLM predicts the distribution of tokens and samples accordingly, introducing randomness. Features, being high-dimensional and continuous, cannot be treated similarly. As depicted in Figure [3](https://arxiv.org/html/2401.15077v3#S1.F3 "Figure 3 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), sampling different tokens like “am” or “always” leads to distinct feature sequences, introducing ambiguity into the feature-level autoregression. Medusa faces a similar issue in predicting spaced tokens, where it is uncertain whether the true target for the input f I subscript 𝑓 𝐼 f_{I}italic_f start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT should be p a⁢m subscript 𝑝 𝑎 𝑚 p_{am}italic_p start_POSTSUBSCRIPT italic_a italic_m end_POSTSUBSCRIPT or p a⁢l⁢w⁢a⁢y⁢s subscript 𝑝 𝑎 𝑙 𝑤 𝑎 𝑦 𝑠 p_{always}italic_p start_POSTSUBSCRIPT italic_a italic_l italic_w italic_a italic_y italic_s end_POSTSUBSCRIPT. To address this issue, EAGLE inputs the token sequence from one time step ahead, which includes the sampling outcomes, into the draft model. In the example illustrated in Figure [3](https://arxiv.org/html/2401.15077v3#S1.F3 "Figure 3 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), this involves predicting f a⁢l⁢w⁢a⁢y⁢s subscript 𝑓 𝑎 𝑙 𝑤 𝑎 𝑦 𝑠 f_{always}italic_f start_POSTSUBSCRIPT italic_a italic_l italic_w italic_a italic_y italic_s end_POSTSUBSCRIPT based on f I subscript 𝑓 𝐼 f_{I}italic_f start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT and t a⁢l⁢w⁢a⁢y⁢s subscript 𝑡 𝑎 𝑙 𝑤 𝑎 𝑦 𝑠 t_{always}italic_t start_POSTSUBSCRIPT italic_a italic_l italic_w italic_a italic_y italic_s end_POSTSUBSCRIPT, and predicting f a⁢m subscript 𝑓 𝑎 𝑚 f_{am}italic_f start_POSTSUBSCRIPT italic_a italic_m end_POSTSUBSCRIPT based on f I subscript 𝑓 𝐼 f_{I}italic_f start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT and t a⁢m subscript 𝑡 𝑎 𝑚 t_{am}italic_t start_POSTSUBSCRIPT italic_a italic_m end_POSTSUBSCRIPT. As illustrated in Figure [4](https://arxiv.org/html/2401.15077v3#S1.F4 "Figure 4 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), by addressing the uncertainty, the speedup ratio further increases from 1.9x to 2.8x.

![Image 3: Refer to caption](https://arxiv.org/html/2401.15077v3/x3.png)

Figure 3: Uncertainty in feature sequences. The next feature following f I subscript 𝑓 𝐼 f_{I}italic_f start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT is contingent on the sampling outcome and cannot be determined solely based on f I subscript 𝑓 𝐼 f_{I}italic_f start_POSTSUBSCRIPT italic_I end_POSTSUBSCRIPT, where both “always” and “am” are possible to follow the token “I” and lead to two branches.

![Image 4: Refer to caption](https://arxiv.org/html/2401.15077v3/x4.png)

Figure 4: Accuracy and speedup ratio of draft models based on tokens, features and feature&shifted-token at temperature=0, tested on MT-bench with Vicuna 7B as the original LLM. Feature&shifted-token refers to using a feature sequence and a token sequence advanced by one time step as inputs.

We conducted experiments across dialogue, code generation, mathematical reasoning, and instruction following tasks using the MT-bench, HumanEval, GSM8K, and Alpaca datasets, respectively. Tested LLMs included all models from the Vicuna and LLaMA2-Chat series, along with Mixtral 8x7B Instruct. For LLaMA2-Chat 70B, EAGLE achieved a speedup ratio of 2.7x-3.5x, doubled throughput, and theoretically guaranteed the preservation of the generated text’s distribution. Figure [1](https://arxiv.org/html/2401.15077v3#S0.F1 "Figure 1 ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") and [2](https://arxiv.org/html/2401.15077v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") illustrates the performance of EAGLE on the MT-bench (Zheng et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib40)), a highly realistic benchmark simulating actual applications and real-world scenarios, including multi-turn instructions akin to dialogues with ChatGPT. We have chosen to utilize this benchmark as it has been employed by the current state-of-the-art, including Lookahead and Medusa, to demonstrate their speedup ratios. This choice facilitates a fair and direct comparison between our approach and these methods. Compared to the recently proposed speculative sampling-based frameworks, Lookahead and Medusa, EAGLE achieves 1.7x-2.1x and 1.5x-1.6x speedups, respectively. EAGLE operates in parallel with other acceleration or throughput-improving methods, such as quantization, compilation, etc. Combining EAGLE with these techniques could further reduce the operational costs of LLM systems. For example, with gpt-fast (PyTorch Labs, [2023](https://arxiv.org/html/2401.15077v3#bib.bib21)), EAGLE accelerates LLaMA2-Chat 7B decoding to 160.4 tokens/s on a single RTX 3090 GPU.

EAGLE boasts low training costs. For the LLaMA2-Chat 70B model, EAGLE trains a decoder layer with fewer than 1B parameters using no more than 70k dialogues from the ShareGPT dataset. The training is completed in 1-2 days on 4x A100 (40G) GPUs. The training of EAGLE on 7B, 13B and 33B models can even be conducted on a RTX 3090 node in 1-2 days. In practical applications, EAGLE requires only a single training session to provide acceleration for each query. As the number of queries increases, the amortized training cost of EAGLE becomes negligible.

Beyond performance, EAGLE offers additional advantages:

*   •Generality: EAGLE is applicable to any autoregressive LLMs (at least in principle). We have applied EAGLE to LLaMA2-Chat (7B, 13B, 70B), Vicuna (7B, 13B, 33B) and Mixtral 8x7B Instruct in a zero-shot way on the MT-bench, GSM8K, HumanEval and alpaca datasets. EAGLE adheres to the commonly used zero-shot/few-shot settings within the LLM community. All experiments employ the same weights, trained exclusively on the ShareGPT dataset, without any additional training on the evaluation datasets. The method adds only a lightweight plug-in (a single transformer decoder layer) to the LLM, which can be easily deployed in a production environment. 
*   •Reliability: EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings. This is in sharp contrast to Lookahead and Medusa which either focus solely on greedy settings or do not guarantee the preservation of distribution in these settings. 

2 Preliminaries
---------------

Notations. In this paper, “target LLM” denotes the LLM intended for acceleration, while “draft model” refers to the model used for draft generation. “Feature” generally signifies the second-to-top-layer feature of a LLM, the hidden state before the LM head. Tokens are denoted by lowercase t 𝑡 t italic_t, their embeddings by e 𝑒 e italic_e, features by f 𝑓 f italic_f, and distributions by p 𝑝 p italic_p. Sequences are represented in uppercase, for example, T i:j subscript 𝑇:𝑖 𝑗 T_{i:j}italic_T start_POSTSUBSCRIPT italic_i : italic_j end_POSTSUBSCRIPT for (t i,t i+1,…,t j)subscript 𝑡 𝑖 subscript 𝑡 𝑖 1…subscript 𝑡 𝑗(t_{i},t_{i+1},\ldots,t_{j})( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_t start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT , … , italic_t start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ). In a LLM, input T 1:j subscript 𝑇:1 𝑗 T_{1:j}italic_T start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT is transformed into embeddings E 1:j subscript 𝐸:1 𝑗 E_{1:j}italic_E start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT through the embedding layer, then to features F 1:j subscript 𝐹:1 𝑗 F_{1:j}italic_F start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT, and the LM Head maps f j subscript 𝑓 𝑗 f_{j}italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT to a distribution p j+1=LM_Head⁢(f j)subscript 𝑝 𝑗 1 LM_Head subscript 𝑓 𝑗 p_{j+1}=\text{LM\_Head}(f_{j})italic_p start_POSTSUBSCRIPT italic_j + 1 end_POSTSUBSCRIPT = LM_Head ( italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ), sampling the next token t j+1 subscript 𝑡 𝑗 1 t_{j+1}italic_t start_POSTSUBSCRIPT italic_j + 1 end_POSTSUBSCRIPT. Vanilla autoregression at the token level is described by T 1:j→E 1:j→f j→p j+1→t j+1→subscript 𝑇:1 𝑗 subscript 𝐸:1 𝑗→subscript 𝑓 𝑗→subscript 𝑝 𝑗 1→subscript 𝑡 𝑗 1 T_{1:j}\rightarrow E_{1:j}\rightarrow f_{j}\rightarrow p_{j+1}\rightarrow t_{j% +1}italic_T start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT → italic_E start_POSTSUBSCRIPT 1 : italic_j end_POSTSUBSCRIPT → italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT → italic_p start_POSTSUBSCRIPT italic_j + 1 end_POSTSUBSCRIPT → italic_t start_POSTSUBSCRIPT italic_j + 1 end_POSTSUBSCRIPT for any integer j≥1 𝑗 1 j\geq 1 italic_j ≥ 1.

Speculative sampling. Speculative sampling operates through draft and verification phases, with the drafting phase using a smaller model to generate γ 𝛾\gamma italic_γ tokens T^j+1:j+γ subscript^𝑇:𝑗 1 𝑗 𝛾\hat{T}_{j+1:j+\gamma}over^ start_ARG italic_T end_ARG start_POSTSUBSCRIPT italic_j + 1 : italic_j + italic_γ end_POSTSUBSCRIPT and their distributions P^j+1:j+γ subscript^𝑃:𝑗 1 𝑗 𝛾\hat{P}_{j+1:j+\gamma}over^ start_ARG italic_P end_ARG start_POSTSUBSCRIPT italic_j + 1 : italic_j + italic_γ end_POSTSUBSCRIPT. In the verification phase, a single forward pass of the target LLM yields the probabilities P j+1:j+γ subscript 𝑃:𝑗 1 𝑗 𝛾 P_{j+1:j+\gamma}italic_P start_POSTSUBSCRIPT italic_j + 1 : italic_j + italic_γ end_POSTSUBSCRIPT. Tokens are then sequentially evaluated, with a token t^j+i subscript^𝑡 𝑗 𝑖\hat{t}_{j+i}over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT having an acceptance probability min⁡(1,p j+i⁢(t^j+i)/p^j+i⁢(t^j+i))1 subscript 𝑝 𝑗 𝑖 subscript^𝑡 𝑗 𝑖 subscript^𝑝 𝑗 𝑖 subscript^𝑡 𝑗 𝑖\min(1,p_{j+i}(\hat{t}_{j+i})/\hat{p}_{j+i}(\hat{t}_{j+i}))roman_min ( 1 , italic_p start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ) / over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ) ). Upon the rejection of a token t^j+i subscript^𝑡 𝑗 𝑖\hat{t}_{j+i}over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT, all subsequent tokens are discarded, and this token is resampled based on a distribution norm⁢(max⁡(0,p j+i−p^j+i))norm 0 subscript 𝑝 𝑗 𝑖 subscript^𝑝 𝑗 𝑖\text{norm}(\max(0,p_{j+i}-\hat{p}_{j+i}))norm ( roman_max ( 0 , italic_p start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT - over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_j + italic_i end_POSTSUBSCRIPT ) ). As proven in Appendix A.1 of speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17)), this method equates to sampling directly from the target LLM. EAGLE adopts this method, ensuring that the distribution of the generated text remains unchanged for both the greedy and non-greedy settings.

3 EAGLE
-------

EAGLE, aligning with other speculative sampling-based methods, incorporates both a drafting phase and a verification phase.

### 3.1 Drafting phase

The primary distinction between EAGLE and other methods lies predominantly in the drafting phase. Figure [5](https://arxiv.org/html/2401.15077v3#S3.F5 "Figure 5 ‣ 3.1 Drafting phase ‣ 3 EAGLE ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") illustrates a schematic of the drafting phase for different methods. Speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17); Chen et al., [2023a](https://arxiv.org/html/2401.15077v3#bib.bib2)) and Lookahead (Fu et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib6)) predict tokens based on tokens. Medusa(Cai et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib1)) independently predicts t 4 subscript 𝑡 4 t_{4}italic_t start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT and t 5 subscript 𝑡 5 t_{5}italic_t start_POSTSUBSCRIPT 5 end_POSTSUBSCRIPT using the feature f 2 subscript 𝑓 2 f_{2}italic_f start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT from the target LLM. EAGLE predicts f 3 subscript 𝑓 3 f_{3}italic_f start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT using the feature sequence (f 1,f 2)subscript 𝑓 1 subscript 𝑓 2(f_{1},f_{2})( italic_f start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_f start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) and the token sequence (t 2,t 3)subscript 𝑡 2 subscript 𝑡 3(t_{2},t_{3})( italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_t start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ), advanced by one time step. From p 4=LM Head⁢(f 3)subscript 𝑝 4 LM Head subscript 𝑓 3 p_{4}=\text{LM Head}(f_{3})italic_p start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT = LM Head ( italic_f start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ), t 4 subscript 𝑡 4 t_{4}italic_t start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT is sampled. Subsequently, f 3 subscript 𝑓 3 f_{3}italic_f start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT and t 4 subscript 𝑡 4 t_{4}italic_t start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT are concatenated into the input sequence to predict the next feature f 4 subscript 𝑓 4 f_{4}italic_f start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT and sample the subsequent token t 5 subscript 𝑡 5 t_{5}italic_t start_POSTSUBSCRIPT 5 end_POSTSUBSCRIPT.

![Image 5: Refer to caption](https://arxiv.org/html/2401.15077v3/x5.png)

Figure 5: A comparison of the methods for drafting the fourth and fifth tokens, t 4 subscript 𝑡 4 t_{4}italic_t start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT and t 5 subscript 𝑡 5 t_{5}italic_t start_POSTSUBSCRIPT 5 end_POSTSUBSCRIPT. t 𝑡 t italic_t (represented by blue blocks) denotes tokens, and f 𝑓 f italic_f (orange blocks) signifies the features, with subscripts indicating their positions in the sequence. The red border indicates the predictions of the draft model. For simplicity, the n 𝑛 n italic_n in the n 𝑛 n italic_n-gram for Lookahead, as shown in the figure, has been set to 2.

As illustrated in Figure [6](https://arxiv.org/html/2401.15077v3#S3.F6 "Figure 6 ‣ 3.1 Drafting phase ‣ 3 EAGLE ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), EAGLE’s draft model comprises three modules: the Embedding layer, LM Head, and Autoregression Head. The Embedding layer and LM Head employ the parameters of the target LLM and do not necessitate additional training. The draft model takes as input a feature sequence of shape (bs, seq_len, hidden_dim) and an advanced token sequence of shape (bs, seq_len). It then converts the token sequence into a token embedding sequence of shape (bs, seq_len, hidden_dim), and concatenates it to form a fused sequence of shape (bs, seq_len, 2×\times×hidden_dim). The Autoregression Head consisting of an FC layer and a decoder layer. The FC layer reduces the dimensionality of the fused sequence to (bs, seq_len, hidden_dim) and then we utilize the decoder layer to predict the next feature. The LM Head calculates the distribution based on the feature, from which the next token is sampled. Finally, the predicted feature and the sampled token are concatenated into the input, facilitating the continuation of the autoregressive process. EAGLE creates a tree-structured draft using tree attention, generating a draft tree with depth m 𝑚 m italic_m and more than m 𝑚 m italic_m tokens through m 𝑚 m italic_m forward passes. For instance, as shown in Figure [6](https://arxiv.org/html/2401.15077v3#S3.F6 "Figure 6 ‣ 3.1 Drafting phase ‣ 3 EAGLE ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), EAGLE drafts a 10-token tree with just 3 forward passes. The actual tree structure employed by EAGLE is detailed in Appendix [A.1](https://arxiv.org/html/2401.15077v3#A1.SS1 "A.1 Tree Structure ‣ Appendix A Implementation Details ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty").

![Image 6: Refer to caption](https://arxiv.org/html/2401.15077v3/x6.png)

Figure 6: Pipeline of EAGLE. The upper section illustrates the computational process, while the lower section displays the corresponding generation results for each step. In the upper section, green blocks represent token embeddings, orange blocks represent features, red boxes indicate the predictions of the draft model, and blue modules with snowflake icons represent the use of target LLM parameters, which are not subject to training.

### 3.2 Training of the draft models

Predicting the next feature constitutes a regression task, for which we employ Smooth L1 loss (see Figure [5](https://arxiv.org/html/2401.15077v3#S3.F5 "Figure 5 ‣ 3.1 Drafting phase ‣ 3 EAGLE ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") EAGLE):

L r⁢e⁢g=Smooth L1⁢(f i+1,Draft_Model⁢(T 2:i+1,F 1:i)).subscript 𝐿 𝑟 𝑒 𝑔 Smooth L1 subscript 𝑓 𝑖 1 Draft_Model subscript 𝑇:2 𝑖 1 subscript 𝐹:1 𝑖 L_{reg}=\text{Smooth L1}(f_{i+1},\text{Draft\_Model}(T_{2:i+1},F_{1:i})).italic_L start_POSTSUBSCRIPT italic_r italic_e italic_g end_POSTSUBSCRIPT = Smooth L1 ( italic_f start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT , Draft_Model ( italic_T start_POSTSUBSCRIPT 2 : italic_i + 1 end_POSTSUBSCRIPT , italic_F start_POSTSUBSCRIPT 1 : italic_i end_POSTSUBSCRIPT ) ) .

Predicting features is an intermediary objective of the draft model, with the ultimate goal being the prediction of tokens to generate a sequence of tokens. Consequently, we also employ classification loss to directly optimize towards this final objective:

p i+2=Softmax⁢(LM_Head⁢(f i+1)),subscript 𝑝 𝑖 2 Softmax LM_Head subscript 𝑓 𝑖 1\displaystyle{p}_{i+2}=\text{Softmax}(\text{LM\_Head}({f}_{i+1})),italic_p start_POSTSUBSCRIPT italic_i + 2 end_POSTSUBSCRIPT = Softmax ( LM_Head ( italic_f start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT ) ) ,
p^i+2=Softmax⁢(LM_Head⁢(f^i+1)),subscript^𝑝 𝑖 2 Softmax LM_Head subscript^𝑓 𝑖 1\displaystyle\hat{p}_{i+2}=\text{Softmax}(\text{LM\_Head}(\hat{f}_{i+1})),over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_i + 2 end_POSTSUBSCRIPT = Softmax ( LM_Head ( over^ start_ARG italic_f end_ARG start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT ) ) ,
L c⁢l⁢s=Cross_Entropy⁢(p i+2,p^i+2).subscript 𝐿 𝑐 𝑙 𝑠 Cross_Entropy subscript 𝑝 𝑖 2 subscript^𝑝 𝑖 2\displaystyle L_{cls}=\text{Cross\_Entropy}({p}_{i+2},\hat{p}_{i+2}).italic_L start_POSTSUBSCRIPT italic_c italic_l italic_s end_POSTSUBSCRIPT = Cross_Entropy ( italic_p start_POSTSUBSCRIPT italic_i + 2 end_POSTSUBSCRIPT , over^ start_ARG italic_p end_ARG start_POSTSUBSCRIPT italic_i + 2 end_POSTSUBSCRIPT ) .

By integrating regression loss and classification loss, we train the Autoregression Head using the combined loss function L=L r⁢e⁢g+w c⁢l⁢s⁢L c⁢l⁢s 𝐿 subscript 𝐿 𝑟 𝑒 𝑔 subscript 𝑤 𝑐 𝑙 𝑠 subscript 𝐿 𝑐 𝑙 𝑠 L=L_{reg}+w_{cls}L_{cls}italic_L = italic_L start_POSTSUBSCRIPT italic_r italic_e italic_g end_POSTSUBSCRIPT + italic_w start_POSTSUBSCRIPT italic_c italic_l italic_s end_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT italic_c italic_l italic_s end_POSTSUBSCRIPT. Typically, the classification loss is an order of magnitude larger than the regression loss in numerical terms. Consequently, we set w c⁢l⁢s subscript 𝑤 𝑐 𝑙 𝑠 w_{cls}italic_w start_POSTSUBSCRIPT italic_c italic_l italic_s end_POSTSUBSCRIPT to 0.1.

EAGLE’s Autoregression Head is ideally trained with autoregressively generated text from the target LLM, yet this approach is costly. Fortunately, EAGLE exhibits low sensitivity to training data (ablation study in Section [4.3.3](https://arxiv.org/html/2401.15077v3#S4.SS3.SSS3 "4.3.3 Training data ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")). Instead of employing text generated by the target LLM, we utilize a fixed dataset, substantially reducing the overhead. During the drafting phase, EAGLE autoregressively processes features. Inaccuracies in features can lead to error accumulation. To mitigate this issue, we employ data augmentation by adding random noise sampled from a uniform distribution 𝒰⁢(−0.1,0.1)𝒰 0.1 0.1\mathcal{U}(-0.1,0.1)caligraphic_U ( - 0.1 , 0.1 ) to features of the target LLM during training(Jain et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib13)).

### 3.3 Verification phase

Employing tree attention, the target LLM computes the probability of each token in the tree-structured draft through a single forward pass. At every node of the draft tree, we recursively apply speculative sampling algorithms to sample or adjust the distribution (details in Appendix [A.2](https://arxiv.org/html/2401.15077v3#A1.SS2 "A.2 Multi-Round Speculative Sampling ‣ Appendix A Implementation Details ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")), consistent with SpecInfer (Miao et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib19)), ensuring that the distribution of the output text aligns with that of the target LLM. Concurrently, we document accepted tokens and their features for use in the next drafting phase.

4 Experiments
-------------

Models and tasks. We conducted experiments on Vicuna models (7B, 13B, 33B), LLaMA2-chat models (7B, 13B, 70B), and Mixtral 8x7B Instruct, encompassing the common sizes of current mainstream LLMs. We evaluated EAGLE across multiple tasks including multi-turn dialogue, code generation, mathematical reasoning, and instruction following, employing the MT-bench (Zheng et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib40)), HumanEval (Chen et al., [2021](https://arxiv.org/html/2401.15077v3#bib.bib3)), GSM8K (Cobbe et al., [2021](https://arxiv.org/html/2401.15077v3#bib.bib5)), and Alpaca (Taori et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib29)) datasets, respectively. Speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17)) conducted experiments with a batch size of 1, a setting subsequently adopted by other works such as DistillSpec (Zhou et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib41)) and BiLD (Kim et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib15)). Similarly, the majority of our experiments also adopted this setting. Experiments with a batch size greater than 1 are presented in Section [4.4](https://arxiv.org/html/2401.15077v3#S4.SS4 "4.4 Batch size and throughput ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty").

Metrics. Like other speculative sampling-based methods, EAGLE primarily focuses on latency rather than throughput. We assess acceleration effects using the following metrics:

*   •Walltime speedup ratio: The actual test speedup ratio relative to vanilla autoregressive decoding. 
*   •Average acceptance length τ 𝜏\tau italic_τ: The average number of tokens accepted per forward pass of the target LLM. 
*   •Acceptance rate α 𝛼\alpha italic_α: The ratio of accepted to generated tokens during drafting, gauges draft accuracy. It’s less applicable for tree drafts due to multiple tokens sampled per location with only one accepted. Hence, when measuring this metric, we utilize chain drafts without tree attention, aligning with speculative sampling and DistillSpec. EAGLE’s draft model inputs feature and token sequences. Autoregressive feature processing can propagate errors, so we measure the acceptance rate as n⁢-⁢α 𝑛-𝛼 n\text{-}\alpha italic_n - italic_α, considering n 𝑛 n italic_n features predicted by the draft model, potentially with inaccuracies. 

Acceleration of EAGLE theoretically guarantees the preservation of the target LLMs’ output distribution. Consequently, evaluating the quality of EAGLE’s generated results is both unnecessary and meaningless.

Training. We fixed the target LLMs. EAGLE was trained on the ShareGPT dataset, utilizing 68,000 dialogue iterations with a learning rate set at 3e-5. We employed the AdamW optimizer with beta values (β 1,β 2)subscript 𝛽 1 subscript 𝛽 2(\beta_{1},\beta_{2})( italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) set to (0.9, 0.95) and implemented gradient clipping of 0.5. The trainable parameters of EAGLE corresponding to the 7B, 13B, 33B, and 70B models are 0.24B, 0.37B, 0.56B, and 0.99B, respectively. The trainable parameters of EAGLE for MoE model Mixtral 8x7B is 0.28B. EAGLE is characterized by its low training cost; the Autoregression Head is trainable within 1-2 days on an A100 40G server for the 70B models.

### 4.1 Effectiveness

Figures [1](https://arxiv.org/html/2401.15077v3#S0.F1 "Figure 1 ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") and [2](https://arxiv.org/html/2401.15077v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), along with Table [1](https://arxiv.org/html/2401.15077v3#S4.T1 "Table 1 ‣ 4.1 Effectiveness ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), display the speedup ratios of EAGLE. EAGLE demonstrates better acceleration at temperature=0 compared to temperature=1. For instance, for LLaMA2-Chat 13B at temperature=0, the speedup ratios range from 3.01x to 3.76x, while at temperature=1, they range from 2.66x to 2.89x. In code generation tasks (HumanEval), EAGLE achieves its best acceleration performance. This is attributed to the prevalence of fixed templates in code, making it easier to generate drafts for these templates. Compared to recently introduced speculative sampling-based methods, Lookahead and Medusa, EAGLE is faster by 1.70x-2.08x and 1.47x-1.60x, respectively. Employing speculative sampling in the Vicuna and LLaMA2-Chat series is challenging. For the 7B model, there is no suitable draft model. For other sizes, using the 7B model as the draft model, we iterated through draft lengths from 2 to 10 and reported the highest speedup ratio. For the 13B model, we observed no improvement in speed. For the 33B and 70B models, the speedup ratios were 1.12x and 1.88x, respectively. For DistillSpec, to ensure fairness, we used the same training data as EAGLE. Additionally, the divergence function employed follows the FKL as detailed in Appendix A.1 of the DistillSpec paper. While distillation slightly improved the speedup ratio, the limited enhancement is because distillation aims to increase the draft model’s acceptance rate, while the bottleneck for speculative sampling performance lies in the high overhead of the draft model.

Tables [1](https://arxiv.org/html/2401.15077v3#S4.T1 "Table 1 ‣ 4.1 Effectiveness ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") and [2](https://arxiv.org/html/2401.15077v3#S4.T2 "Table 2 ‣ 4.1 Effectiveness ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") indicate that in EAGLE, the target LLM generates 3.2-4.5 tokens per forward pass, surpassing vanilla decoding which produces only one token per forward pass, thereby significantly increasing generation speed. As shown in Figure [2](https://arxiv.org/html/2401.15077v3#S4.T2 "Table 2 ‣ 4.1 Effectiveness ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") and Appendix [B](https://arxiv.org/html/2401.15077v3#A2 "Appendix B Detailed experimental results ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), the acceptance rate for completely accurate feature sequences, 0⁢-⁢α 0-𝛼 0\text{-}\alpha 0 - italic_α, significantly exceeds that for sequences with a single erroneous feature, 1⁢-⁢α 1-𝛼 1\text{-}\alpha 1 - italic_α, indicating the impact of feature errors on draft model performance. Yet, the slight variation between 1⁢-⁢α 1-𝛼 1\text{-}\alpha 1 - italic_α to 4⁢-⁢α 4-𝛼 4\text{-}\alpha 4 - italic_α underscores EAGLE’s robustness to feature errors and its adept handling of error accumulation.

Table 1: Speedup ratio and average acceptance length τ 𝜏\tau italic_τ on HumanEval, GSM8K, and Alpaca. T denotes temperature, V represents Vicuna, and LC stands for LLaMA2-Chat.

Table 2: Average acceptance length τ 𝜏\tau italic_τ and acceptance rate α 𝛼\alpha italic_α on MT-bench. T denotes temperature.

Table [3](https://arxiv.org/html/2401.15077v3#S4.T3 "Table 3 ‣ 4.1 Effectiveness ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") reveals that EAGLE achieved a 1.5x speedup with the Mixtral 8x7B Instruct model. This modest acceleration, compared to models like LLaMA, is due to a shorter average acceptance length and the complexity of accelerating MoE models via speculative sampling. MoE models typically require reading the weights of only two experts per token during vanilla autoregressive decoding. However, during the verification phase of speculative sampling, processing multiple tokens may necessitate accessing the weights of more than two experts, contrasting with dense decoder-only models where all weights are read regardless of the number of tokens forwarded.

Table 3: Speedup ratio, average acceptance length τ 𝜏\tau italic_τ, and acceptance rate α 𝛼\alpha italic_α on MT-bench at temperature=0. The target LLM is Mixtral 8x7B Instruct-v0.1.

### 4.2 Case study: EAGLE + gpt-fast

EAGLE is compatible with other acceleration technologies. We conducted experiments combining EAGLE with gpt-fast, which employs quantization and compilation for acceleration. As shown in Figure [4](https://arxiv.org/html/2401.15077v3#S4.T4 "Table 4 ‣ 4.2 Case study: EAGLE + gpt-fast ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), by integrating EAGLE with gpt-fast, we increased the generation speed of LLaMA2-Chat 7B on a single RTX 3090 to 160.4 tokens/s.

Table 4: Generation speed of EAGLE combined with gpt-fast, evaluated on MT-bench with LLaMA2-Chat 7B at temperature=0.

### 4.3 Ablation study

#### 4.3.1 Tree attention

EAGLE, similar to SpecInfer and Medusa, employs tree attention, where both the generation and validation of drafts are tree-structured. In contrast, methods like speculative sampling do not use tree attention, resulting in chain-structured draft generation and validation. Table [5](https://arxiv.org/html/2401.15077v3#S4.T5 "Table 5 ‣ 4.3.1 Tree attention ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") and Figure [7](https://arxiv.org/html/2401.15077v3#S4.F7 "Figure 7 ‣ 4.3.1 Tree attention ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") present comparative results indicating the impact of using tree attention. The implementation of tree draft and verification in EAGLE results in an approximate increase of 0.6-0.8 in the average acceptance length and about 0.3-0.5 in the speedup ratio. Compared to chain draft and verification, tree draft and verification do not increase the number of forward passes in the model (both the target LLM and the draft model), but they do increase the number of tokens processed per forward pass. Consequently, the improvement in the speedup ratio is less pronounced than the increase in average acceptance length. Notably, even without employing tree draft and verification, EAGLE demonstrates a significant acceleration effect, approximately in the range of 2.3x-2.7x.

![Image 7: Refer to caption](https://arxiv.org/html/2401.15077v3/x7.png)

Figure 7: Speedup ratios of EAGLE with and without the use of tree attention. The evaluation dataset is MT-bench, with the temperature parameter set to 0.

Table 5: Average acceptance length τ 𝜏\tau italic_τ of EAGLE with and without the use of tree attention. The evaluation dataset is MT-bench, with the temperature parameter set to 0.

#### 4.3.2 Inputs of draft models

Compared to other speculative sampling-based methods, the key innovation of EAGLE lies in its utilization of features computed by the target LLM and the incorporation of sampling outcomes into the input of the draft model to address randomness. We conducted an ablation study on Vicuna 7B, assessing draft models with varying inputs. We tested four types of inputs: feature&shifted-token (EAGLE), feature&unshifted-token, token, and feature. Both feature&shifted-token (EAGLE) and feature&unshifted-token integrate semantic information at different levels. The distinction lies in the fact that feature&shifted-token (EAGLE) inputs tokens advanced by one time step, equipping it to address randomness effectively. Apart from the use of a FC layer to reduce dimensionality for the feature&token input, the structure of the draft model remains entirely consistent. Figure [8](https://arxiv.org/html/2401.15077v3#S4.F8 "Figure 8 ‣ 4.3.2 Inputs of draft models ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") presents the experimental outcomes on the MT-bench with Vicuna 7B as the target LLM. Three observations can be drawn.

*   •First, when the number of parameters of the draft model is limited, utilizing features yields slightly better results than tokens. 
*   •Second, merging features and tokens modestly boosts performance, mainly as discrete, error-free tokens mitigate feature error accumulation, evident from the similar 0⁢-⁢α 0-𝛼 0\text{-}\alpha 0 - italic_α of feature&unshifted-token and feature-only draft models, with a significantly improved 1⁢-⁢α 1-𝛼 1\text{-}\alpha 1 - italic_α. 
*   •Third, addressing the randomness inherent in the sampling process results in the most significant improvement. The feature&shifted-token scheme, compared to feature&unshifted-token, adds no complexity yet markedly enhances the draft model’s capability by simply advancing the token by one time step, allowing the draft model to account for the randomness in sampling. 

![Image 8: Refer to caption](https://arxiv.org/html/2401.15077v3/x8.png)

Figure 8: Performance of draft models with varying inputs. The target LLM is Vicuna 7B, and the test dataset is MT-bench. Speed refers to the walltime speedup ratio, τ 𝜏\tau italic_τ denotes the average acceptance length, 0⁢-⁢α 0-𝛼 0\text{-}\alpha 0 - italic_α represents the acceptance rate with entirely precise inputs, 1⁢-⁢α 1-𝛼 1\text{-}\alpha 1 - italic_α indicates the acceptance rate when the input includes one imprecise feature, and T 𝑇 T italic_T refers to the temperature.

#### 4.3.3 Training data

EAGLE uses a fixed dataset for training, avoiding increased overhead from using the target LLM for generating training data. Ablation study (see Table [6](https://arxiv.org/html/2401.15077v3#S4.T6 "Table 6 ‣ 4.3.3 Training data ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")) shows that data from the target LLM marginally improves performance, indicating EAGLE’s low sensitivity to training data and justifying the fixed dataset approach for cost reduction.

Table 6: The speedup ratios and average acceptance length τ 𝜏\tau italic_τ using different training datasets evaluated on the MT-bench, with the target LLM being LLaMA2-Chat 7B and the temperature set to 0. “Fixed dataset” refers to both questions and answers originating from the ShareGPT dataset. “Data generated by target LLM” denotes that while questions are sourced from the ShareGPT dataset, the answers are generated by the target LLM.

### 4.4 Batch size and throughput

Inference in LLMs is memory-bound (Patterson, [2004](https://arxiv.org/html/2401.15077v3#bib.bib20); Shazeer, [2019](https://arxiv.org/html/2401.15077v3#bib.bib24)), leaving GPU computational resources underutilized. The principle behind the speculative sampling-based approach in enhancing generation speed lies in more effectively utilizing GPU computational resources. As the batch size increases, the available computational capacity of the GPU decreases, leading to a reduction in the acceleration effect. In this section, we present experimental results for scenarios where the batch size exceeds 1. As demonstrated in Table [7](https://arxiv.org/html/2401.15077v3#S4.T7 "Table 7 ‣ 4.4 Batch size and throughput ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), the speedup ratio diminishes with increasing batch size. When using Vicuna 7B as the target LLM, the speedup ratio at bs=4 is higher than at bs=3. This is attributed to the fact that, during the verification phase of EAGLE, the target LLM processes multiple tokens in a single forward pass, and the processing at bs=4 is faster than at bs=3. In contrast, with vanilla autoregressive decoding where the target LLM processes one token per forward pass, the speeds at bs=3 and bs=4 are nearly identical.

Although speculative sampling-based methods predominantly focus on latency, we also investigated EAGLE’s throughput for batch size >1 absent 1>1> 1, another key metric for LLM systems. Compared to vanilla autoregressive decoding, EAGLE requires slightly more CUDA memory. For Vicuna 7B as the target LLM, operating under a memory constraint of a single RTX 3090 with 24G of CUDA memory, the maximum batch size (bs) for vanilla autoregressive decoding and EAGLE are 8 and 7, respectively. In the case of LLaMA2-Chat 70B, constrained by 4 A100 (40G) GPUs totaling 160G of CUDA memory, the maximum bs for vanilla autoregressive decoding and EAGLE are 5 and 4, respectively. All evaluations were conducted at FP16 precision. We calculated the throughput for different bs and selected the maximum value. Both vanilla autoregressive decoding and EAGLE achieve maximum throughput at their respective maximum bs. Tree attention consumes more computational resources. At bs=7, the computational resources are less abundant, making the non-use of tree attention more advantageous. As illustrated in Table [7](https://arxiv.org/html/2401.15077v3#S4.T7 "Table 7 ‣ 4.4 Batch size and throughput ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"), EAGLE achieves a 2x increase in throughput.

Table 7: Speedup ratios at different batch sizes and throughput of EAGLE. The evaluation dataset is MT-bench, with the temperature parameter set to 0.

5 Related Work
--------------

There has been considerable research into accelerating language models, involving techniques such as distillation (Hinton et al., [2015](https://arxiv.org/html/2401.15077v3#bib.bib10)), quantization (Hubara et al., [2018](https://arxiv.org/html/2401.15077v3#bib.bib12); Shen et al., [2020](https://arxiv.org/html/2401.15077v3#bib.bib25); Kim et al., [2021](https://arxiv.org/html/2401.15077v3#bib.bib14); Zadeh et al., [2020](https://arxiv.org/html/2401.15077v3#bib.bib36); Zafrir et al., [2019](https://arxiv.org/html/2401.15077v3#bib.bib37)), pruning (Gale et al., [2019](https://arxiv.org/html/2401.15077v3#bib.bib7); Sanh et al., [2020](https://arxiv.org/html/2401.15077v3#bib.bib22); Kurtic et al., [2022](https://arxiv.org/html/2401.15077v3#bib.bib16); Voita et al., [2019](https://arxiv.org/html/2401.15077v3#bib.bib31)), and innovative network architecture designs (Gu & Dao, [2023](https://arxiv.org/html/2401.15077v3#bib.bib8); Wu et al., [2020](https://arxiv.org/html/2401.15077v3#bib.bib32)). These methods aim to reduce the latency per forward pass.

Similar to our approach are frameworks based on speculative sampling. Early works (Stern et al., [2018](https://arxiv.org/html/2401.15077v3#bib.bib27); Sun et al., [2021](https://arxiv.org/html/2401.15077v3#bib.bib28)) accelerated greedy decoding, while speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17); Chen et al., [2023a](https://arxiv.org/html/2401.15077v3#bib.bib2)) extended it to non-greedy sampling, provably maintaining the original output distribution. Ensuring unchanged output distribution makes acceleration more challenging; many studies have explored lossy acceleration as a trade-off. For instance, DistillSpec (Zhou et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib41)) modifies acceptance probabilities using a lenience function, BiLD (Kim et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib15)) accepts drafts if the distance metric from the target LLM distribution is below a certain threshold, and Medusa (Cai et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib1)) uses a minimum of a hard threshold and an entropy-dependent threshold for truncation. In contrast, EAGLE does not employ any relaxations and maintains the output distribution of the LLM unchanged.

The primary differences among speculative sampling-based methods manifest predominantly in the drafting phase. Speculative sampling (Leviathan et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib17); Chen et al., [2023a](https://arxiv.org/html/2401.15077v3#bib.bib2)) utilizes a lower-parameter version of the target LLM as the draft model. Self-Speculative Decoding (Zhang et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib38)) skips some layers of the target LLM during draft generation. SpecInfer (Miao et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib19)) employs a set of small models to generate drafts in parallel. Cascade Speculative Drafting (Chen et al., [2023b](https://arxiv.org/html/2401.15077v3#bib.bib4)) and Staged Speculative Decoding (Spector & Re, [2023](https://arxiv.org/html/2401.15077v3#bib.bib26)) cascade different overhead draft models. Online Speculative Decoding (Liu et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib18)) trains the draft model on a distribution of queries. Methods (Hooper et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib11); Fu et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib6); Yang et al., [2023b](https://arxiv.org/html/2401.15077v3#bib.bib35)) such as Medusa (Cai et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib1)) do not employ a separate target LLM; instead, they generate drafts by utilizing features or weights from the target LLM. REST (He et al., [2023](https://arxiv.org/html/2401.15077v3#bib.bib9)) generates drafts based on retrieval methods. LLMA (Yang et al., [2023a](https://arxiv.org/html/2401.15077v3#bib.bib34)), used for tasks like grammatical correction where input and output overlap, retrieves drafts directly from the input.

6 Conclusion
------------

In this paper, we introduce EAGLE, an efficient framework for speculative sampling. EAGLE conducts the drafting process autoregressively at the more structured (second-to-top-layer) feature level and mitigates sampling uncertainty in predicting the next feature by incorporating tokens from one time step ahead. EAGLE is guaranteed to preserve the output distribution of the LLM while significantly enhancing generation speed. On MT-bench, EAGLE is 2.1x-3.8x faster than vanilla autoregressive decoding, 1.7x-2.1x faster than Lookahead, and 1.5x-1.6x faster than Medusa.

Acknowledgements. We acknowledge useful discussions with the Medusa’s team leader Tianle Cai, the Lookahead’s team leader Hao Zhang, the SpecTr’s team leader Ziteng Sun, interactions with the gpt-fast team leaders Horace He and Soumith Chintala on X, and Yihan Wu.

References
----------

*   Cai et al. (2023) Cai, T., Li, Y., Geng, Z., Peng, H., and Dao, T. Medusa: Simple framework for accelerating LLM generation with multiple decoding heads. [https://github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa), 2023. 
*   Chen et al. (2023a) Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. Accelerating large language model decoding with speculative sampling. _arXiv preprint arXiv:2302.01318_, 2023a. 
*   Chen et al. (2021) Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_, 2021. 
*   Chen et al. (2023b) Chen, Z., Yang, X., Lin, J., Sun, C., Huang, J., and Chang, K. C.-C. Cascade speculative drafting for even faster LLM inference. _arXiv preprint arXiv:2312.11462_, 2023b. 
*   Cobbe et al. (2021) Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   Fu et al. (2023) Fu, Y., Bailis, P., Stoica, I., and Zhang, H. Breaking the sequential dependency of LLM inference using lookahead decoding, November 2023. URL [https://lmsys.org/blog/2023-11-21-lookahead-decoding/](https://lmsys.org/blog/2023-11-21-lookahead-decoding/). 
*   Gale et al. (2019) Gale, T., Elsen, E., and Hooker, S. The state of sparsity in deep neural networks.(2019). _arXiv preprint cs.LG/1902.09574_, 2019. 
*   Gu & Dao (2023) Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_, 2023. 
*   He et al. (2023) He, Z., Zhong, Z., Cai, T., Lee, J.D., and He, D. Rest: Retrieval-based speculative decoding. _arXiv preprint arXiv:2311.08252_, 2023. 
*   Hinton et al. (2015) Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_, 2015. 
*   Hooper et al. (2023) Hooper, C., Kim, S., Mohammadzadeh, H., Genc, H., Keutzer, K., Gholami, A., and Shao, S. Speed: Speculative pipelined execution for efficient decoding. _arXiv preprint arXiv:2310.12072_, 2023. 
*   Hubara et al. (2018) Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., and Bengio, Y. Quantized neural networks: Training neural networks with low precision weights and activations. _journal of machine learning research_, 18(187):1–30, 2018. 
*   Jain et al. (2023) Jain, N., Chiang, P.-y., Wen, Y., Kirchenbauer, J., Chu, H.-M., Somepalli, G., Bartoldson, B.R., Kailkhura, B., Schwarzschild, A., Saha, A., et al. NEFTune: Noisy embeddings improve instruction finetuning. _arXiv preprint arXiv:2310.05914_, 2023. 
*   Kim et al. (2021) Kim, S., Gholami, A., Yao, Z., Mahoney, M.W., and Keutzer, K. I-bert: Integer-only bert quantization. In _International conference on machine learning_, pp. 5506–5518. PMLR, 2021. 
*   Kim et al. (2023) Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M.W., Gholami, A., and Keutzer, K. Speculative decoding with big little decoder. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. 
*   Kurtic et al. (2022) Kurtic, E., Campos, D., Nguyen, T., Frantar, E., Kurtz, M., Fineran, B., Goin, M., and Alistarh, D. The optimal bert surgeon: Scalable and accurate second-order pruning for large language models. _arXiv preprint arXiv:2203.07259_, 2022. 
*   Leviathan et al. (2023) Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning_, pp. 19274–19286. PMLR, 2023. 
*   Liu et al. (2023) Liu, X., Hu, L., Bailis, P., Stoica, I., Deng, Z., Cheung, A., and Zhang, H. Online speculative decoding. _arXiv preprint arXiv:2310.07177_, 2023. 
*   Miao et al. (2023) Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Wong, R. Y.Y., Chen, Z., Arfeen, D., Abhyankar, R., and Jia, Z. SpecInfer: Accelerating generative LLM serving with speculative inference and token tree verification. _arXiv preprint arXiv:2305.09781_, 2023. 
*   Patterson (2004) Patterson, D.A. Latency lags bandwith. _Communications of the ACM_, 47(10):71–75, 2004. 
*   PyTorch Labs (2023) PyTorch Labs. gpt-fast. [https://github.com/pytorch-labs/gpt-fast/](https://github.com/pytorch-labs/gpt-fast/), 2023. 
*   Sanh et al. (2020) Sanh, V., Wolf, T., and Rush, A. Movement pruning: Adaptive sparsity by fine-tuning. _Advances in Neural Information Processing Systems_, 33:20378–20389, 2020. 
*   Santilli et al. (2023) Santilli, A., Severino, S., Postolache, E., Maiorca, V., Mancusi, M., Marin, R., and Rodola, E. Accelerating transformer inference for translation via parallel decoding. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pp. 12336–12355, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.689. URL [https://aclanthology.org/2023.acl-long.689](https://aclanthology.org/2023.acl-long.689). 
*   Shazeer (2019) Shazeer, N. Fast transformer decoding: One write-head is all you need. _arXiv preprint arXiv:1911.02150_, 2019. 
*   Shen et al. (2020) Shen, S., Dong, Z., Ye, J., Ma, L., Yao, Z., Gholami, A., Mahoney, M.W., and Keutzer, K. Q-bert: Hessian based ultra low precision quantization of bert. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 34, pp. 8815–8821, 2020. 
*   Spector & Re (2023) Spector, B. and Re, C. Accelerating LLM inference with staged speculative decoding. _arXiv preprint arXiv:2308.04623_, 2023. 
*   Stern et al. (2018) Stern, M., Shazeer, N., and Uszkoreit, J. Blockwise parallel decoding for deep autoregressive models. _Advances in Neural Information Processing Systems_, 31, 2018. 
*   Sun et al. (2021) Sun, X., Ge, T., Wei, F., and Wang, H. Instantaneous grammatical error correction with shallow aggressive decoding. _arXiv preprint arXiv:2106.04970_, 2021. 
*   Taori et al. (2023) Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T.B. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023. 
*   Touvron et al. (2023) Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. LlAMA 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023. 
*   Voita et al. (2019) Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. _arXiv preprint arXiv:1905.09418_, 2019. 
*   Wu et al. (2020) Wu, Z., Liu, Z., Lin, J., Lin, Y., and Han, S. Lite transformer with long-short range attention. _arXiv preprint arXiv:2004.11886_, 2020. 
*   Xia et al. (2023) Xia, H., Ge, T., Wang, P., Chen, S.-Q., Wei, F., and Sui, Z. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In _Findings of the Association for Computational Linguistics: EMNLP 2023_, pp. 3909–3925, 2023. 
*   Yang et al. (2023a) Yang, N., Ge, T., Wang, L., Jiao, B., Jiang, D., Yang, L., Majumder, R., and Wei, F. Inference with reference: Lossless acceleration of large language models. _arXiv preprint arXiv:2304.04487_, 2023a. 
*   Yang et al. (2023b) Yang, S., Lee, G., Cho, J., Papailiopoulos, D., and Lee, K. Predictive pipelined decoding: A compute-latency trade-off for exact llm decoding. _arXiv preprint arXiv:2307.05908_, 2023b. 
*   Zadeh et al. (2020) Zadeh, A.H., Edo, I., Awad, O.M., and Moshovos, A. Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference. In _2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_, pp. 811–824. IEEE, 2020. 
*   Zafrir et al. (2019) Zafrir, O., Boudoukh, G., Izsak, P., and Wasserblat, M. Q8bert: Quantized 8bit bert. In _2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS)_, pp. 36–39. IEEE, 2019. 
*   Zhang et al. (2023) Zhang, J., Wang, J., Li, H., Shou, L., Chen, K., Chen, G., and Mehrotra, S. Draft & verify: Lossless large language model acceleration via self-speculative decoding. _arXiv preprint arXiv:2309.08168_, 2023. 
*   Zhang et al. (2024) Zhang, P., Zeng, G., Wang, T., and Lu, W. TinyLlama: An open-source small language model. _arXiv preprint arXiv:2401.02385_, 2024. 
*   Zheng et al. (2023) Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _arXiv preprint arXiv:2306.05685_, 2023. 
*   Zhou et al. (2023) Zhou, Y., Lyu, K., Rawat, A.S., Menon, A.K., Rostamizadeh, A., Kumar, S., Kagy, J.-F., and Agarwal, R. DistillSpec: Improving speculative decoding via knowledge distillation. _arXiv preprint arXiv:2310.08461_, 2023. 

Appendix A Implementation Details
---------------------------------

### A.1 Tree Structure

Utilizing tree attention, EAGLE generates a tree-structured draft. The left side of Figure [9](https://arxiv.org/html/2401.15077v3#A1.F9 "Figure 9 ‣ A.1 Tree Structure ‣ Appendix A Implementation Details ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") illustrates the tree structure of the draft, while the right side depicts the corresponding chain-structured draft when tree attention is not used (as utilized in the ablation study detailed in Section [4.3.1](https://arxiv.org/html/2401.15077v3#S4.SS3.SSS1 "4.3.1 Tree attention ‣ 4.3 Ablation study ‣ 4 Experiments ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")). In a greedy setting, we select the top k 𝑘 k italic_k tokens with the highest probabilities as child nodes. In a non-greedy setting, we sample k 𝑘 k italic_k tokens. The number of child nodes, k 𝑘 k italic_k, can be inferred from Figure 9; for instance, k=4 𝑘 4 k=4 italic_k = 4 at the root node. Regardless of employing a tree-structured or chain-structured draft, the draft model undergoes 5 forward passes during the draft phase. During the verification phase, each token’s probability is obtained through a single forward pass by the target LLM.

Why do we use such a tree structure? The choice of the tree structure, as depicted in Figure 9, was not rigorously optimized but rather based on intuition: branches of higher-probability tokens should be deeper and wider. For this paper, all models across all experiments utilized the draft structure shown in Figure 9. However, the optimal tree structure is likely context-dependent. For instance, as batch size increases and redundant computational resources decrease, a smaller tree might be preferable. Tuning the draft structure could potentially lead to improved performance.

![Image 9: Refer to caption](https://arxiv.org/html/2401.15077v3/x9.png)

Figure 9: Structure of EAGLE’s draft. The left side shows the draft structure when tree attention is employed, while the right side depicts the draft structure without the use of tree attention.

### A.2 Multi-Round Speculative Sampling

Unlike the chain-structured draft of speculative sampling, EAGLE employs a tree-structured draft, necessitating modifications to the sampling algorithm. The sampling algorithm A of speculative sampling can be briefly described as: if a token is accepted, it returns that token; otherwise, it samples a token from the adjusted distribution. For a tree-structured draft with k 𝑘 k italic_k candidate tokens, Multi-round speculative sampling recursively invokes algorithm A. Instead of directly sampling from the adjusted distribution after rejecting a token, Multi-round speculative sampling calls A again. If all tokens are rejected, it then directly samples from the adjusted distribution. The pseudocode for Multi-round speculative sampling is provided in Algorithm [1](https://arxiv.org/html/2401.15077v3#alg1 "Algorithm 1 ‣ A.2 Multi-Round Speculative Sampling ‣ Appendix A Implementation Details ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty").

Algorithm 1 Multi-round speculative sampling

Input: Target distribution

p 𝑝 p italic_p
, samples

t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
and distributions

p i^^subscript 𝑝 𝑖\hat{p_{i}}over^ start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG
for each

i 𝑖 i italic_i
from 1 to

k 𝑘 k italic_k
, where

t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
is sampled from

p i^^subscript 𝑝 𝑖\hat{p_{i}}over^ start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG
,

Output: a sample

x∼p similar-to 𝑥 𝑝 x\sim p italic_x ∼ italic_p
;

i←1←𝑖 1 i\leftarrow 1 italic_i ← 1

for

i≤k 𝑖 𝑘 i\leq k italic_i ≤ italic_k
do

r←U⁢(0,1)←𝑟 𝑈 0 1 r\leftarrow U(0,1)italic_r ← italic_U ( 0 , 1 )

if

r<p⁢(t i)/p i^⁢(t i)𝑟 𝑝 subscript 𝑡 𝑖^subscript 𝑝 𝑖 subscript 𝑡 𝑖 r<{p(t_{i})}/{\hat{p_{i}}(t_{i})}italic_r < italic_p ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) / over^ start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )
then

Return

t i subscript 𝑡 𝑖 t_{i}italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

end if

p←n⁢o⁢r⁢m⁢(m⁢a⁢x⁢(0,p⁢(t i)−p i^⁢(t i)))←𝑝 𝑛 𝑜 𝑟 𝑚 𝑚 𝑎 𝑥 0 𝑝 subscript 𝑡 𝑖^subscript 𝑝 𝑖 subscript 𝑡 𝑖 p\leftarrow norm(max(0,p(t_{i})-\hat{p_{i}}(t_{i})))italic_p ← italic_n italic_o italic_r italic_m ( italic_m italic_a italic_x ( 0 , italic_p ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) - over^ start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ) )

i←i+1←𝑖 𝑖 1 i\leftarrow i+1 italic_i ← italic_i + 1

end for

Sample

t∼p similar-to 𝑡 𝑝 t\sim p italic_t ∼ italic_p

Return

t 𝑡 t italic_t

Appendix B Detailed experimental results
----------------------------------------

Table [8](https://arxiv.org/html/2401.15077v3#A2.T8 "Table 8 ‣ Appendix B Detailed experimental results ‣ EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") displays the speedup ratio, average acceptance length τ 𝜏\tau italic_τ and acceptance rate α 𝛼\alpha italic_α of EAGLE on HumanEval, GSM8K, and Alpaca datasets.

Table 8: Speedup ratio, average acceptance length τ 𝜏\tau italic_τ and acceptance rate α 𝛼\alpha italic_α on HumanEval, GSM8K, and Alpaca at temperature = 0.

