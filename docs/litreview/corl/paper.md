Co-Reinforcement Learning for Unified Multimodal Understanding and Generation

1 Introduction

2 Related Work

3 Methodology

3.1 Preliminary

3.2 Pilot Exploration

3.3 Co-Reinforcement Learning

3.3.1 Verifiable Reward for Multimodal modeling

3.3.2 Unified Reinforcement Learning for Synergistic Multimodal Modeling

3.3.3 Refined Reinforcement Learning for Task-specific Enhancement

4 Experiment

4.1 Experimental Setups

4.2 Quantitative Results

4.3 Qualitative Results

4.4 Ablation Studies

5 Conclusion

A Appendix

A.1 Training Data

A.2 Supplementary Experimental Setups

Co-Reinforcement Learning for

Unified Multimodal Understanding and Generation

Jingjing Jiang1,2, Chongjie Si2, Jun Luo1, Hanwang Zhang1, Chao Ma2

1Nanyang Technological University, 2Shanghai Jiao Tong University

{jingjing.jiang,junluo,hanwangzhang}@ntu.edu.sg

{chongjiesi,chaoma}@sjtu.edu.cn

Abstract

This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement.
With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs.

1 Introduction

As large foundation models (LFMs) continue to advance in general capabilities and breadth of knowledge, post-training [27, 67, 101] has emerged as a critical paradigm for further refining pretrained LFMs toward specialized applications, facilitating task adaptation and human-aligned behaviors. Recently, reinforcement learning (RL)-based approaches [47, 48, 66, 88, 64, 55, 60] have exhibited considerable promise due to their data efficiency and strong alignment abilities. A notable exemplar is DeepSeek-R1 [18], which demonstrates that RL with verifiable rewards and the group relative policy optimization (GRPO) algorithm constitutes a practical and stable strategy that sidesteps explicit preference modeling [70] and reward model learning [76].
This promising paradigm indicates the significant potential of LFMs to acquire advanced capabilities and generalize effectively without dependence on large-scale, high-quality supervised data.

In the multimodal AI research community, the prevailing implementation [37, 22, 8, 49, 38, 94, 93, 64] of the GRPO algorithm centers on crafting diverse rule-based reward mechanisms to incentivize long-chain reasoning capabilities of multimodal large language models (MLLMs). These initiatives primarily target multimodal understanding, with particular focus on visual and mathematical reasoning tasks. Conversely, its application to visual generation remains surprisingly limited, with only pioneering explorations [72, 23] suggesting its feasibility. More importantly, extending GRPO to unified MLLMs (ULMs) [7, 83, 81, 36] capable of concurrently performing visual understanding and generation tasks remains considerably under-explored. Intuitively, ULMs could significantly benefit from GRPO owing to their inherent advantages of cross-task synergy and LLM sharing, which enables ULMs to share reward signals across various tasks and effectively mitigate reward imbalance, particularly when GRPO operates by jointly ranking outputs within task-agnostic groups.

This paper aims to enhance the understanding and generation capabilities of ULMs without dependence on supervised data. We begin with pilot experiments to explore efficient reinforcement learning paradigms. Specifically, we systematically examine four rule-based training strategies:
(i) separate RL for individual tasks,
(ii) separate RL with weight merging,
(iii) cycle RL alternating between tasks,
and (iv) unified RL with joint optimization.
Our explorations reveal two critical findings.
First, direct task-specific RL fails to achieve the anticipated improvements, particularly in visual generation, and even impairs other abilities.
Second, compared with alternative strategies, unified RL showcases comprehensive advantages across tasks.
These results demonstrate the synergistic co-evolution of dual capabilities under a shared policy optimization paradigm.

In light of our preliminary findings, we propose CoRL, a co-evolutionary reinforcement learning framework designed to synergistically improve the understanding and generation capabilities of ULMs. CoRL follows a two-stage RL procedure: a unified RL stage for joint optimization of dual capabilities and a refined RL stage for task-specific enhancement.
In the first stage, the policy ULM is optimized through a unified GRPO algorithm with diverse rewards on a carefully curated dataset spanning both understanding and generation tasks. To effectively guide policy optimization in visual generation, we introduce a bidirectional cycle consistency reward and a text-image matching reward, which together promote semantic consistency and faithfulness of synthesized images to their corresponding prompts. The designed rewards complement typical multimodal understanding rewards (i.e., accuracy and format) within a unified group, enabling cross-task joint optimization.
In the subsequent stage, we independently reinforce the policy’s understanding and generation capabilities using respective rewards and tailored datasets for task-specific refinement.

Applying the two-stage CoRL training to the baseline ULM Janus-Pro [7] yields ULM-R1, a unified model with reinforced capabilities in both understanding and generation. To comprehensively assess its performance, we conduct extensive comparisons against state-of-the-art unified MLLMs and dedicated models across three visual generation and nine multimodal understanding benchmarks. Notably, ULM-R1 achieves substantial gains over its baseline on complex mathematical and logical reasoning tasks, such as WeMath (+15.2) and LogicVista (+10.6).
These results underscore the effectiveness of CoRL, providing compelling empirical evidence for the efficacy of RL in simultaneously advancing visual understanding and generation tasks.

We summarize our main contributions as follows:

•

We establish that RL with verifiable rewards and GRPO constitutes a data-efficient paradigm for cross-task co-optimization and capability enhancement.

•

We introduce a co-evolutionary reinforcement learning framework, CoRL, to synergistically enhance the dual capabilities of ULMs using a unified-then-refined RL paradigm.

•

We demonstrate the effectiveness of CoRL and the advantage of ULM-R1 through extensive qualitative and quantitative experiments across diverse benchmarks.

2 Related Work

Unified Multimodal Understanding and Generation.
Recent advancements [80, 7, 36, 81, 79, 65, 83, 100, 84, 75, 62, 69] have witnessed increasing attention to jointly model multimodal understanding and visual generation within a unified model. Pioneering attempts [16, 11] predominantly rely on continuous diffusion models, integrating external diffusion decoders for image synthesis.
Inspired by autoregressive next-token prediction, a growing line of research [65, 80, 75, 78, 7, 53, 36, 81, 79, 28, 43] encode visual inputs into discrete tokens and generate images in a fully autoregressive (F-AR) manner.
Specifically, this approach employs a vector quantized (VQ) tokenizer [13, 86] to convert images into discrete tokens, analogous to text tokenization. To mitigate information loss in VQ discretization, another stream of work [83, 69, 21, 26, 15, 44, 85] explores autoregressive and diffusion (AR-Diff) hybrid modeling approaches.
Architecturally, these models typically comprise a vision autoencoder, a text tokenizer, and an LLM. This work builds upon the F-AR model to develop an effective reinforcement learning framework.

RL-based Post-Training for MLLMs.
Post-training [101] aims to further enhance the performance of pretrained models for customized applications and user needs. Recently, RL [77, 71] has emerged as a powerful post-training technique, enabling models to learn from feedback and align with human values. RL in MLLMs can be broadly categorized into two paradigms: (1) RL from human/AI feedback (RLHF) [75, 85, 76, 63, 88, 89, 102, 31, 99, 50, 70, 57, 74, 92] and (2) RL with verifiable reward mechanisms [72, 93, 64, 38, 37, 32].
RLHF involves learning reward models from preference data before RL optimization, whereas the latter directly optimizes models using task-specific reward functions, bypassing explicit preference modeling. For example, DPO [55] is a notable implementation of RLHF and has been adopted by Emu3 [75] and HermesFlow [85] to narrow the performance gap between understanding and generation. In contrast, GRPO [60] exemplifies the second paradigm, simplifying reward formulation via group-wise relative advantage estimation.
Our work also falls into this paradigm but diverges from prior work such as SimpleAR [72], which utilizes GRPO with external CLIP reward for autoregressive visual generation, and R1-like MLLMs [93, 22, 64, 38] that focus on incentivizing reasoning capabilities.
First, our work demonstrates the significant potential of RL in co-optimizing understanding and generation, thereby broadening its applicability beyond reasoning. Moreover, we identify semantic consistency rewards and a co-evolutionary reinforcement strategy as crucial components in enhancing ULMs.

3 Methodology

3.1 Preliminary

Group relative policy optimization (GRPO) [60] is a value-free policy optimization algorithm with improved training stability and sample efficiency. Building upon PPO [58], GRPO introduces a group-wise relative advantage approach to bound policy updates while maintaining optimization flexibility. Let π𝜽subscript𝜋𝜽\pi_{\bm{\theta}}italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT denote a policy parameterized by 𝜽𝜽\bm{\theta}bold_italic_θ. Formally, given an input content c𝑐citalic_c, the algorithm first samples a group of G𝐺Gitalic_G outputs {o1,o2,…,oG}subscript𝑜1subscript𝑜2…subscript𝑜𝐺\{o_{1},o_{2},\dots,o_{G}\}{ italic_o start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_o start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_o start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT } from the current policy π𝜽oldsubscript𝜋subscript𝜽old\pi_{\bm{\theta}_{\mathrm{old}}}italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT roman_old end_POSTSUBSCRIPT end_POSTSUBSCRIPT. Each output is then evaluated using predefined, verifiable reward functions, yielding the reward set {r1,r2,…,rG}subscript𝑟1subscript𝑟2…subscript𝑟𝐺\{r_{1},r_{2},\dots,r_{G}\}{ italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT }. These rewards are subsequently normalized to compute group-relative advantages as follows:

Ai=ri−mean⁢({r1,r2,…,rG})std⁢({r1,r2,…,rG})⁢.subscript𝐴𝑖
subscript𝑟𝑖meansubscript𝑟1subscript𝑟2…subscript𝑟𝐺stdsubscript𝑟1subscript𝑟2…subscript𝑟𝐺.A_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{1},r_{2},\dots,r_{G}\})}{\mathrm{std}(\{r%
_{1},r_{2},\dots,r_{G}\})}\text{.}italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = divide start_ARG italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - roman_mean ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT } ) end_ARG start_ARG roman_std ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT } ) end_ARG .

(1)

After obtaining the advantage set {A1,A2,…,AG}subscript𝐴1subscript𝐴2…subscript𝐴𝐺\{A_{1},A_{2},\dots,A_{G}\}{ italic_A start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_A start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_A start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT } via group relative advantage estimation, the policy π𝜽subscript𝜋𝜽\pi_{\bm{\theta}}italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT is optimized by maximizing the following objective:

ℒ⁢(𝜽)=𝔼{oi}i=1G∼π𝜽old⁢1G⁢∑i=1G[π𝜽⁢(oi)π𝜽old⁢(oi)⁢Ai−β⁢𝔻KL⁢(π𝜽∥πref)]⁢,ℒ𝜽subscript𝔼similar-tosuperscriptsubscriptsubscript𝑜𝑖𝑖1𝐺subscript𝜋subscript𝜽old
1𝐺superscriptsubscript𝑖1𝐺delimited-[]
subscript𝜋𝜽subscript𝑜𝑖subscript𝜋subscript𝜽oldsubscript𝑜𝑖subscript𝐴𝑖𝛽subscript𝔻KLconditionalsubscript𝜋𝜽subscript𝜋ref,\mathcal{L}(\bm{\theta})=\mathbb{E}_{\{o_{i}\}_{i=1}^{G}\sim\pi_{\bm{\theta}_{%
\text{old}}}}\,\frac{1}{G}\sum_{i=1}^{G}\left[\frac{\pi_{\bm{\theta}}(o_{i})}{%
\pi_{\bm{\theta}_{\text{old}}}(o_{i})}A_{i}-\beta\,\mathbb{D}_{\mathrm{KL}}%
\left(\pi_{\bm{\theta}}\,\|\,\pi_{\mathrm{ref}}\right)\right]\text{,}caligraphic_L ( bold_italic_θ ) = blackboard_E start_POSTSUBSCRIPT { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_G end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT [ divide start_ARG italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_ARG italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_β blackboard_D start_POSTSUBSCRIPT roman_KL end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT ∥ italic_π start_POSTSUBSCRIPT roman_ref end_POSTSUBSCRIPT ) ] ,

(2)

where 𝔻KLsubscript𝔻KL\mathbb{D}_{\mathrm{KL}}blackboard_D start_POSTSUBSCRIPT roman_KL end_POSTSUBSCRIPT denotes the KL-divergence used to constrain the deviation between π𝜽subscript𝜋𝜽\pi_{\bm{\theta}}italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT and its reference policy πrefsubscript𝜋ref\pi_{\mathrm{ref}}italic_π start_POSTSUBSCRIPT roman_ref end_POSTSUBSCRIPT, and β𝛽\betaitalic_β is a regularization coefficient.

3.2 Pilot Exploration

Figure 1:
Results of different RL paradigms.
Janus-Pro-1B [7] serves as the baseline.

Given the exceptional performance and data efficiency of DeepSeek-R1-Zero [18], we explore the potential of ULMs to enhance understanding and generation capabilities without dependence on task-specific supervised fine-tuning. To accomplish this, we curate a dataset111https://huggingface.co/datasets/mm-vl/x2x_rft_16k comprising 16K samples sourced from the COCO 2017 train split [35]. Each sample includes a real image, an associated caption as a textual prompt for visual generation, and a corresponding QA pair for the multimodal understanding task. We adopt CLIP Score [54] as the verifiable reward for image generation, and a combination of formatting correctness and answer accuracy as the reward for text generation. We investigate four distinct RL paradigms:
(i) separate RL, where understanding and generation tasks are independently optimized under their respective reward mechanisms;
(ii) separate RL followed by weight merging, where each task is separately optimized, and the resulting weights are subsequently merged using a Gaussian distribution-based merging strategy [61] to incorporate both abilities;
(iii) cycle RL, which applies a scheduled alternation between the two tasks throughout the training process;
and (iv) unified RL, in which both tasks are jointly optimized within a unified paradigm to promote co-evolution.

As presented in Figure 1, we observe that (1) direct task-specific RL fails to achieve the expected improvements for ULMs, particularly in the visual generation task, and may even impair performance on the other task; and (2) unified RL demonstrates substantial advantages over alternative paradigms.
These findings indicate that the dual capabilities co-evolving within a shared training framework contribute to more effective cross-task synergy and knowledge transfer.

Figure 2: Overview of CoRL, a co-evolutionary reinforcement learning framework to jointly improve the dual capabilities of ULMs. CoRL adopts a two-stage RL procedure, comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement.

3.3 Co-Reinforcement Learning

3.3.1 Verifiable Reward for Multimodal modeling

In this section, we develop a suite of verifiable rewards for multimodal modeling, which provide clear and objective feedback to steer ULMs toward generating high-quality image and text outputs.

Bidirectional Cycle Consistency Reward in Text-to-Image Generation.
To encourage ULMs to generate images that faithfully reflect the concepts and entities described in the input prompt, we introduce a bidirectional cycle consistency reward ℛcyclesubscriptℛcycle\mathcal{R}_{\text{cycle}}caligraphic_R start_POSTSUBSCRIPT cycle end_POSTSUBSCRIPT, which measures the consistency between predictions and ground truth in both visual and textual spaces.
For visual consistency, we adopt LPIPS [97] to assess the patch-level perceptual similarity between the real image ℐrealsubscriptℐreal\mathcal{I}_{\text{real}}caligraphic_I start_POSTSUBSCRIPT real end_POSTSUBSCRIPT and the synthesized image ℐgensubscriptℐgen\mathcal{I}_{\text{gen}}caligraphic_I start_POSTSUBSCRIPT gen end_POSTSUBSCRIPT.
Textual consistency is implemented in a re-captioning manner. Specifically, we first employ BLIP [29] to generate a caption 𝒞re-capsubscript𝒞re-cap\mathcal{C}_{\text{re-cap}}caligraphic_C start_POSTSUBSCRIPT re-cap end_POSTSUBSCRIPT for each synthesized image, and then compute the SPICE [1] score between 𝒞re-capsubscript𝒞re-cap\mathcal{C}_{\text{re-cap}}caligraphic_C start_POSTSUBSCRIPT re-cap end_POSTSUBSCRIPT and its original prompt 𝒫orgsubscript𝒫org\mathcal{P}_{\text{org}}caligraphic_P start_POSTSUBSCRIPT org end_POSTSUBSCRIPT to measure semantic fidelity.
The combined bidirectional cycle reward is defined as:

ℛcycle=1−LPIPS⁢(ℐreal,ℐgen)+SPICE⁢(𝒫org,𝒞re-cap)⁢.subscriptℛcycle
1LPIPSsubscriptℐrealsubscriptℐgenSPICEsubscript𝒫orgsubscript𝒞re-cap.\mathcal{R}_{\text{cycle}}=1-\text{LPIPS}(\mathcal{I}_{\text{real}},\,\mathcal%
{I}_{\text{gen}})+\text{SPICE}(\mathcal{P}_{\text{org}},\,\mathcal{C}_{\text{%
re-cap}})\text{.}caligraphic_R start_POSTSUBSCRIPT cycle end_POSTSUBSCRIPT = 1 - LPIPS ( caligraphic_I start_POSTSUBSCRIPT real end_POSTSUBSCRIPT , caligraphic_I start_POSTSUBSCRIPT gen end_POSTSUBSCRIPT ) + SPICE ( caligraphic_P start_POSTSUBSCRIPT org end_POSTSUBSCRIPT , caligraphic_C start_POSTSUBSCRIPT re-cap end_POSTSUBSCRIPT ) .

(3)

This bidirectional reward forms a closed feedback loop that promotes mutual consistency between text and image, effectively penalizing hallucinated content and reinforcing prompt-adherent visual generation by simultaneously optimizing for both visual and textual consistency.

Text-Image Matching Reward.
While CLIP Score [54] provides a holistic measure of text-image alignment, it underperforms in Sec. 3.2 due to its limited ability to assess fine-grained semantics. To address this limitation, we instead propose a text-image matching reward ℛTIMsubscriptℛTIM\mathcal{R}_{\text{TIM}}caligraphic_R start_POSTSUBSCRIPT TIM end_POSTSUBSCRIPT, which leverages the ULM itself to assess cross-modal alignment at the token level.
Given a textual representation 𝑻={𝒕1,𝒕2,…,𝒕Lt}∈ℝLt×d𝑻subscript𝒕1subscript𝒕2…subscript𝒕subscript𝐿𝑡superscriptℝsubscript𝐿𝑡𝑑\bm{T}=\{\bm{t}_{1},\bm{t}_{2},\ldots,\bm{t}_{L_{t}}\}\in\mathbb{R}^{L_{t}%
\times d}bold_italic_T = { bold_italic_t start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_t start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , bold_italic_t start_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_POSTSUBSCRIPT } ∈ blackboard_R start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT × italic_d end_POSTSUPERSCRIPT of the prompt and the corresponding visual representation 𝑰={𝒊1,𝒊2,…,𝒊Li}∈ℝLi×d𝑰subscript𝒊1subscript𝒊2…subscript𝒊subscript𝐿𝑖superscriptℝsubscript𝐿𝑖𝑑\bm{I}=\{\bm{i}_{1},\bm{i}_{2},\ldots,\bm{i}_{L_{i}}\}\in\mathbb{R}^{L_{i}%
\times d}bold_italic_I = { bold_italic_i start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_i start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , bold_italic_i start_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT } ∈ blackboard_R start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT × italic_d end_POSTSUPERSCRIPT of a generated image, the reward is computed as:

ℛTIM=12⁢(1Li⁢∑j=1Limaxk∈[1,Lt]⁡cos⁡(𝒊j,𝒕k)+1Lt⁢∑k=1Ltmaxj∈[1,Li]⁡cos⁡(𝒕k,𝒊j))⁢,subscriptℛTIM
12

1subscript𝐿𝑖superscriptsubscript𝑗1subscript𝐿𝑖subscript𝑘1subscript𝐿𝑡subscript𝒊𝑗subscript𝒕𝑘
1subscript𝐿𝑡superscriptsubscript𝑘1subscript𝐿𝑡subscript𝑗1subscript𝐿𝑖subscript𝒕𝑘subscript𝒊𝑗,\mathcal{R}_{\text{TIM}}=\frac{1}{2}\left(\frac{1}{L_{i}}\sum_{j=1}^{L_{i}}%
\max_{k\in[1,L_{t}]}\cos(\bm{i}_{j},\bm{t}_{k})+\frac{1}{L_{t}}\sum_{k=1}^{L_{%
t}}\max_{j\in[1,L_{i}]}\cos(\bm{t}_{k},\bm{i}_{j})\right)\text{,}caligraphic_R start_POSTSUBSCRIPT TIM end_POSTSUBSCRIPT = divide start_ARG 1 end_ARG start_ARG 2 end_ARG ( divide start_ARG 1 end_ARG start_ARG italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUPERSCRIPT roman_max start_POSTSUBSCRIPT italic_k ∈ [ 1 , italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ] end_POSTSUBSCRIPT roman_cos ( bold_italic_i start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , bold_italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) + divide start_ARG 1 end_ARG start_ARG italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_POSTSUPERSCRIPT roman_max start_POSTSUBSCRIPT italic_j ∈ [ 1 , italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] end_POSTSUBSCRIPT roman_cos ( bold_italic_t start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , bold_italic_i start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) ) ,

(4)

where Ltsubscript𝐿𝑡L_{t}italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT and Lisubscript𝐿𝑖L_{i}italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT are the lengths of textual and visual token sequences, and d𝑑ditalic_d is the embedding dimension. This reward captures the fine-grained correspondence between textual concepts and visual elements through maximum cosine similarity, ensuring that each visual token aligns with its most relevant textual counterpart and vice versa.

Accuracy Reward in Multimodal Question Answering.
Accuracy rewards leverage task-specific metrics to directly evaluate the correctness of ULM predictions. We consider two accuracy rewards tailored to different question types: ℛMCQ-AccsubscriptℛMCQ-Acc\mathcal{R}_{\text{MCQ-Acc}}caligraphic_R start_POSTSUBSCRIPT MCQ-Acc end_POSTSUBSCRIPT for multi-choice questions and ℛOE-AccsubscriptℛOE-Acc\mathcal{R}_{\text{OE-Acc}}caligraphic_R start_POSTSUBSCRIPT OE-Acc end_POSTSUBSCRIPT for open-ended questions. These rewards follow a binary evaluation mechanism, assigning a value of 1 when the predicted answer matches the ground truth and 0 otherwise.

Format Reward in Text Generation.
To encourage ULMs to generate structured and interpretable textual responses, we adopt the format reward [18], which requires the model to enclose its thinking process inside <think> ⋯⋯\cdots⋯ </think>, and provide its final answer within <answer> and <answer> tags. The format reward ℛFormatsubscriptℛFormat\mathcal{R}_{\text{Format}}caligraphic_R start_POSTSUBSCRIPT Format end_POSTSUBSCRIPT returns 1 for strict compliance, and 0 otherwise.

3.3.2 Unified Reinforcement Learning for Synergistic Multimodal Modeling

As illustrated in Figure 2, the policy ULM first undergoes unified reinforcement learning with diverse rewards across understanding and generation tasks. This unified process aims to jointly enhance its dual capabilities and establish a solid foundation for subsequent task-specific refinement.

Reward Function and Training Objective.
To ensure diversity and complementarity in reward signals for unified multimodal modeling, we formulate a joint reward function as

ℛUni-S1=ℛcycle+ℛTIM+λ⋅(ℛAcc+ℛFormat)⁢,subscriptℛUni-S1
subscriptℛcyclesubscriptℛTIM⋅𝜆
subscriptℛAccsubscriptℛFormat,\displaystyle\mathcal{R}_{\text{Uni-S1}}=\mathcal{R}_{\text{cycle}}+\mathcal{R%
}_{\text{TIM}}+\lambda\cdot(\mathcal{R}_{\text{Acc}}+\mathcal{R}_{\text{Format%
}})\text{,}caligraphic_R start_POSTSUBSCRIPT Uni-S1 end_POSTSUBSCRIPT = caligraphic_R start_POSTSUBSCRIPT cycle end_POSTSUBSCRIPT + caligraphic_R start_POSTSUBSCRIPT TIM end_POSTSUBSCRIPT + italic_λ ⋅ ( caligraphic_R start_POSTSUBSCRIPT Acc end_POSTSUBSCRIPT + caligraphic_R start_POSTSUBSCRIPT Format end_POSTSUBSCRIPT ) ,

(5)

where λ𝜆\lambdaitalic_λ is a coefficient that balances the two types of rewards.
During training, given an input prompt and an image-question pair, the policy model π𝜽oldsubscript𝜋subscript𝜽old\pi_{\bm{\theta}_{\mathrm{old}}}italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT roman_old end_POSTSUBSCRIPT end_POSTSUBSCRIPT first generates G𝐺Gitalic_G candidate responses, o={(ℐ1,𝒯1),(ℐ2,𝒯2),…,(ℐG,𝒯G)}𝑜subscriptℐ1subscript𝒯1subscriptℐ2subscript𝒯2…subscriptℐ𝐺subscript𝒯𝐺o=\{(\mathcal{I}_{1},\mathcal{T}_{1}),(\mathcal{I}_{2},\mathcal{T}_{2}),\dots,%
(\mathcal{I}_{G},\mathcal{T}_{G})\}italic_o = { ( caligraphic_I start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , caligraphic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ) , ( caligraphic_I start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , caligraphic_T start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) , … , ( caligraphic_I start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT , caligraphic_T start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT ) }, each comprising a synthesized image ℐℐ\mathcal{I}caligraphic_I and a CoT-format solution 𝒯𝒯\mathcal{T}caligraphic_T.
Concurrently, the joint reward function ℛUni-S1subscriptℛUni-S1\mathcal{R}_{\text{Uni-S1}}caligraphic_R start_POSTSUBSCRIPT Uni-S1 end_POSTSUBSCRIPT evaluates each candidate pair, yielding the reward set r={r1,r2,…,rG}𝑟subscript𝑟1subscript𝑟2…subscript𝑟𝐺r=\{r_{1},r_{2},\dots,r_{G}\}italic_r = { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT }. These rewards are subsequently normalized according to Eq. 1 to compute the corresponding group-relative advantages A={A1,A2,…,AG}𝐴subscript𝐴1subscript𝐴2…subscript𝐴𝐺A=\{A_{1},A_{2},\dots,A_{G}\}italic_A = { italic_A start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_A start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_A start_POSTSUBSCRIPT italic_G end_POSTSUBSCRIPT }. The new policy model π𝜽subscript𝜋𝜽\pi_{\bm{\theta}}italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT is then updated by maximizing the following GRPO-based objective:

ℒS1=𝔼{oi}i=1G∼π𝜽old⁢1G⁢∑i=1Gπ𝜽⁢(oi)π𝜽old⁢(oi)⁢Ai⁢, where ⁢oi=(ℐi,𝒯i)⁢.subscriptℒS1subscript𝔼similar-tosuperscriptsubscriptsubscript𝑜𝑖𝑖1𝐺subscript𝜋subscript𝜽old
1𝐺superscriptsubscript𝑖1𝐺
subscript𝜋𝜽subscript𝑜𝑖subscript𝜋subscript𝜽oldsubscript𝑜𝑖subscript𝐴𝑖, where subscript𝑜𝑖subscriptℐ𝑖subscript𝒯𝑖.\mathcal{L}_{\text{S1}}=\mathbb{E}_{\{o_{i}\}_{i=1}^{G}\sim\pi_{\bm{\theta}_{%
\mathrm{old}}}}\,\frac{1}{G}\sum_{i=1}^{G}\frac{\pi_{\bm{\theta}}(o_{i})}{\pi_%
{\bm{\theta}_{\mathrm{old}}}(o_{i})}A_{i}\,\text{, where }o_{i}=(\mathcal{I}_{%
i},\mathcal{T}_{i})\text{.}caligraphic_L start_POSTSUBSCRIPT S1 end_POSTSUBSCRIPT = blackboard_E start_POSTSUBSCRIPT { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT roman_old end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_G end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT divide start_ARG italic_π start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT bold_italic_θ start_POSTSUBSCRIPT roman_old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_ARG italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , where italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = ( caligraphic_I start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , caligraphic_T start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) .

(6)

Notably, based on empirical findings from recent work [87], we omit the KL-divergence constraint during this stage to improve both optimization efficiency and generalization capability.

Training Data.
To support unified RL for synergistic multimodal modeling, we curate a comprehensive dataset comprising 22K samples222https://huggingface.co/datasets/mm-vl/x2x_rft_22k, which follows the data structure established in Sec. 3.2. Each sample includes a real image, a prompt for visual generation, and a CoT-format QA pair for multimodal understanding. This balanced data composition facilitates joint optimization of dual capabilities within a unified framework, while preserving the granularity of task-specific supervision.

3.3.3 Refined Reinforcement Learning for Task-specific Enhancement

After completing unified RL, as shown in Figure 2, we apply a targeted learning strategy to further enhance the task-specific performance of the policy model. This second-stage optimization leverages task-specific rewards and tailored datasets for individual tasks.

Reward Function and Training Objective.
For text-to-image generation, the reward is defined as ℛT2I-S2=ℛcycle+ℛTIMsubscriptℛT2I-S2
subscriptℛcyclesubscriptℛTIM\mathcal{R}_{\text{T2I-S2}}=\mathcal{R}_{\text{cycle}}+\mathcal{R}_{\text{TIM}}caligraphic_R start_POSTSUBSCRIPT T2I-S2 end_POSTSUBSCRIPT = caligraphic_R start_POSTSUBSCRIPT cycle end_POSTSUBSCRIPT + caligraphic_R start_POSTSUBSCRIPT TIM end_POSTSUBSCRIPT. For multimodal understanding, we define two distinct reward formulations: (1) ℛMCQ-S2=ℛMCQ-Acc+ℛFormatsubscriptℛMCQ-S2
subscriptℛMCQ-AccsubscriptℛFormat\mathcal{R}_{\text{MCQ-S2}}=\mathcal{R}_{\text{MCQ-Acc}}+\mathcal{R}_{\text{%
Format}}caligraphic_R start_POSTSUBSCRIPT MCQ-S2 end_POSTSUBSCRIPT = caligraphic_R start_POSTSUBSCRIPT MCQ-Acc end_POSTSUBSCRIPT + caligraphic_R start_POSTSUBSCRIPT Format end_POSTSUBSCRIPT for multiple-choice questions, and (2) ℛOE-S2=ℛOE-Acc+ℛFormatsubscriptℛOE-S2
subscriptℛOE-AccsubscriptℛFormat\mathcal{R}_{\text{OE-S2}}=\mathcal{R}_{\text{OE-Acc}}+\mathcal{R}_{\text{%
Format}}caligraphic_R start_POSTSUBSCRIPT OE-S2 end_POSTSUBSCRIPT = caligraphic_R start_POSTSUBSCRIPT OE-Acc end_POSTSUBSCRIPT + caligraphic_R start_POSTSUBSCRIPT Format end_POSTSUBSCRIPT for open-ended questions.
The training objective in this stage adheres to the standard GRPO formulation in Eq. 2, with the appropriate task-specific reward (ℛT2I-S2subscriptℛT2I-S2\mathcal{R}_{\text{T2I-S2}}caligraphic_R start_POSTSUBSCRIPT T2I-S2 end_POSTSUBSCRIPT, ℛMCQ-S2subscriptℛMCQ-S2\mathcal{R}_{\text{MCQ-S2}}caligraphic_R start_POSTSUBSCRIPT MCQ-S2 end_POSTSUBSCRIPT, or ℛOE-S2subscriptℛOE-S2\mathcal{R}_{\text{OE-S2}}caligraphic_R start_POSTSUBSCRIPT OE-S2 end_POSTSUBSCRIPT) replacing Aisubscript𝐴𝑖A_{i}italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT depending on the task. To ensure stable optimization, we reintroduce the KL-divergence constraint at this stage to limit policy deviation from the reference distribution.

Training Data.
For text-to-image generation, we continue training on the curated dataset introduced in Sec. 3.2. For multimodal understanding, we utilize two specialized datasets: mcot_r1_mcq333https://huggingface.co/datasets/mm-vl/mcot_r1_mcq_66k for multiple-choice questions and mcot_r1_vqa444https://huggingface.co/datasets/mm-vl/mcot_r1_vqa_66k for open-ended questions.
These task-specific datasets enable the model to develop more refined and robust capabilities within each task domain.

4 Experiment

\mymidsize

Table 1: Results on text-to-image generation benchmarks. ♣♣{\color[rgb]{1,.5,0}\clubsuit}♣ and ♣♣{\color[rgb]{0.0,0.5,0.0}\clubsuit}♣ denote models trained using DPO and GRPO strategies. The best performance in each category is highlighted in bold.

Model
Scale
Res.
Type
GenEval ↑↑\uparrow↑
WISE ↑↑\uparrow↑
DPG ↑↑\uparrow↑

Two Obj.
Counting
Position
Color Attri.
Overall
Overall
Overall

▼▼\blacktriangledown▼ Generation Only

PixArt-α𝛼\alphaitalic_α [4]

0.6B
5122

Diff
0.50
0.44
0.08
0.07
0.48
0.47
71.11

SDv1.5 [56]

0.9B
5122

Diff
0.38
0.35
0.04
0.06
0.43
0.32
63.18

SDv2.1 [56]

0.9B
5122

Diff
0.51
0.44
0.07
0.17
0.50
0.32
68.09

SD3-Medium [14]

2B
5122

Diff
0.94
0.72
0.33
0.60
0.74
0.42
84.08

SDXL [51]

2.6B
10242

Diff
0.74
0.39
0.15
0.23
0.55
0.43
74.65

DALL·E 3 [3]

-
10242

Diff
0.87
0.47
0.43
0.45
0.67
-
83.50

LlamaGen [62]

0.8B
2562

F-AR
0.34
0.21
0.07
0.04
0.32
-
65.16

SimpleAR [72] ♣♣{\color[rgb]{0.0,0.5,0.0}\clubsuit}♣

1.5B
10242

F-AR
0.90
-
0.28
0.45
0.63
-
81.97

▼▼\blacktriangledown▼ Unified Understanding and Generation

TokenFlow [53]

8B
2562

F-AR
0.60
0.41
0.16
0.24
0.55
-
73.38

Emu3 [75]
8B
5122
F-AR
-
-
-
-
0.66
0.39
80.60

Emu3-DPO [75] ♣♣{\color[rgb]{1,.5,0}\clubsuit}♣
8B
5122
F-AR
-
-
-
-
0.64
-
81.60

LWM [36]

7B
5122

F-AR
0.41
0.46
0.09
0.15
0.47
-
-

Orthus [26]

7B
5122

AR-Diff
-
-
-
-
0.58
0.27
-

Janus-Pro [7]

7B
3842

F-AR
0.89
0.59
0.79
0.88
0.80
0.35
84.19

ILLUME+ [21]

3B
3842

AR-Diff
0.88
0.62
0.42
0.53
0.72
-
-

D-DiT [34]

2B
5122

Diff
0.80
0.54
0.32
0.50
0.65
-
-

Harmon [80]

1.5B
5122

F-AR
0.86
0.66
0.74
0.48
0.76
0.41
-

show-o [83]
1.3B
5122
AR-Diff
0.80
0.66
0.31
0.50
0.68
0.35
67.48

HermesFlow [85] ♣♣{\color[rgb]{1,.5,0}\clubsuit}♣
1.3B
5122
AR-Diff
0.84
0.66
0.32
0.52
0.69
-
70.22

Janus [78]

1.3B
3842

F-AR
0.68
0.30
0.46
0.42
0.61
0.23
79.68

Janus-Pro [7]
1.5B
3842
F-AR
0.82
0.51
0.65
0.56
0.73
0.26
82.63

ULM-R1 ♣♣{\color[rgb]{0.0,0.5,0.0}\clubsuit}♣
1.5B
3842
F-AR
0.85
0.71
0.68
0.80
0.77
0.33
83.92

4.1 Experimental Setups

Evaluation Benchmarks.
We evaluate visual generation capabilities on the GenEval [17], WISE [46], and DPG-Bench [20] benchmarks. GenEval employs an object-centric evaluation protocol to assess compositional and attribute-level alignment, while DPG-Bench adopts a VQA-based setting to evaluate dense prompt-following and semantic fidelity. WISE provides a holistic evaluation of models’ world knowledge, considering consistency, realism, and aesthetics.
We also evaluate multimodal understanding capabilities across diverse benchmarks. Specifically, MMStar [5], MMMU [91], and WeMath (MathWeWe{}^{\text{We}}start_FLOATSUPERSCRIPT We end_FLOATSUPERSCRIPT) [52] are used for multi-choice evaluation, while MMVet [90], POPE [33], and LogicVista (LogicVTVT{}^{\text{VT}}start_FLOATSUPERSCRIPT VT end_FLOATSUPERSCRIPT) [82] are used for open-ended evaluation. In addition, we employ MathVista (MathVTVT{}^{\text{VT}}start_FLOATSUPERSCRIPT VT end_FLOATSUPERSCRIPT) [40], MathVerse-Vision (MathVSVS{}^{\text{VS}}start_FLOATSUPERSCRIPT VS end_FLOATSUPERSCRIPT) [95], and MathVision (MathVisVis{}^{\text{Vis}}start_FLOATSUPERSCRIPT Vis end_FLOATSUPERSCRIPT) [73] to assess complex mathematical reasoning capabilities, covering both multi-choice and open-ended QA formats. On these benchmarks, we compute accuracy using the toolkit VLMEvalKit [12].

Implementation Details.
We develop ULM-R1 using Janus-Pro-1B [7] as the baseline ULM for unified multimodal understanding and generation. To ensure reproducibility and scalability, our RL training is built upon the trl [68] framework. In the unified RL stage, we employ the AdamW optimizer with an initial learning rate of 4e-6 and a batch size of 16. We sample 8 responses for both understanding and generation tasks, and set the reward balancing factor in Eq. 5 to 0.8.
In the refined RL stage, we sample 16 responses for both multimodal understanding and text-to-image generation tasks. Additionally, we reduce the learning rate to 1e-6 to facilitate fine-grained optimization.
All training is conducted on 8 NVIDIA H20 (96G) GPUs.
During inference, greedy decoding is used for text generation in multimodal understanding tasks. For text-to-image generation, we employ classifier-free guidance (CFG) [19] with a guidance weight set to 5.
More details on the training data and settings are provided in App. A.

\mymidsize

Table 2: Results on multimodal understanding benchmarks. The best performance within each category is highlighted in bold. † denotes results obtained from our evaluation.

Model
LLM
Multi-Choice (MC) ↑↑\uparrow↑
Open-Ended (OE) ↑↑\uparrow↑
MC&OE Mixed ↑↑\uparrow↑

MMMU
MMStar
MathWeWe{}^{\text{We}}start_FLOATSUPERSCRIPT We end_FLOATSUPERSCRIPT

MMVet
POPE
LogicVTVT{}^{\text{VT}}start_FLOATSUPERSCRIPT VT end_FLOATSUPERSCRIPT

MathVTVT{}^{\text{VT}}start_FLOATSUPERSCRIPT VT end_FLOATSUPERSCRIPT

MathVSVS{}^{\text{VS}}start_FLOATSUPERSCRIPT VS end_FLOATSUPERSCRIPT

MathVisVis{}^{\text{Vis}}start_FLOATSUPERSCRIPT Vis end_FLOATSUPERSCRIPT

▼▼\blacktriangledown▼ Understanding Only

SmolVLM [45]

SmolLM2-1.7B
38.8
41.7
9.1
33.8
85.5
28.0
43.6
12.6
12.8

SAIL-VL [10]

Qwen2.5-1.5B
44.1
56.5
14.6
44.2
88.1
30.4
62.8
17.4
17.3

Ovis2 [42]

Qwen2.5-1.5B
45.6
56.7
9.9
58.3
87.8
34.7
64.1
29.4
17.7

InternVL3 [103]

Qwen2.5-1.5B
48.7
61.1
22.9
67.0
90.1
34.7
57.6
24.5
20.2

Qwen2.5-VL [2]
Qwen2.5-3B
51.2
56.3
22.9
60.0
85.9
40.3
61.2
31.2
21.9

LMM-R1 [49]
Qwen2.5-3B
-
58.0
-
-
-
-
63.2
41.6
26.4

▼▼\blacktriangledown▼ Unified Understanding and Generation

ILLUME+ [21]

Qwen2.5-3B
44.3
-
-
40.3
87.6
-
-
-
-

Harmon [80]

Qwen2.5-1.5B
38.9
-
-
-
87.6
-
-
-
-

VILA-U [81]

LLaMA-2-7B
-
-
-
33.5
85.8
-
-
-
-

Orthus [26]

Chameleon-7B
28.2
-
-
-
79.6
-
-
-
-

UniToken [24]

Chameleon-7B
32.8
46.1
-
-
-
-
38.5
-

SGen-VL [28]

InternLM2-1.8B
34.2
-
-
34.5
85.3
-
42.7
-

Show-o [83]
Phi-1.3B
26.7
-
-
-
80.0
-
-
-
-

HermesFlow [85]
Phi-1.3B
28.3
-
-
-
81.4
-
-
-
-

Janus-Pro [7]

DeepSeek-LLM-7B
41.0
46.5
9.7
50.0
87.4
28.0
42.5
15.9
14.7

Janus [78]

DeepSeek-LLM-1.3B
30.5
37.6
3.4†

34.3
87.0
23.9†

33.7
14.9†

13.4†

Janus-Pro [7]
DeepSeek-LLM-1.5B
36.3
43.1†
5.9†
39.8
86.2
23.9†
37.3†
13.5†
13.4†

ULM-R1
DeepSeek-LLM-1.5B
42.3
47.6
21.1
43.9
88.9
34.5
42.5
25.4
22.0

4.2 Quantitative Results

Text-to-Image Generation.
Table 1 presents a comprehensive comparison between ULM-R1 and state-of-the-art models across three visual generation benchmarks.
Among unified models, our model ranks second on both GenEval and WISE benchmarks. Notably, it achieves balanced performance across diverse task categories within GenEval, with the best score of 0.71 in object counting. When compared with specialized generation-only models, ULM-R1 surpasses the top performer SD3-Medium [14] by a slight margin (0.77 vs. 0.74 on GenEval). Moreover, ULM-R1 shows consistent improvements over its base model across all benchmarks.
These results collectively demonstrate the effectiveness and advantage of our CoRL in enhancing visual generation quality.

Multimodal Understanding.
Results are shown in Table 2. We continue to apply the Gaussian distribution-based merging strategy [61] to combine the two task-specific policy models for mixed QA format evaluation.
Overall, ULM-R1 markedly outperforms existing unified models across most benchmarks, and substantially narrows the performance gap with leading understanding-only MLLMs of comparable model scale. More specifically, our model achieves state-of-the-art performance among unified models on MMStar (47.6), WeMath (21.1), LogicVista (34.5), and on several mixed-format math benchmarks, including MathVerse (25.4) and MathVision (22.0). Particularly, ULM-R1 demonstrates considerable improvements over its base model in mathematical and logical reasoning tasks, achieving gains of 15.2 on WeMath and 10.6 on LogicVista.
These results not only demonstrate the effectiveness of CoRL in enhancing ULMs’ understanding capabilities, but also establish that reinforcement learning provides a data-efficient pathway for achieving both robust generalization and sophisticated reasoning capabilities, without the need for large-scale supervised data.

\mymidsize

Table 3: Comparison between different RL paradigms for ULMs.
The cold SFT data is consist of x2x_rft_22k, mcot_r1_mcq (22K), and mcot_r1_vqa (22K). #7: CoRL.

#
Ablated Setting
Stage
GenEval
DPG
MMMU
MathWeWe{}^{\text{We}}start_FLOATSUPERSCRIPT We end_FLOATSUPERSCRIPT
MMVet
LogicVTVT{}^{\text{VT}}start_FLOATSUPERSCRIPT VT end_FLOATSUPERSCRIPT

0
Baseline
-
73.0
82.6
36.3
5.9
39.8
23.9

1
+ Cold-SFT
S1
72.8 (-0.3)
82.5 (-0.1)
41.0 (+4.7)
18.0 (+12.1)
42.0 (+2.2)
27.9 (+4.0)

2
+ Unified-RL
S1
75.9 (+2.9)
83.3 (+0.7)
40.3 (+4.0)
14.0 (+8.1)
42.5 (+2.7)
30.2 (+6.3)

3
+ Refined-RL (T2I)
S2
75.1 (+2.1)
83.0 (+0.4)
/
/
/
/

4
+ Refined-RL (MM2T-MC)
S2
/
/
39.6 (+3.3)
15.8 (+9.9)
/
/

5
+ Refined-RL (MM2T-OE)
S2
/
/
/
/
42.2 (+2.4)
29.5 (+5.6)

6
+ Refined-RL w/ Cold-SFT
S1&S2
74.5 (+1.5)
82.8 (+0.2)
41.8 (+5.5)
22.5 (+16.6)
43.7 (+3.9)
35.9 (+12.0)

7
+ Refined-RL w/ Unified-RL
S1&S2
77.3 (+4.3)
83.9 (+1.3)
42.3 (+6.0)
21.1 (+15.2)
43.9 (+4.1)
34.5 (+10.6)

Figure 3: Qualitative comparison of text-to-image generation between Janus-Pro and ULM-R1. The red box marks an exemplary failure case.

4.3 Qualitative Results

In this section, we first present a qualitative comparison between ULM-R1 and Janus-Pro on visual generation, as illustrated in Figure 3. The results clearly showcase that ULM-R1 achieves superior text-to-image alignment and object grounding across diverse prompts, with especially notable improvements in spatial object arrangement and compositional consistency.
Furthermore, as shown in Figure 4, we visualize several representative examples of multimodal understanding. Compared to Janus-Pro, ULM-R1 exhibits significantly enhanced understanding capabilities, particularly in mathematical reasoning.
These comprehensive qualitative results demonstrate the effectiveness of CoRL in simultaneously improving both visual generation and multimodal understanding in ULMs.

4.4 Ablation Studies

In this section, we primarily validate the effectiveness of our training strategy and designed rewards.

Comparison Between Various RL Paradigms.
As presented in Table 3, we conduct a comprehensive ablation study to evaluate the effects of different RL paradigms for ULMs.
The results reveal two key findings:
▶▶\blacktriangleright▶ #2 vs.  #1: Unified-RL effectively enhances both the generation and understanding capabilities of ULMs, whereas Cold-SFT has minimal impact on visual generation.
▶▶\blacktriangleright▶ #7 vs.  #6: Compared to the de facto paradigm, our CoRL consistently outperforms it on visual generation benchmarks while achieving comparable results on multimodal understanding benchmarks.
These findings indicate that unified RL provides a robust foundation for task-specific refinement, even without reliance on supervised data.
Additionally, CoRL consistently outperforms both its baseline and task-specific RL variants (#3-#5), achieving improvements of 2.1 points on GenEval (vs.  generation-only RL, #3) and 5.3 points on WeMath (vs.  understanding-only RL, #4).
These results demonstrate the efficacy of CoRL as our final RL paradigm.

\mymidsize

Table 4: Effect of visual generation rewards.

#
ℛcyclesubscriptℛcycle\mathcal{R}_{\mathrm{cycle}}caligraphic_R start_POSTSUBSCRIPT roman_cycle end_POSTSUBSCRIPT
ℛTIMsubscriptℛTIM\mathcal{R}_{\mathrm{TIM}}caligraphic_R start_POSTSUBSCRIPT roman_TIM end_POSTSUBSCRIPT
GenEval
DPG
Avg. ↑↑\uparrow↑

0

73.0
82.6
77.8

1
✓

76.2
83.5
79.9 (+2.1)

2

✓
74.1
83.0
78.6 (+0.8)

3
✓
✓
77.3
83.9
80.6 (+2.8)

Rewards in Text-to-Image Generation.
To assess the effectiveness of our proposed rewards for text-to-image generation, we conduct ablation experiments as detailed in Table 4. The results demonstrate that incorporating either reward individually improves performance over the baseline: ℛcyclesubscriptℛcycle\mathcal{R}_{\mathrm{cycle}}caligraphic_R start_POSTSUBSCRIPT roman_cycle end_POSTSUBSCRIPT yields an increase of 2.1 in average score, while ℛTIMsubscriptℛTIM\mathcal{R}_{\mathrm{TIM}}caligraphic_R start_POSTSUBSCRIPT roman_TIM end_POSTSUBSCRIPT results in an increase of 0.8. Notably, combining both rewards leads to the best overall performance, achieving an average score of 80.6.
These findings suggest a modest but complementary effect between ℛcyclesubscriptℛcycle\mathcal{R}_{\mathrm{cycle}}caligraphic_R start_POSTSUBSCRIPT roman_cycle end_POSTSUBSCRIPT and ℛTIMsubscriptℛTIM\mathcal{R}_{\mathrm{TIM}}caligraphic_R start_POSTSUBSCRIPT roman_TIM end_POSTSUBSCRIPT, enhancing their joint benefit in enhancing visual generation quality.

Figure 4: Qualitative comparison of multimodal understanding between Janus-Pro and ULM-R1. The red box marks an exemplary failure case.

5 Conclusion

In this work, we investigate how to jointly enhance the understanding and generation capabilities of ULMs, and propose a co-evolutionary RL framework (CoRL). Within the proposed CoRL, the policy model undergoes unified RL for joint optimization and refined RL for task-specific enhancement, yielding ULM-R1. Extensive evaluations across diverse understanding and generation benchmarks demonstrate the effectiveness of CoRL and the advantage of ULM-R1.

Limitation.
Despite the substantial improvements achieved, several limitations remain that warrant further investigation. First, a notable performance gap still exists between generation and understanding tasks of ULMs. Second, our rewards for multimodal understanding are relatively simple and primary.
These limitations highlight the need for more sophisticated RL designs that can further enhance understanding capabilities and narrow the performance gap. We hope our work provides valuable insights for future RL research in ULMs.

Appendix A Appendix

A.1 Training Data

Training Data for Unified Reinforcement Learning.
To support synergistic multimodal modeling during unified RL, we curate a dataset (i.e., x2x_rft_22k) that simultaneously involves text-to-image generation and multimodal understanding tasks.
As illustrated in Figure 5, each sample includes a real image, a prompt for generation, and a problem for understanding.
The real images are sourced from the COCO 2017 train split [35], while the problems and their corresponding solutions are adapted from A-OKVQA [59] and GPT-VQA [98].
In addition, prompts are selected from the original COCO captions based on their entity coverage with the problem solutions.

Figure 5: Illustration of training examples used in unified reinforcement learning.

Training Data for Refined Reinforcement Learning.
In this stage, we collect three specialized datasets for task-specific RL. For text-to-image generation, we continue constructing a dataset (i.e., x2x_rft_16k) with prompts derived from COCO captions. Moreover, we curate mcot_r1_mcq and mcot_r1_vqa for multiple-choice and open-ended multimodal understanding, respectively. These two datasets encompass a diverse range of multimodal tasks, including mathematical reasoning, science-problem solving, and visual commonsense reasoning, across multiple source datasets. Specifically, mcot_r1_mcq consists of A-OKVQA [59], M3CoT [6], SQA-IMG (train) [39], ArxivQA [30], TabMWP (MC) [41], and MAVIS-Instruct (MC) [96], while mcot_r1_vqa includes GeomVerse [25], R-CoT [9], TabMWP (OE) [41], and MAVIS-Instruct (OE) [96].

A.2 Supplementary Experimental Setups

Table 5 provides detailed hyperparameter settings for ULM-R1’s RL training.

\mymidsize

Table 5: Training hyperparameter setting.

Configuration
Unified RL
Refined RL (T2I)
Refined RL (MM2T-MC)
Refined RL (MM2T-OE)

Number of sampled outputs (G𝐺Gitalic_G)
8
16
16
16

Regularization coefficient of 𝔻KLsubscript𝔻KL\mathbb{D}_{\text{KL}}blackboard_D start_POSTSUBSCRIPT KL end_POSTSUBSCRIPT (β𝛽\betaitalic_β)
0
0.02
0.02
0.02

Max prompt length
1024
256
1024
1024

Max completion length
512
/
512
512

Batch size
16
16
32
32

Peak learning rate
4e-6
1e-6
1e-6
1e-6

Epoch
1
1
1
1

References

Anderson et al. [2016]

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould.

Spice: Semantic propositional image caption evaluation.

In ECCV, pages 382–398, 2016.

Bai et al. [2025]

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai
Dang, Peng Wang, Shijie Wang, Jun Tang, et al.

Qwen2. 5-vl technical report.

arXiv:2502.13923, 2025.

Betker et al. [2023]

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long
Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al.

Improving image generation with better captions, 2023.

URL https://cdn.openai.com/papers/dall-e-3.pdf.

Chen et al. [2023]

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao
Wang, James Kwok, Ping Luo, Huchuan Lu, et al.

Pixart-alpha: Fast training of diffusion transformer for
photorealistic text-to-image synthesis.

arXiv:2310.00426, 2023.

Chen et al. [2024a]

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong
Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al.

Are we on the right way for evaluating large vision-language models?

arXiv:2403.20330, 2024a.

Chen et al. [2024b]

Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che.

M3CoT: A novel benchmark for multi-domain multi-step multi-modal
chain-of-thought.

arXiv:2405.16473, 2024b.

Chen et al. [2025]

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie,
Xingkai Yu, and Chong Ruan.

Janus-pro: Unified multimodal understanding and generation with data
and model scaling.

arXiv:2501.17811, 2025.

Deng et al. [2025]

Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang.

Boosting the generalization and reasoning of vision language models
with curriculum reinforcement learning.

arXiv:2503.07065, 2025.

Deng et al. [2024]

Linger Deng, Yuliang Liu, Bohan Li, Dongliang Luo, Liang Wu, Chengquan Zhang,
Pengyuan Lyu, Ziyang Zhang, Gang Zhang, Errui Ding, et al.

R-cot: Reverse chain-of-thought problem generation for geometric
reasoning in large multimodal models.

arXiv:2410.17885, 2024.

Dong et al. [2025]

Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran.

Scalable vision language model training via high quality data
curation.

arXiv:2501.05952, 2025.

Dong et al. [2024]

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang
Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al.

Dreamllm: Synergistic multimodal comprehension and creation.

In ICLR, 2024.

Duan et al. [2024]

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi
Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al.

Vlmevalkit: An open-source toolkit for evaluating large
multi-modality models.

arXiv:2407.11691, 2024.

Esser et al. [2021]

Patrick Esser, Robin Rombach, and Bjorn Ommer.

Taming transformers for high-resolution image synthesis.

In CVPR, pages 12873–12883, 2021.

Esser et al. [2024]

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas
Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic
Boesel, et al.

Scaling rectified flow transformers for high-resolution image
synthesis.

In ICML, 2024.

Fan et al. [2025]

Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao,
Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al.

Unified autoregressive visual generation and understanding with
continuous tokens.

arXiv:2503.13436, 2025.

Ge et al. [2024]

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li,
Xiaohan Ding, and Ying Shan.

Seed-x: Multimodal models with unified multi-granularity
comprehension and generation.

arXiv:2404.14396, 2024.

Ghosh et al. [2023]

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt.

Geneval: An object-focused framework for evaluating text-to-image
alignment.

In NeurIPS, pages 52132–52152, 2023.

Guo et al. [2025]

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning.

arXiv:2501.12948, 2025.

Ho and Salimans [2022]

Jonathan Ho and Tim Salimans.

Classifier-free diffusion guidance.

arXiv:2207.12598, 2022.

Hu et al. [2024]

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu.

Ella: Equip diffusion models with llm for enhanced semantic
alignment.

arXiv:2403.05135, 2024.

Huang et al. [2025a]

Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua
Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al.

Illume+: Illuminating unified mllm with dual visual tokenization and
diffusion refinement.

arXiv:2504.01934, 2025a.

Huang et al. [2025b]

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe
Xu, Yao Hu, and Shaohui Lin.

Vision-r1: Incentivizing reasoning capability in multimodal large
language models.

arXiv:2503.06749, 2025b.

Jiang et al. [2025]

Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin
Yan, Pheng-Ann Heng, and Hongsheng Li.

T2i-r1: Reinforcing image generation with collaborative
semantic-level and token-level cot.

arXiv:2505.00703, 2025.

Jiao et al. [2025]

Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, and
Yu-Gang Jiang.

Unitoken: Harmonizing multimodal understanding and generation through
unified visual encoding.

arXiv:2504.04423, 2025.

Kazemi et al. [2023]

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu
Soricut.

Geomverse: A systematic evaluation of large models for geometric
reasoning.

arXiv:2312.12241, 2023.

Kou et al. [2024]

Siqi Kou, Jiachun Jin, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, and
Zhijie Deng.

Orthus: Autoregressive interleaved image-text generation with
modality-specific heads.

arXiv:2412.00127, 2024.

Kumar et al. [2025]

Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham
Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip HS Torr, Fahad Shahbaz
Khan, and Salman Khan.

Llm post-training: A deep dive into reasoning large language models.

arXiv:2502.21321, 2025.

Li et al. [2024a]

Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan
Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, et al.

Synergen-vl: Towards synergistic image understanding and generation
with vision experts and token folding.

arXiv:2412.09604, 2024a.

Li et al. [2023a]

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.

Blip-2: Bootstrapping language-image pre-training with frozen image
encoders and large language models.

In ICML, pages 19730–19742, 2023a.

Li et al. [2024b]

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and
Qi Liu.

Multimodal arxiv: A dataset for improving scientific comprehension of
large vision-language models.

arXiv:2403.00231, 2024b.

Li et al. [2024c]

Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng
Yang, Benyou Wang, Lingpeng Kong, and Qi Liu.

Vlfeedback: A large-scale ai feedback dataset for large
vision-language models alignment.

arXiv:2410.09421, 2024c.

Li et al. [2025]

Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang,
Yu Qiao, Yi Wang, and Limin Wang.

Videochat-r1: Enhancing spatio-temporal perception via reinforcement
fine-tuning.

arXiv:2504.06958, 2025.

Li et al. [2023b]

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen.

Evaluating object hallucination in large vision-language models.

arXiv:2305.10355, 2023b.

Li et al. [2024d]

Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie
Yang, and Peng Wang.

Dual diffusion for unified image generation and understanding.

arXiv:2501.00289, 2024d.

Lin et al. [2014]

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva
Ramanan, Piotr Dollár, and C Lawrence Zitnick.

Microsoft coco: Common objects in context.

In ECCV, pages 740–755, 2014.

Liu et al. [2025a]

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel.

World model on million-length video and language with ringattention.

In ICLR, 2025a.

Liu et al. [2025b]

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya
Jia.

Seg-zero: Reasoning-chain guided segmentation via cognitive
reinforcement.

arXiv:2503.06520, 2025b.

Liu et al. [2025c]

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua
Lin, and Jiaqi Wang.

Visual-rft: Visual reinforcement fine-tuning.

arXiv:2503.01785, 2025c.

Lu et al. [2022]

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu,
Oyvind Tafjord, Peter Clark, and Ashwin Kalyan.

Learn to explain: Multimodal reasoning via thought chains for science
question answering.

In NeurIPS, pages 2507–2521, 2022.

Lu et al. [2023a]

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh
Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao.

Mathvista: Evaluating mathematical reasoning of foundation models in
visual contexts.

arXiv:2310.02255, 2023a.

Lu et al. [2023b]

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay
Rajpurohit, Peter Clark, and Ashwin Kalyan.

Dynamic prompt learning via policy gradient for semi-structured
mathematical reasoning.

In ICLR, 2023b.

Lu et al. [2024]

Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and
Han-Jia Ye.

Ovis: Structural embedding alignment for multimodal large language
model.

arXiv:2405.20797, 2024.

Ma et al. [2025]

Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue
Peng, and Xiaojuan Qi.

Unitok: A unified tokenizer for visual generation and understanding.

arXiv:2502.20321, 2025.

Ma et al. [2024]

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng
Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al.

Janusflow: Harmonizing autoregression and rectified flow for unified
multimodal understanding and generation.

arXiv:2411.07975, 2024.

Marafioti et al. [2025]

Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch,
Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi,
et al.

Smolvlm: Redefining small and efficient multimodal models.

arXiv:2504.05299, 2025.

Niu et al. [2025]

Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng
Ning, Bin Zhu, and Li Yuan.

Wise: A world knowledge-informed semantic evaluation for
text-to-image generation.

arXiv:2503.07265, 2025.

OpenAI [2024]

OpenAI.

Openai o1 system card.

arXiv:2412.16720, 2024.

OpenAI [2025]

OpenAI.

Openai o3 and o4-mini system card, 2025.

URL
https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf.

Peng et al. [2025]

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu,
Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang.

Lmm-r1: Empowering 3b lmms with strong reasoning abilities through
two-stage rule-based rl.

arXiv:2503.07536, 2025.

Pi et al. [2024]

Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong
Zhang.

Strengthening multimodal large language model with bootstrapped
preference optimization.

In ECCV, pages 382–398, 2024.

Podell et al. [2023]

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas
Müller, Joe Penna, and Robin Rombach.

Sdxl: Improving latent diffusion models for high-resolution image
synthesis.

arXiv:2307.01952, 2023.

Qiao et al. [2024]

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song,
Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al.

We-math: Does your large multimodal model achieve human-like
mathematical reasoning?

arXiv:2407.01284, 2024.

Qu et al. [2024]

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye,
Daniel K Du, Zehuan Yuan, and Xinglong Wu.

Tokenflow: Unified image tokenizer for multimodal understanding and
generation.

arXiv:2412.03069, 2024.

Radford et al. [2021]

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.

Learning transferable visual models from natural language
supervision.

In ICML, pages 8748–8763, 2021.

Rafailov et al. [2023]

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn.

Direct preference optimization: Your language model is secretly a
reward model.

In NeurIPS, pages 53728–53741, 2023.

Rombach et al. [2022]

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn
Ommer.

High-resolution image synthesis with latent diffusion models.

In CVPR, pages 10684–10695, 2022.

Sarkar et al. [2024]

Pritam Sarkar, Sayna Ebrahimi, Ali Etemad, Ahmad Beirami, Sercan Ö Arık,
and Tomas Pfister.

Mitigating object hallucination via data augmented contrastive
tuning.

arXiv:2405.18654, 2024.

Schulman et al. [2017]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

arXiv:1707.06347, 2017.

Schwenk et al. [2022]

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and
Roozbeh Mottaghi.

A-OKVQA: A benchmark for visual question answering using world
knowledge.

In ECCV, pages 146–162, 2022.

Shao et al. [2024]

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.

Deepseekmath: Pushing the limits of mathematical reasoning in open
language models.

arXiv:2402.03300, 2024.

Si et al. [2025]

Chongjie Si, Jingjing Jiang, and Wei Shen.

Unveiling the mystery of weight in large foundation models: Gaussian
distribution never fades.

arXiv:2501.10661, 2025.

Sun et al. [2024]

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and
Zehuan Yuan.

Autoregressive model beats diffusion: Llama for scalable image
generation.

arXiv:2406.06525, 2024.

Sun et al. [2023]

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen,
Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al.

Aligning large multimodal models with factually augmented rlhf.

arXiv:2309.14525, 2023.

Tan et al. [2025]

Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan
Wang, and Shanghang Zhang.

Reason-rft: Reinforcement fine-tuning for visual reasoning.

arXiv:2503.20752, 2025.

Team [2024]

Chameleon Team.

Chameleon: Mixed-modal early-fusion foundation models.

arXiv:2405.09818, 2024.

Team et al. [2025]

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng
Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.

Kimi k1. 5: Scaling reinforcement learning with llms.

arXiv:2501.12599, 2025.

Tie et al. [2025]

Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin,
Zhejian Yang, Jiangyue Yan, Yao Su, et al.

A survey on post-training of large language models.

arXiv:2503.06072, 2025.

von Werra et al. [2020]

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan
Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec.

Trl: Transformer reinforcement learning.

https://github.com/huggingface/trl, 2020.

Wang et al. [2024a]

Chunwei Wang, Guansong Lu, Junwei Yang, Runhui Huang, Jianhua Han, Lu Hou, Wei
Zhang, and Hang Xu.

Illume: Illuminating your llms to see, draw, and self-enhance.

arXiv:2412.06673, 2024a.

Wang et al. [2024b]

Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and
Muhao Chen.

mdpo: Conditional preference optimization for multimodal large
language models.

arXiv:2406.11839, 2024b.

Wang et al. [2016]

Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo,
Remi Munos, Charles Blundell, Dharshan Kumaran, and Matt Botvinick.

Learning to reinforcement learn.

arXiv:1611.05763, 2016.

Wang et al. [2025a]

Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and
Yu-Gang Jiang.

Simplear: Pushing the frontier of autoregressive visual generation
through pretraining, sft, and rl.

arXiv:2504.11455, 2025a.

Wang et al. [2024c]

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li.

Measuring multimodal mathematical reasoning with math-vision dataset.

arXiv:2402.14804, 2024c.

Wang et al. [2024d]

Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo
Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al.

Enhancing the reasoning ability of multimodal large language models
via mixed preference optimization.

arXiv:2411.10442, 2024d.

Wang et al. [2024e]

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng
Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al.

Emu3: Next-token prediction is all you need.

arXiv:2409.18869, 2024e.

Wang et al. [2025b]

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang.

Unified reward model for multimodal understanding and generation.

arXiv:2503.05236, 2025b.

Wiering and Van Otterlo [2012]

Marco A Wiering and Martijn Van Otterlo.

Reinforcement learning.

Adaptation, learning, and optimization, 12(3):729, 2012.

Wu et al. [2024a]

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen
Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al.

Janus: Decoupling visual encoding for unified multimodal
understanding and generation.

arXiv:2410.13848, 2024a.

Wu et al. [2024b]

Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan,
Song Bai, and Xiang Bai.

Liquid: Language models are scalable multi-modal generators.

arXiv:2412.04332, 2024b.

Wu et al. [2025a]

Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao
Liu, Wei Li, and Chen Change Loy.

Harmonizing visual representations for unified multimodal
understanding and generation.

arXiv:2503.21979, 2025a.

Wu et al. [2025b]

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang,
Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al.

Vila-u: a unified foundation model integrating visual understanding
and generation.

In ICLR, 2025b.

Xiao et al. [2024]

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang.

Logicvista: Multimodal llm logical reasoning benchmark in visual
contexts.

arXiv:2407.04973, 2024.

Xie et al. [2025]

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang,
Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng
Shou.

Show-o: One single transformer to unify multimodal understanding and
generation.

In ICLR, 2025.

Xie et al. [2024]

Rongchang Xie, Chen Du, Ping Song, and Chang Liu.

Muse-vl: Modeling unified vlm through semantic discrete encoding.

arXiv:2411.17762, 2024.

Yang et al. [2025]

Ling Yang, Xinchen Zhang, Ye Tian, Chenming Shang, Minghao Xu, Wentao Zhang,
and Bin Cui.

Hermesflow: Seamlessly closing the gap in multimodal understanding
and generation.

arXiv:2502.12148, 2025.

Yu et al. [2021]

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander
Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu.

Vector-quantized image modeling with improved vqgan.

arXiv:2110.04627, 2021.

Yu et al. [2025]

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian
Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al.

Dapo: An open-source llm reinforcement learning system at scale.

arXiv:2503.14476, 2025.

Yu et al. [2024a]

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu,
Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al.

Rlhf-v: Towards trustworthy mllms via behavior alignment from
fine-grained correctional human feedback.

In CVPR, pages 13807–13816, 2024a.

Yu et al. [2024b]

Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui,
Taiwen He, Zhiyuan Liu, Tat-Seng Chua, et al.

Rlaif-v: Aligning mllms through open-source ai feedback for super
gpt-4v trustworthiness.

arXiv:2405.17220, 2024b.

Yu et al. [2023]

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu,
Xinchao Wang, and Lijuan Wang.

Mm-vet: Evaluating large multimodal models for integrated
capabilities.

arXiv:2308.02490, 2023.

Yue et al. [2024]

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel
Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin
Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao
Huang, Huan Sun, Yu Su, and Wenhu Chen.

Mmmu: A massive multi-discipline multimodal understanding and
reasoning benchmark for expert agi.

In CVPR, pages 9556–9567, 2024.

Zhai et al. [2024]

Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr,
Saining Xie, Yann LeCun, Yi Ma, et al.

Fine-tuning large vision-language models as decision-making agents
via reinforcement learning.

In NeurIPS, pages 110935–110971, 2024.

Zhan et al. [2025]

Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and
Jinqiao Wang.

Vision-r1: Evolving human-free alignment in large vision-language
models via vision-guided reinforcement learning.

arXiv:2503.18013, 2025.

Zhang et al. [2025]

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu,
and Dacheng Tao.

R1-vl: Learning to reason with multimodal large language models via
step-wise group relative policy optimization.

arXiv:2503.12937, 2025.

Zhang et al. [2024a]

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu,
Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al.

Mathverse: Does your multi-modal llm truly see the diagrams in visual
math problems?

arXiv:2403.14624, 2024a.

Zhang et al. [2024b]

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong,
Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al.

Mavis: Mathematical visual instruction tuning.

arXiv:2407.08739, 2024b.

Zhang et al. [2018]

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.

The unreasonable effectiveness of deep features as a perceptual
metric.

In CVPR, pages 586–595, 2018.

Zhao et al. [2023a]

Zhiyuan Zhao, Linke Ouyang, Bin Wang, Siyuan Huang, Pan Zhang, Xiaoyi Dong,
Jiaqi Wang, and Conghui He.

Mllm-dataengine: An iterative refinement approach for mllm.

arXiv:2308.13566, 2023a.

Zhao et al. [2023b]

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He.

Beyond hallucinations: Enhancing lvlms through hallucination-aware
direct preference optimization.

arXiv:2311.16839, 2023b.

Zhou et al. [2024a]

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid
Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy.

Transfusion: Predict the next token and diffuse images with one
multi-modal model.

arXiv:2408.11039, 2024a.

Zhou et al. [2025]

Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, and
Minghui Qiu.

Reinforced mllm: A survey on rl-based reasoning in multimodal large
language models.

arXiv:2504.21277, 2025.

Zhou et al. [2024b]

Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao.

Aligning modalities in vision large language models via preference
fine-tuning.

arXiv:2402.11411, 2024b.

Zhu et al. [2025]

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen
Duan, Hao Tian, Weijie Su, Jie Shao, et al.

Internvl3: Exploring advanced training and test-time recipes for
open-source multimodal models.

arXiv:2504.10479, 2025.

Generated on Fri May 23 06:38:45 2025 by LaTeXML