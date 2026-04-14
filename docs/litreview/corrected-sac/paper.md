Corrected Soft Actor Critic for Continuous Control

1 Introduction

2 Related Work

2.1 Soft Actor-Critic (SAC)

2.2 Action Sampling Methods

2.3 Nonlinear Transformations

2.4 Advancements in Reinforcement Learning

3 Background

3.1 Soft Actor-Critic Algorithm Details

3.2 Impact of the tanh Transformation

4 Methodology

4.1 Inference Phase: Optimal Action Sampling

4.1.1 Deriving the Action Probability Density Function

4.1.2 Identifying the Most Probable Action

4.2 Training Phase: Inverse Transform Sampling

4.2.1 Inverse Transform Sampling Method

4.3 Discussion of the Two Sampling Methods

5 Experiments

5.1 Experimental Setup

5.1.1 Benchmark Tasks

5.1.2 Training Configuration

5.1.3 Compared Methods

5.2 Results and Analysis

5.2.1 Cumulative Rewards

5.2.2 Convergence Speed

5.2.3 Discussion

6 Conclusion

Corrected Soft Actor Critic for Continuous Control

Yanjun Chen1,2, Xinming Zhang2, Xianghui Wang2, Zhiqiang Xu2,
Xiaoyu Shen2, Wei Zhang2

1 Department of Computing, The Hong Kong Polytechnic University

2 Digital Twin Institute, Eastern Institute of Technology, Ningbo, China

yan-jun.chen@connect.polyu.hk   {xyshen,zhw}@eitech.edu.cn

Abstract

The Soft Actor-Critic (SAC) algorithm is known for its stability and high sample efficiency in deep reinforcement learning. However, the tanh transformation applied to sampled actions in SAC distorts the action distribution, hindering the selection of the most probable actions. This paper presents a novel action sampling method that directly identifies and selects the most probable actions within the transformed distribution, thereby addressing this issue. Extensive experiments on standard continuous control benchmarks demonstrate that the proposed method significantly enhances SAC’s performance, resulting in faster convergence and higher cumulative rewards compared to the original algorithm.

1 Introduction

Reinforcement Learning (RL) has become a cornerstone of modern artificial intelligence, underpinning advancements in domains ranging from autonomous systems and robotics to strategic game playing. Among the plethora of RL algorithms, the Soft Actor-Critic (SAC) algorithm introduced by Haarnoja et al. Haarnoja et al. (2018) stands out due to its integration of policy gradient techniques with entropy maximization, which together promote efficient exploration and robust learning in complex environments. SAC has been widely adopted in diverse and challenging tasks, as demonstrated in various applications Kalashnikov et al. (2018).

However, despite its broad adoption and proven success, the SAC algorithm’s action sampling method, particularly the tanh transformation employed to constrain actions within bounded spaces, has not been sufficiently scrutinized. This transformation, while essential for ensuring that actions remain within valid bounds, introduces a significant non-linear distortion to the underlying Gaussian distribution from which actions are sampled Fujimoto et al. (2018). This distortion can lead to a mismatch between the sampled actions and the most probable actions in the original distribution, a problem that becomes increasingly pronounced in high-dimensional action spaces.

In high-dimensional environments, even minor distortions can compound, resulting in substantial errors in action selection. Such errors are particularly detrimental in tasks that require precise action control, such as those involving intricate dynamic interactions or high-dimensional action spaces. The sampling errors induced by the tanh transformation can thus severely hinder the learning process, leading to suboptimal policy performance and slower convergence.

Figure 1: Comparison of probability densities between the Original Sample and the Refine Sample.

To address this critical issue, this paper proposes an optimized action sampling method specifically designed to mitigate the distortive effects of the tanh transformation. The proposed method involves explicitly computing the likelihood of actions within the transformed bounded space and selecting actions that maximize this likelihood. This approach aligns the sampling process more closely with the underlying Gaussian distribution, thereby enhancing the accuracy of action selection and improving the overall performance of the SAC algorithm.

The main contributions of this work are threefold:

•

A novel derivation of the probability density function that accurately characterizes the action distribution post-tanh transformation, addressing the inherent distortion.

•

The development of an efficient sampling algorithm that optimally selects actions based on their likelihood within the bounded action space, leading to improved policy performance across various tasks.

•

Extensive experimental validation using standard continuous control benchmarks, demonstrating significant improvements in both convergence speed and cumulative rewards compared to the baseline SAC algorithm.

2 Related Work

This section provides a comprehensive review of the literature related to the Soft Actor-Critic (SAC) algorithm, action sampling methodologies in reinforcement learning (RL), advancements in RL, and the impact of nonlinear transformations in these contexts. The discussion highlights the challenges and innovations that shape the current landscape of RL, with a particular focus on the distortions introduced by nonlinear transformations like the tanh function.

2.1 Soft Actor-Critic (SAC)

The Soft Actor-Critic (SAC) algorithm, introduced by Haarnoja et al. Haarnoja et al. (2018), represents a significant advancement in reinforcement learning due to its unique integration of entropy maximization with off-policy learning. SAC optimizes a stochastic policy using a maximum entropy objective, encouraging exploration by promoting diverse actions, thereby effectively balancing exploration and exploitation. This balance is crucial for enhancing the algorithm’s stability and sample efficiency, particularly in complex environments. SAC has been successfully applied across various domains, including robotics Levine et al. (2018), autonomous systems Kalashnikov et al. (2018), and gaming Espeholt et al. (2018), demonstrating its robustness and adaptability. However, despite these successes, SAC faces limitations in handling environments with high-dimensional action spaces, particularly due to the tanh transformation used to map actions to a bounded space. This transformation, while necessary, introduces significant distortions to the Gaussian action distribution, potentially leading to suboptimal action selection, as highlighted by Fujimoto et al. Fujimoto et al. (2018). Further comparative studies, such as those by Schulman et al. Schulman et al. (2017), suggest that while SAC excels in stability, Proximal Policy Optimization (PPO) may offer superior performance in scenarios where action distribution distortion is a critical concern. Thus, exploring methods to mitigate these distortions is essential for advancing SAC’s applicability in more complex tasks.

2.2 Action Sampling Methods

Action sampling is a fundamental component of reinforcement learning algorithms, directly influencing the effectiveness of the learned policies. Several sampling techniques have been developed to enhance the accuracy and efficiency of action selection. Importance sampling Precup (2000), for instance, adjusts the probability distribution of actions to better reflect their expected value under a given policy, thus improving learning outcomes. Moreover, adaptive sampling methods, such as those proposed by Mnih et al. Mnih et al. (2016), dynamically adjust the sampling strategy based on the evolving policy, thereby improving convergence rates and overall performance. Another important strategy is the application of Thompson Sampling, which has shown promise in balancing exploration and exploitation by incorporating uncertainty in action selection Russo et al. (2018). However, these methods typically assume that the action space is either unbounded or not subject to nonlinear transformations, an assumption challenged in high-dimensional environments where transformations like tanh are applied. In SAC, these distortions can lead to suboptimal action selection, particularly in complex environments. Studies by Fujimoto et al. Fujimoto et al. (2018) have explored these challenges, yet the need for methods that directly address these distortions remains. Our research fills this gap by proposing a refined action sampling technique that accurately models the transformed action distribution, ensuring optimal action selection within bounded spaces.

2.3 Nonlinear Transformations

Nonlinear transformations, such as the tanh and sigmoid functions, play a critical role in neural network-based reinforcement learning by ensuring that actions remain within a valid range. However, these transformations introduce significant challenges, particularly in high-dimensional action spaces. The primary issue arises from the distortion of the original action distribution when mapped through these nonlinear functions. For instance, the tanh function compresses the output range to [−1,1]11[-1,1][ - 1 , 1 ], leading to a non-uniform distribution that distorts the original Gaussian distribution typically assumed in many RL algorithms, as discussed by Bengio et al. Bengio et al. (1994) and further analyzed in the context of reinforcement learning by Dinh et al. Dinh et al. (2017). The impact of these distortions is particularly pronounced in high-dimensional spaces, where even small deviations can lead to significant errors in policy optimization. For example, in the SAC algorithm, the tanh transformation can result in the underestimation of the probability density near the boundaries of the action space, leading to suboptimal action selection and slower convergence. Kingma et al. Kingma and Welling (2013) explored similar challenges in the context of variational autoencoders, where nonlinear transformations affect the latent space distribution, a concept that parallels the challenges faced in reinforcement learning. Recent research has proposed several approaches to mitigate these issues, including reparameterization of the action space Jang et al. (2016) and the application of regularization techniques Ioffe and Szegedy (2015) that counterbalance the distortive effects. These strategies have been particularly effective in stabilizing learning processes in the presence of nonlinearities, and our work extends these ideas by focusing specifically on the SAC algorithm. We propose a novel action sampling method that directly addresses the distributional distortions caused by the tanh transformation, ensuring that the selected actions are representative of the most probable outcomes, leading to improved policy performance and more stable learning, particularly in high-dimensional and complex environments.

2.4 Advancements in Reinforcement Learning

The field of reinforcement learning has seen substantial progress in improving algorithmic stability, sample efficiency, and overall performance. Notable advancements include curiosity-driven exploration strategies Pathak et al. (2017), which incentivize agents to explore novel states, and value function approximation techniques, such as those used in Deep Deterministic Policy Gradient (DDPG) Lillicrap et al. (2015), which provide more accurate estimates of expected returns. Moreover, hybrid methods that integrate model-free and model-based approaches, like those proposed by Nagabandi et al. Nagabandi et al. (2018), have demonstrated significant potential in enhancing both learning efficiency and policy robustness. Beyond these, the field is witnessing the emergence of distributed reinforcement learning frameworks Espeholt et al. (2018), which leverage parallelism to scale learning processes across multiple environments. This approach not only accelerates training but also increases the robustness of learned policies by exposing the agent to a more diverse set of experiences. Additionally, multi-task and meta-reinforcement learning Finn et al. (2017) are gaining traction as they allow agents to generalize across tasks by leveraging shared knowledge, thus improving adaptability and reducing the need for task-specific tuning. Further, the rise of multi-agent reinforcement learning Lowe et al. (2017) and adversarial learning Pinto et al. (2017) presents new frontiers in RL research, enabling agents to learn in competitive and cooperative settings. Despite these advancements, the specific issue of action sampling distortion in algorithms like SAC remains underexplored. This research aims to fill this gap by developing an optimized sampling technique that addresses the distortive effects of the tanh transformation, thereby enhancing SAC’s performance in terms of both convergence speed and cumulative rewards.

3 Background

The Soft Actor-Critic (SAC) algorithm integrates policy optimization with entropy regularization, providing a robust framework for reinforcement learning, particularly in environments requiring a balance between exploration and exploitation. As a leading method in model-free, off-policy reinforcement learning, SAC has been widely recognized for its ability to maintain a stochastic policy throughout training, which is crucial for avoiding premature convergence to suboptimal strategies. The use of twin Q-functions to estimate the value of state-action pairs further enhances its robustness, reducing the overestimation bias commonly encountered in value-based reinforcement learning methods. However, despite these strengths, SAC is not without its challenges. One significant issue arises from the tanh transformation applied to the action outputs, which, while ensuring actions remain within a bounded range, introduces specific challenges that impact policy performance. Understanding these challenges necessitates an in-depth examination of how action sampling and distributional distortions affect reinforcement learning algorithms like SAC.

3.1 Soft Actor-Critic Algorithm Details

SAC, a leading method in model-free, off-policy reinforcement learning, integrates policy optimization with entropy regularization. The algorithm employs two Q-functions to evaluate state-action pairs and a policy network to generate actions. These Q-functions minimize Bellman error, while the policy network maximizes expected rewards combined with an entropy term. The entropy term, modulated by a temperature parameter (α𝛼\alphaitalic_α), is essential in balancing exploration and exploitation, ensuring the policy remains stochastic and preventing convergence to suboptimal strategies. The SAC optimization objective is:

J(π)=∑t=0T𝔼(st,at)∼ρπ[r(st,at)+αℋ(π(⋅|st))],J(\pi)=\sum_{t=0}^{T}\mathbb{E}_{(s_{t},a_{t})\sim\rho_{\pi}}\left[r(s_{t},a_{%
t})+\alpha\mathcal{H}(\pi(\cdot|s_{t}))\right],italic_J ( italic_π ) = ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT blackboard_E start_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ∼ italic_ρ start_POSTSUBSCRIPT italic_π end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ italic_r ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) + italic_α caligraphic_H ( italic_π ( ⋅ | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ) ] ,

(1)

This equation underscores the dual objectives of maximizing cumulative rewards and maintaining high policy entropy, crucial for robust exploration and stable learning in complex environments. Notably, SAC’s ability to operate in continuous action spaces with high dimensionality has made it particularly effective in robotics and other domains where precise control is paramount.

3.2 Impact of the tanh Transformation

In SAC, actions are typically sampled from a Gaussian distribution to support unbounded exploration. However, the tanh transformation is applied to ensure actions remain within the feasible range [−1,1]11[-1,1][ - 1 , 1 ]. While necessary, this transformation introduces significant distortions, particularly at the distribution’s extremes Chou et al. (2017); Kim et al. (2019). The probability density function (PDF) of transformed actions is expressed as:

π⁢(a|s)=μ⁢(u|s)⁢|det(d⁢ad⁢u)|−1,𝜋conditional𝑎𝑠𝜇conditional𝑢𝑠superscript
𝑑𝑎𝑑𝑢1\pi(a|s)=\mu(u|s)\left|\det\left(\frac{da}{du}\right)\right|^{-1},italic_π ( italic_a | italic_s ) = italic_μ ( italic_u | italic_s ) | roman_det ( divide start_ARG italic_d italic_a end_ARG start_ARG italic_d italic_u end_ARG ) | start_POSTSUPERSCRIPT - 1 end_POSTSUPERSCRIPT ,

(2)

Here, μ⁢(u|s)𝜇conditional𝑢𝑠\mu(u|s)italic_μ ( italic_u | italic_s ) represents the unbounded Gaussian distribution’s probability density, with the Jacobian determinant given by:

|det(d⁢ad⁢u)|=∏i(1−tanh2⁡(ui)).
𝑑𝑎𝑑𝑢subscriptproduct𝑖1superscript2subscript𝑢𝑖\left|\det\left(\frac{da}{du}\right)\right|=\prod_{i}\left(1-\tanh^{2}(u_{i})%
\right).| roman_det ( divide start_ARG italic_d italic_a end_ARG start_ARG italic_d italic_u end_ARG ) | = ∏ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( 1 - roman_tanh start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ( italic_u start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ) .

(3)

This transformation compresses the distribution’s tails, leading to an underestimation of boundary actions’ probabilities Fujimoto et al. (2018); Gu et al. (2017). Such distortions can result in suboptimal policy updates, especially in tasks requiring precise control. The tanh transformation’s impact is further exacerbated in high-dimensional action spaces, where even minor distortions can significantly alter the action distribution, leading to inefficient policy learning and suboptimal decision-making.

The tanh transformation poses significant challenges in tasks where boundary actions are critical. For instance, in tasks requiring fine motor control or environments with high-dimensional action spaces, the compression of the action distribution’s tails can degrade performance Fujimoto et al. (2018). Experimental studies indicate that this effect intensifies with task complexity, complicating the learning process further. Our proposed method optimizes action sampling by modeling the transformed distribution accurately, ensuring that selected actions reflect the most probable outcomes, thereby enhancing policy effectiveness across complex tasks.

4 Methodology

This section introduces two optimal action sampling methods designed to correct the distributional distortions caused by the tanh transformation in the Soft Actor-Critic (SAC) algorithm. The first method is applied during the inference phase, where precise action selection is critical, and the second method is employed during the training phase to maintain necessary stochasticity for effective exploration.

4.1 Inference Phase: Optimal Action Sampling

The optimal action sampling approach during the inference phase addresses the distortions introduced by the tanh transformation when mapping actions sampled from a Gaussian distribution to the bounded space [−1,1]11[-1,1][ - 1 , 1 ]. The goal is to identify the most probable action that best represents the underlying distribution after transformation.

4.1.1 Deriving the Action Probability Density Function

Given a Gaussian-distributed action u𝑢uitalic_u sampled from the SAC policy network, represented as:

u∼𝒩⁢(μ,σ2),similar-to𝑢𝒩𝜇superscript𝜎2u\sim\mathcal{N}(\mu,\sigma^{2}),italic_u ∼ caligraphic_N ( italic_μ , italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) ,

(4)

where μ𝜇\muitalic_μ and σ2superscript𝜎2\sigma^{2}italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT denote the mean and variance of the distribution, respectively. The action u𝑢uitalic_u is transformed using the tanh function to confine it within the bounded space [−1,1]11[-1,1][ - 1 , 1 ]:

y=tanh⁡(u).𝑦𝑢y=\tanh(u).italic_y = roman_tanh ( italic_u ) .

(5)

To derive the probability density function (PDF) of the transformed action y𝑦yitalic_y, the change of variables technique is employed. The original Gaussian PDF is given by:

p⁢(u)=12⁢π⁢σ2⁢exp⁡(−(u−μ)22⁢σ2).𝑝𝑢
12𝜋superscript𝜎2
superscript𝑢𝜇22superscript𝜎2p(u)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{(u-\mu)^{2}}{2\sigma^{2}}%
\right).italic_p ( italic_u ) = divide start_ARG 1 end_ARG start_ARG square-root start_ARG 2 italic_π italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG end_ARG roman_exp ( - divide start_ARG ( italic_u - italic_μ ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG start_ARG 2 italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG ) .

(6)

u𝑢uitalic_u can be expressed as a function of y𝑦yitalic_y as:

u=12⁢log⁡(1+y1−y).𝑢
12

1𝑦1𝑦u=\frac{1}{2}\log\left(\frac{1+y}{1-y}\right).italic_u = divide start_ARG 1 end_ARG start_ARG 2 end_ARG roman_log ( divide start_ARG 1 + italic_y end_ARG start_ARG 1 - italic_y end_ARG ) .

(7)

The Jacobian determinant d⁢ud⁢y
𝑑𝑢𝑑𝑦\frac{du}{dy}divide start_ARG italic_d italic_u end_ARG start_ARG italic_d italic_y end_ARG of the transformation, which accounts for the non-linear distortion introduced by the tanh function, is calculated as:

d⁢ud⁢y=11−y2.
𝑑𝑢𝑑𝑦
11superscript𝑦2\frac{du}{dy}=\frac{1}{1-y^{2}}.divide start_ARG italic_d italic_u end_ARG start_ARG italic_d italic_y end_ARG = divide start_ARG 1 end_ARG start_ARG 1 - italic_y start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG .

(8)

Thus, the probability density function of the transformed action y𝑦yitalic_y becomes:

p⁢(y)=11−y2⋅12⁢π⁢σ2⁢exp⁡(−12⁢σ2⁢[12⁢log⁡(1+y1−y)−μ]2).𝑝𝑦⋅
11superscript𝑦2
12𝜋superscript𝜎2
12superscript𝜎2superscriptdelimited-[]
12

1𝑦1𝑦𝜇2p(y)=\frac{1}{1-y^{2}}\cdot\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{1}{%
2\sigma^{2}}\left[\frac{1}{2}\log\left(\frac{1+y}{1-y}\right)-\mu\right]^{2}%
\right).italic_p ( italic_y ) = divide start_ARG 1 end_ARG start_ARG 1 - italic_y start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG ⋅ divide start_ARG 1 end_ARG start_ARG square-root start_ARG 2 italic_π italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG end_ARG roman_exp ( - divide start_ARG 1 end_ARG start_ARG 2 italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG [ divide start_ARG 1 end_ARG start_ARG 2 end_ARG roman_log ( divide start_ARG 1 + italic_y end_ARG start_ARG 1 - italic_y end_ARG ) - italic_μ ] start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) .

(9)

This equation accurately characterizes the effect of the tanh transformation on the action distribution, providing a corrected probability density function used to identify the most probable action.

4.1.2 Identifying the Most Probable Action

To maximize the probability density function derived in Equation 9, the objective during the inference phase is to find the action y∗superscript𝑦y^{*}italic_y start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT that satisfies:

y∗=arg⁡maxy∈(−1,1)⁡p⁢(y).superscript𝑦subscript𝑦11𝑝𝑦y^{*}=\arg\max_{y\in(-1,1)}p(y).italic_y start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = roman_arg roman_max start_POSTSUBSCRIPT italic_y ∈ ( - 1 , 1 ) end_POSTSUBSCRIPT italic_p ( italic_y ) .

(10)

This is achieved by discretizing the action space y𝑦yitalic_y, calculating p⁢(y)𝑝𝑦p(y)italic_p ( italic_y ) at each point, and selecting the action with the highest probability.

Algorithm 1 Sampling for Inference Phase

Input: Mean μ𝜇\muitalic_μ and variance σ2superscript𝜎2\sigma^{2}italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT from the policy network.

Output: Most probable action y∗superscript𝑦y^{*}italic_y start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT.

1:  Initialize y0=−0.999subscript𝑦00.999y_{0}=-0.999italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = - 0.999, Δ⁢y=0.001Δ𝑦0.001\Delta y=0.001roman_Δ italic_y = 0.001, N=2000𝑁2000N=2000italic_N = 2000.

2:  Define p⁢(y)𝑝𝑦p(y)italic_p ( italic_y ) using Eq. 9.

3:  for  i=0𝑖0i=0italic_i = 0 to N𝑁Nitalic_N do

4:     yi=y0+Δ⁢y⋅isubscript𝑦𝑖
subscript𝑦0⋅Δ𝑦𝑖y_{i}=y_{0}+\Delta y\cdot iitalic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT + roman_Δ italic_y ⋅ italic_i.

5:     Calculate p⁢(yi)𝑝subscript𝑦𝑖p(y_{i})italic_p ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ).

6:  end for

7:  y∗=arg⁡maxyi⁡p⁢(yi)superscript𝑦subscriptsubscript𝑦𝑖𝑝subscript𝑦𝑖y^{*}=\arg\max_{y_{i}}p(y_{i})italic_y start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = roman_arg roman_max start_POSTSUBSCRIPT italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT italic_p ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ).

8:  return y∗superscript𝑦y^{*}italic_y start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT.

4.2 Training Phase: Inverse Transform Sampling

During training, maintaining stochasticity in action selection is essential for effective exploration. Instead of selecting the most probable action, the inverse transform sampling method is used to sample actions according to their likelihood, preserving the necessary randomness for robust learning.

4.2.1 Inverse Transform Sampling Method

The cumulative distribution function (CDF) is computed from the probability density function:

C⁢(y)=∫−1yp⁢(y′)⁢𝑑y′.𝐶𝑦superscriptsubscript1𝑦𝑝superscript𝑦′differential-dsuperscript𝑦′C(y)=\int_{-1}^{y}p(y^{\prime})\,dy^{\prime}.italic_C ( italic_y ) = ∫ start_POSTSUBSCRIPT - 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_y end_POSTSUPERSCRIPT italic_p ( italic_y start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) italic_d italic_y start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT .

(11)

A random value r𝑟ritalic_r uniformly sampled from [0,1]01[0,1][ 0 , 1 ] is used to determine the sampled action ysampledsubscript𝑦sampledy_{\text{sampled}}italic_y start_POSTSUBSCRIPT sampled end_POSTSUBSCRIPT, where C⁢(y)=r𝐶𝑦𝑟C(y)=ritalic_C ( italic_y ) = italic_r.

Algorithm 2 Sampling for Training Phase

Input: Mean μ𝜇\muitalic_μ and variance σ2superscript𝜎2\sigma^{2}italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT from the policy network.

Output: Sampled action ysampledsubscript𝑦sampledy_{\text{sampled}}italic_y start_POSTSUBSCRIPT sampled end_POSTSUBSCRIPT.

1:  Initialize y0=−0.999subscript𝑦00.999y_{0}=-0.999italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = - 0.999, Δ⁢y=0.001Δ𝑦0.001\Delta y=0.001roman_Δ italic_y = 0.001, N=2000𝑁2000N=2000italic_N = 2000.

2:  Define p⁢(y)𝑝𝑦p(y)italic_p ( italic_y ) using Eq. 9.

3:  Compute the cumulative distribution function (CDF) C⁢(yi)𝐶subscript𝑦𝑖C(y_{i})italic_C ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) based on p⁢(yi)𝑝subscript𝑦𝑖p(y_{i})italic_p ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ).

4:  Generate a random value r∼𝒰⁢(0,1)similar-to𝑟𝒰01r\sim\mathcal{U}(0,1)italic_r ∼ caligraphic_U ( 0 , 1 ).

5:  Identify the smallest yisubscript𝑦𝑖y_{i}italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT such that C⁢(yi)≥r𝐶subscript𝑦𝑖𝑟C(y_{i})\geq ritalic_C ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ≥ italic_r.

6:  return ysampled=yisubscript𝑦sampledsubscript𝑦𝑖y_{\text{sampled}}=y_{i}italic_y start_POSTSUBSCRIPT sampled end_POSTSUBSCRIPT = italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

4.3 Discussion of the Two Sampling Methods

These two sampling methods are designed to cater to the distinct needs of SAC during inference and training. The inference phase requires precise action selection, which is achieved through optimal sampling, while the training phase benefits from maintaining diversity in exploration, supported by inverse transform sampling. Together, these methods enhance SAC’s overall performance by addressing the distortions introduced by the tanh transformation.

5 Experiments

This section rigorously evaluates the proposed optimal action sampling methods within the MuJoCo environment, renowned for its high-fidelity physics simulation. The experiments aim to assess the impact of these methods on the performance of the Soft Actor-Critic (SAC) algorithm, particularly focusing on key metrics such as convergence speed and cumulative rewards. By selecting a diverse set of continuous control tasks, this study ensures a comprehensive evaluation across varying levels of complexity and control precision.

5.1 Experimental Setup

5.1.1 Benchmark Tasks

The evaluation is conducted on six continuous control tasks within the MuJoCo environment: HalfCheetah-v4, Hopper-v4, Walker2d-v4, Humanoid-v4, HumanoidStandup-v4, and Reacher-v4. These tasks were selected for their ability to challenge different aspects of reinforcement learning algorithms:

•

HalfCheetah-v4: A task emphasizing speed and balance in a planar cheetah model, often used to evaluate algorithms’ efficiency in optimizing forward motion.

•

Hopper-v4: This task tests the stability and precision of a single-legged robot’s hopping motion, highlighting the importance of balance and control.

•

Walker2d-v4: A bipedal robot tasked with walking forward, providing a moderate level of complexity that requires both stability and dynamic control.

•

Humanoid-v4: One of the most complex tasks, involving a humanoid robot that must maintain balance while walking, representing a high-dimensional control challenge.

•

HumanoidStandup-v4: A highly challenging task where a humanoid must stand up from a lying position, requiring significant coordination and control.

•

Reacher-v4: A simpler task where a robotic arm must reach a target, used to assess precision in low-dimensional action spaces.

These tasks were chosen not only for their varying levels of difficulty but also for their relevance to real-world applications, such as robotics and autonomous systems, where precise control and optimization are critical.

5.1.2 Training Configuration

To ensure fair and consistent evaluation across all tasks, the experiments were conducted using a standardized set of configurations, as detailed in Table 1. Each parameter was meticulously selected to optimize the stability and learning efficiency of the SAC algorithm:

Parameter
Value

Temperature Parameter (α𝛼\alphaitalic_α)
0.2

Neural Network Architecture
[256, 256]

Learning Rate
1×10−31superscript1031\times 10^{-3}1 × 10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT

Discount Factor (γ𝛾\gammaitalic_γ)
0.99

Soft Update Coefficient (τ𝜏\tauitalic_τ)
0.005

Initial Exploration Steps
10,000

Training Episodes
1,000

Number of Seeds
10

Table 1: Training Configuration Parameters

The training process employed a sliding window average with a window size of 10 to smooth the reward curves, thereby enhancing the clarity of the results. The reported values reflect the mean performance over 10 trials, with standard deviation bands included to illustrate the variability across runs.

Figure 2: Cumulative rewards across different MuJoCo tasks. Each curve represents the mean of 10 trials, smoothed with a sliding window (window size = 10). Shaded regions denote the standard deviation across trials, highlighting the stability of each method.

5.1.3 Compared Methods

The proposed optimal action sampling methods were rigorously compared against a baseline configuration of the Soft Actor-Critic (SAC) algorithm under identical experimental conditions. The methods evaluated include:

•

Original-deterministic (Baseline): This method represents the standard SAC algorithm, where actions are sampled from a Gaussian distribution during training. During inference, deterministic action selection is performed by using the mean of the Gaussian distribution, followed by a tanh transformation to map the actions into the bounded action space.

•

Original-refineSampling: In this configuration, the standard SAC approach is maintained during training, employing Gaussian sampling as described in the Baseline method. However, during inference, this method utilizes the Sampling for Inference Phase algorithm , which refines action selection by considering the likelihood of actions within the transformed space.

•

RefineT-refineSampling: This method introduces a significant change by applying the Sampling for Training Phase algorithm during the training process, which incorporates the proposed inverse transform sampling technique. This approach maintains the stochastic nature of the action selection process while simultaneously correcting the distortions introduced by the tanh transformation. For the inference phase, this method again applies the Sampling for Inference Phase algorithm, ensuring precise action selection.

These methods were evaluated across a suite of continuous control tasks within the MuJoCo environment, with a primary focus on assessing improvements in convergence speed and cumulative rewards relative to the baseline, thus providing a comprehensive analysis of the proposed sampling strategies.

5.2 Results and Analysis

5.2.1 Cumulative Rewards

Cumulative rewards represent the total accumulated rewards over a defined training period in reinforcement learning. Mathematically, it is expressed as:

Cumulative Reward (CR)=∑t=0Trt,Cumulative Reward (CR)superscriptsubscript𝑡0𝑇subscript𝑟𝑡\text{Cumulative Reward (CR)}=\sum_{t=0}^{T}r_{t},Cumulative Reward (CR) = ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ,

(12)

where rtsubscript𝑟𝑡r_{t}italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT denotes the reward at time step t𝑡titalic_t, and T𝑇Titalic_T is the total number of time steps. This metric provides a comprehensive evaluation of the policy’s effectiveness in maximizing returns during its interactions with the environment.

Figure 2 illustrates the cumulative rewards across various MuJoCo tasks, highlighting the effectiveness of the proposed methods in different environments.

In the complex Humanoid-v4 task, both RefineT-refineSampling and Original-refineSampling methods surpass the Original-deterministic method, achieving a 10% to 20% improvement, with RefineT-refineSampling exhibiting the highest stability, as indicated by narrower standard deviation bands.

Similarly, in the HumanoidStandup-v4 task, RefineT-refineSampling outperforms the baseline by 5% to 10%, demonstrating the benefits of advanced sampling techniques in complex environments.

For tasks of moderate complexity, such as Hopper-v4, RefineT-refineSampling slightly exceeds the other methods, with reduced variability, underscoring its consistency.

In contrast, for simpler tasks like Reacher-v4, the Original-deterministic and Original-refineSampling methods perform marginally better than RefineT-refineSampling, reflecting the diminished need for complex sampling strategies in less demanding environments.

Figure 3: Convergence speed across different MuJoCo tasks, represented as bar plots. Each bar shows the mean convergence speed across 10 trials.

5.2.2 Convergence Speed

Convergence speed is redefined as the average rate of change in reward values across training epochs. Instead of merely measuring the number of epochs required to reach a fixed percentage of the maximum reward, this refined definition captures the overall learning efficiency by evaluating the consistency and pace of reward improvement during training.

Mathematically, convergence speed is calculated as the mean difference in reward values between successive epochs. This approach provides a more granular assessment of the learning process, reflecting not only the time to reach a specific performance level but also the stability and uniformity of the learning progression.

Formally, the convergence speed S𝑆Sitalic_S can be expressed as:

S=1N−1⁢∑i=2N(Ri−Ri−1),𝑆
1𝑁1superscriptsubscript𝑖2𝑁subscript𝑅𝑖subscript𝑅𝑖1S=\frac{1}{N-1}\sum_{i=2}^{N}(R_{i}-R_{i-1}),italic_S = divide start_ARG 1 end_ARG start_ARG italic_N - 1 end_ARG ∑ start_POSTSUBSCRIPT italic_i = 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT ( italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_R start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT ) ,

(13)

where Risubscript𝑅𝑖R_{i}italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denotes the reward at the i𝑖iitalic_ith epoch, and N𝑁Nitalic_N is the total number of epochs. This metric highlights the average improvement in reward per epoch, offering a more comprehensive view of the model’s learning dynamics and optimization efficiency.

Figure 3 illustrates the convergence speed across various tasks, highlighting significant performance differences among the proposed methods.

In the most complex task, Humanoid-v4, RefineT-refineSampling outperforms both Original-refineSampling and Original-deterministic, demonstrating its effectiveness in high-dimensional action spaces. In contrast, for the HumanoidStandup-v4 task, both refined methods exhibit slower convergence compared to the baseline, likely due to the task’s inherent difficulty, which constrains further reward improvements.

For moderately complex tasks such as Hopper-v4 and Walker2d-v4, RefineT-refineSampling consistently surpasses the other methods, reflecting its capability to effectively balance exploration and exploitation. However, in the HalfCheetah-v4 task, Original-refineSampling achieves faster convergence, suggesting that in specific scenarios, a simpler refinement may be more suitable.

In the simplest task, Reacher-v4, the baseline method achieves the fastest convergence, as anticipated, due to the task’s lower complexity, which reduces the necessity for advanced sampling strategies.

5.2.3 Discussion

The experimental results provide compelling evidence of the advantages offered by the RefineT-refineSampling method, particularly in high-dimensional action spaces. As demonstrated in the Humanoid-v4 and HumanoidStandup-v4 tasks, RefineT-refineSampling not only achieves significant improvements in cumulative rewards—ranging from 5% to 20%—but also exhibits faster convergence compared to the Original-deterministic methods. This enhancement is largely attributed to the method’s ability to accurately model the action distribution post-tanh transformation, thereby mitigating the distortive effects that often impede policy optimization in complex environments.

In the Humanoid-v4 task, RefineT-refineSampling’s ability to refine the action selection process during both training and inference phases ensures that the selected actions align more closely with the most probable outcomes within the bounded action space. This alignment is crucial in high-dimensional tasks where even minor distortions can significantly affect policy performance. The narrow standard deviation bands associated with RefineT-refineSampling further underscore its stability, making it a robust choice for tasks that demand precise control and optimal exploration.

However, the performance of the Original-deterministic method in simpler tasks such as Reacher-v4 reveals an important consideration: the complexity of the task and the dimensionality of the action space play a pivotal role in determining the effectiveness of advanced sampling techniques. In low-dimensional tasks where the action space is less complex and the potential for distortion is reduced, the overhead introduced by methods like RefineT-refineSampling may not provide substantial benefits. Here, the simplicity and efficiency of the baseline SAC approach suffice, achieving optimal performance without the need for additional refinements.

These findings emphasize the necessity of selecting an appropriate sampling strategy based on the task’s complexity and the dimensionality of the action space. RefineT-refineSampling excels in high-dimensional and challenging environments where precision in action selection is paramount. In contrast, simpler tasks may not require such advanced techniques, allowing traditional methods to perform effectively with lower computational overhead. Future research should focus on developing hybrid approaches that dynamically adjust the sampling strategy based on the task characteristics, optimizing both performance and computational efficiency across a broader range of environments.

6 Conclusion

This study addresses a critical challenge in the Soft Actor-Critic (SAC) algorithm— the distributional distortions introduced by the tanh transformation during action sampling. By deriving an accurate probability density function and developing an optimized sampling method, this research significantly enhances SAC’s performance across various continuous control tasks. The proposed RefineT-refineSampling method not only improves convergence speed in high-dimensional tasks like Humanoid-v4 but also achieves higher cumulative rewards, demonstrating its effectiveness in complex environments. These findings underscore the method’s potential for broader applications in reinforcement learning, particularly in tasks requiring precise action control in high-dimensional spaces.

References

Bengio et al. [1994]

Yoshua Bengio, Patrice Simard, and Paolo Frasconi.

Learning long-term dependencies with gradient descent is difficult.

IEEE transactions on neural networks, 5(2):157–166, 1994.

Chou et al. [2017]

Po-Wei Chou, Daniel Maturana, and Sebastian Scherer.

Improving stochastic policy gradients in continuous control with deep reinforcement learning using the beta distribution.

In International conference on machine learning, pages 834–843. PMLR, 2017.

Dinh et al. [2017]

Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio.

Sharp minima can generalize for deep nets.

In International Conference on Machine Learning, pages 1019–1028. PMLR, 2017.

Espeholt et al. [2018]

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al.

Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures.

In International conference on machine learning, pages 1407–1416. PMLR, 2018.

Finn et al. [2017]

Chelsea Finn, Pieter Abbeel, and Sergey Levine.

Model-agnostic meta-learning for fast adaptation of deep networks.

In International conference on machine learning, pages 1126–1135. PMLR, 2017.

Fujimoto et al. [2018]

Scott Fujimoto, Herke Hoof, and David Meger.

Addressing function approximation error in actor-critic methods.

In International conference on machine learning, pages 1587–1596. PMLR, 2018.

Gu et al. [2017]

Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E. Turner, and Sergey Levine.

Q-prop: Sample-efficient policy gradient with an off-policy critic, 2017.

URL https://arxiv.org/abs/1611.02247.

Haarnoja et al. [2018]

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.

In International conference on machine learning, pages 1861–1870. PMLR, 2018.

Ioffe and Szegedy [2015]

Sergey Ioffe and Christian Szegedy.

Batch normalization: Accelerating deep network training by reducing internal covariate shift.

In International conference on machine learning, pages 448–456. pmlr, 2015.

Jang et al. [2016]

Eric Jang, Shixiang Gu, and Ben Poole.

Categorical reparameterization with gumbel-softmax.

arXiv preprint arXiv:1611.01144, 2016.

Kalashnikov et al. [2018]

Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al.

Scalable deep reinforcement learning for vision-based robotic manipulation.

In Conference on robot learning, pages 651–673. PMLR, 2018.

Kim et al. [2019]

Seungchan Kim, Kavosh Asadi, Michael Littman, and George Konidaris.

Deepmellow: removing the need for a target network in deep q-learning.

In Proceedings of the twenty eighth international joint conference on artificial intelligence, 2019.

Kingma and Welling [2013]

Diederik P Kingma and Max Welling.

Auto-encoding variational bayes.

arXiv preprint arXiv:1312.6114, 2013.

Levine et al. [2018]

Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen.

Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection.

The International journal of robotics research, 37(4-5):421–436, 2018.

Lillicrap et al. [2015]

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.

Continuous control with deep reinforcement learning.

arXiv preprint arXiv:1509.02971, 2015.

Lowe et al. [2017]

Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch.

Multi-agent actor-critic for mixed cooperative-competitive environments.

Advances in neural information processing systems, 30, 2017.

Mnih et al. [2016]

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu.

Asynchronous methods for deep reinforcement learning.

In International conference on machine learning, pages 1928–1937. PMLR, 2016.

Nagabandi et al. [2018]

Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine.

Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning.

In 2018 IEEE international conference on robotics and automation (ICRA), pages 7559–7566. IEEE, 2018.

Pathak et al. [2017]

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell.

Curiosity-driven exploration by self-supervised prediction.

In International conference on machine learning, pages 2778–2787. PMLR, 2017.

Pinto et al. [2017]

Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta.

Robust adversarial reinforcement learning.

In International conference on machine learning, pages 2817–2826. PMLR, 2017.

Precup [2000]

Doina Precup.

Eligibility traces for off-policy policy evaluation.

Computer Science Department Faculty Publication Series, page 80, 2000.

Russo et al. [2018]

Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al.

A tutorial on thompson sampling.

Foundations and Trends® in Machine Learning, 11(1):1–96, 2018.

Schulman et al. [2017]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347, 2017.

Generated on Tue Oct 22 06:46:04 2024 by LaTeXML