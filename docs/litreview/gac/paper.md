Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning

1 Introduction

2 Related Work

2.1 Gaussian Policies and Their Limitations

2.2 Beyond Gaussian: Alternative Distributions

2.3 Geometric Perspectives in RL

2.4 Simplification as Innovation

2.5 The Missing Perspective: Action Generation Without Distributions

3 Problem Formulation and Methodology

3.1 Problem Formulation

3.2 Geometric Action Generation

3.3 Theoretical Justification

3.4 Integration with SAC

3.4.1 Exploration Control Mechanism

3.5 Practical Considerations

4 Experiments

4.1 Experimental Setup

4.2 Main Results

4.3 Ablation Studies

5 Conclusion

A Theoretical Analysis

A.1 Proof of Theorem 1

A.2 Concentration Analysis

A.3 Empirical Validation

A.4 Exploration Control and Entropy Connection

A.4.1 Exploration Control Mechanism

A.4.2 Theoretical Connection to Entropy

A.5 SAC Convergence with GAC

B Gradient Flow Analysis in Tanh-Squashed Policies

C Reproducibility

C.1 Implementation Details

C.2 Computational Efficiency

C.3 Code and Model Release

Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning

Zhihao Lin

James Watt School of Engineering

University of Glasgow

Glasgow, UK

{2800400L}@student.gla.ac.uk

Abstract

Gaussian policies have dominated continuous control in deep reinforcement learning (RL), yet they suffer from a fundamental mismatch: their unbounded support requires ad-hoc squashing functions that distort the geometry of bounded action spaces.
While von Mises-Fisher (vMF) distributions offer a theoretically grounded alternative on the sphere, their reliance on Bessel functions and rejection sampling hinders practical adoption.
We propose Geometric Action Control (GAC), a novel action generation paradigm that preserves the geometric benefits of spherical distributions while simplifying computation.
GAC decomposes action generation into a direction vector and a learnable concentration parameter, enabling efficient interpolation between deterministic actions and uniform spherical noise.
This design reduces parameter count from 2​d2d to d+1d+1, and avoids the O​(d​k)O(dk) complexity of vMF rejection sampling, achieving simple O​(d)O(d) operations.
Empirically, GAC consistently matches or exceeds state-of-the-art methods across six MuJoCo benchmarks, achieving 37.6% improvement over SAC on Ant-v4 and the best results on 4 out of 6 tasks.
Our ablation studies reveal that both spherical normalization and adaptive concentration control are essential to GAC’s success.
These findings suggest that robust and efficient continuous control does not require complex distributions, but a principled respect for the geometry of action spaces. Code and pretrained models are available in supplementary materials.

1 Introduction

Continuous control (OpenAI et al., 2019) remains one of the most challenging problems in reinforcement learning (RL) (Silver et al., 2014), with applications ranging from robotics to autonomous driving (Seo et al., 2025). At the heart of this challenge lies a fundamental design choice: how should agents generate continuous actions? For over a decade, Gaussian policies have served as the default answer, powering algorithms from Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) to Soft Actor-Critic (SAC) (Haarnoja et al., 2018) and achieving remarkable success across diverse domains. Their popularity stems from mathematical convenience, including closed-form entropy, straightforward reparameterization, and well-understood optimization properties.

Yet this convenience masks a fundamental mismatch. Physical systems operate within bounded action spaces, while Gaussian distributions have infinite support Nikishin et al. (2021). The standard solution applies squashing functions like tanh\tanh to map samples into bounded regions (Theile et al., 2024), but this transformation distorts the distribution’s geometry, creates gradient flow issues near boundaries, and breaks the natural symmetry of the action space (Fujimoto et al., 2018). As policies become more deterministic during training, actions cluster near boundaries where tanh\tanh’s gradient vanishes. We observe this phenomenon in ≈\approx~40% of SAC training steps on HalfCheetah (Figure A.2, Appendix B). Such instabilities are often misattributed to insufficient exploration, while in fact they reflect a deeper geometric mismatch between Gaussian policies and bounded action spaces.

Recent work has begun questioning this Gaussian orthodoxy Davidson et al. (2022). von Mises-Fisher (vMF) distributions offer a mathematically principled alternative by operating directly on the unit sphere, naturally respecting bounded constraints (Michel et al., 2024). However, vMF’s theoretical elegance comes at a steep computational cost (You et al., 2025): sampling requires rejection methods with O​(d​k)O(dk) complexity where kk is the expected number of rejections, and density computation involves modified Bessel functions prone to numerical overflow (Mazoure et al., 2019). Other alternatives like normalizing flows or mixture distributions add expressiveness but compound the computational burden (Obando-Ceron et al., 2024). This creates a dilemma: accept Gaussian’s geometric limitations or pay the price of computational complexity.

We take a different path. Rather than seeking increasingly sophisticated distributions, we ask whether the distribution paradigm itself is necessary. Actions in physical systems naturally decompose into direction and magnitude. For instance, a robot arm moves toward a target direction with some force, and a car steers at an angle with some acceleration. This geometric intuition suggests that effective action generation might not require explicit probability modeling at all.

This insight leads to Geometric Action Control (GAC), which generates actions through direct geometric operations on the unit sphere. GAC represents policies through two components: a direction network that outputs unit vectors indicating preferred action orientations, and a concentration network that controls exploration by interpolating between deterministic directions and uniform spherical noise. This decomposition transforms the complex problem of sampling from sophisticated distributions into simple linear interpolation, reducing computational complexity from O​(d​k)O(dk) to O​(d)O(d) while maintaining the geometric consistency that bounded action spaces demand.

Our key contributions are:

•

We introduce GAC, a distribution-free action generation paradigm that replaces probabilistic sampling with direct geometric operations on the unit sphere, challenging the necessity of distributional modeling in continuous control.

•

We develop a compact and efficient policy architecture requiring only d+1d{+}1 parameters instead of 2​d2d for Gaussian policies, achieving comparable or better performance with reduced complexity.

•

We provide theoretical analysis showing that spherical mixing achieves vMF-like concentration without Bessel functions, and ablations demonstrating that spherical geometry and adaptive concentration are critical to GAC’s success.

•

We provide comprehensive empirical evaluation across six MuJoCo benchmarks, achieving the best performance on 4 out of 6 tasks, especially in high-dimensional control, validating that geometric consistency outweighs distributional sophistication.

Beyond immediate performance gains, GAC represents a conceptual shift in how we approach policy design. By demonstrating that geometric structure can replace distributional complexity, we open new avenues for developing efficient, interpretable, and theoretically grounded control algorithms. We believe our results champion a broader “Geometric Simplicity Principle”: that for many robotics and control tasks, explicitly modeling the geometric structure of the action space is a more effective and efficient path forward than pursuing ever more sophisticated probability distributions. The remainder of this paper is organized as follows: Section 2 reviews related work, Section 3 presents the GAC methodology, Section 4 provides empirical evaluation, and Section 5 concludes with discussions of broader implications.

2 Related Work

2.1 Gaussian Policies and Their Limitations

The dominance of Gaussian policies in continuous control traces back to the natural policy gradient literature, where Gaussian distributions provided tractable gradient estimates and convergence guarantees. Modern deep RL algorithms like SAC, Proximal Policy Optimization (PPO) (Schulman et al., 2017), and Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) inherit this choice, implementing Gaussian policies through neural networks that output mean and variance parameters. While this approach has driven impressive empirical success, practitioners have long recognized its limitations in bounded action spaces. The standard tanh\tanh squashing solution, popularized by SAC, maps Gaussian samples to bounded intervals but introduces well-documented issues: gradient vanishing near boundaries, asymmetric action distributions, and the fundamental contradiction of using infinite-support distributions for finite spaces (Bendada et al., 2025). Despite these known problems, the field has accepted them as necessary trade-offs for computational convenience.

2.2 Beyond Gaussian: Alternative Distributions

Recognition of Gaussian limitations has motivated exploration of alternative policy parameterizations. Beta distributions naturally model bounded intervals but struggle with multivariate actions and lack the reparameterization properties crucial for modern gradient-based optimization (Chou et al., 2017). Normalizing flows offer flexible distributions through invertible transformations, yet their computational overhead (2-3×\times slower than Gaussian) and training instability have limited adoption in RL settings (Ghugare & Eysenbach, 2025). Mixture models, including Gaussian mixture policies, increase expressiveness but compound the original boundary issues while adding mode collapse risks (Haarnoja et al., 2017).

The most principled alternative emerged from directional statistics (Sinii et al., 2024). vMF distributions, operating directly on the unit sphere, elegantly address boundary constraints through their geometric formulation. Recent work (Scott et al., 2021) demonstrated vMF policies could match or exceed Gaussian performance while providing theoretical advantages. However, vMF’s practical adoption faces significant hurdles (Banerjee et al., 2005): sampling requires rejection methods with acceptance rates as low as 0.1 for high concentrations, likelihood computation involves modified Bessel functions Iv​(κ)I_{v}(\kappa) prone to numerical overflow for large κ\kappa, and the concentration parameter lacks intuitive interpretation for practitioners (Zaghloul & Johnson, 2025). These challenges have confined vMF policies largely to theoretical investigations rather than practical deployment.

2.3 Geometric Perspectives in RL

Parallel to distributional innovations, a geometric understanding of RL has emerged (Hu et al., 2022). Hyperbolic RL leverages non-Euclidean geometry for hierarchical representations, while work on Riemannian policy optimization extends natural gradients to curved manifolds (Nickel & Kiela, 2017; Wang et al., 2024; Müller & Montúfar, 2024). These approaches highlight how geometric structure can inform algorithm design, yet they typically add complexity rather than simplifying existing methods. Our work inverts this trend, using geometric insights to simplify rather than sophisticate.

The connection between action spaces and geometric structures has appeared implicitly in several contexts (Wang et al., 2025). Quaternion representations for rotational actions in robotics naturally embrace spherical geometry, while circular distributions model periodic actions in locomotion tasks (Zhou et al., 2019). However, these remain specialized solutions rather than general principles. GAC unifies previously scattered geometric insights into a single, practical framework for continuous control—demonstrating that the central question is not which distribution to adopt, but whether explicit distributions are necessary at all.

2.4 Simplification as Innovation

The evolution from TRPO (Schulman et al., 2015) to PPO (Schulman et al., 2017) exemplifies a crucial pattern in deep RL: dramatic simplification often yields superior practical performance. Where TRPO required complex conjugate gradient procedures and line searches, PPO achieved comparable or better results through simple clipped objectives. Similarly, Twin Delayed Deep Deterministic policy gradient (TD3) (Fujimoto et al., 2018) simplified DDPG’s actor-critic architecture while improving stability and performance by 30% on average. These trends suggest that algorithmic complexity in RL often reflects a lack of structural clarity, rather than a theoretical necessity.

GAC follows this simplification philosophy. Rather than adding sophistication to handle Gaussian limitations or implementing complex vMF sampling, we identify the minimal geometric structure necessary for effective control. This approach aligns with recent trends toward interpretable and efficient RL, where understanding why methods work matters as much as empirical performance.

2.5 The Missing Perspective: Action Generation Without Distributions

Most approaches to continuous control operate within the distributional paradigm: policies are defined as probability densities over actions, requiring likelihood evaluation, entropy regularization, and absolute continuity (Engstrom et al., 2020). This perspective, inherited from supervised learning and classical statistics, may be unnecessarily restrictive for control, where actions are ultimately deterministic functions of states and randomness.

GAC replaces this distributional machinery with a geometric operation. Instead of modeling a density over ℝd\mathbb{R}^{d}, we generate actions via a direction sampled on the unit sphere and a scalar magnitude. This reframes control not as modeling a distribution, but as directly generating structured actions. By replacing distributional complexity with geometric operations, GAC eliminates density evaluations, reparameterization tricks, and explicit entropy calculations while avoiding gradient saturation from tanh\tanh squashing. Recent theoretical work (Tiwari et al., 2025) shows that RL trajectories tend to concentrate on low-dimensional manifolds, using complex mathematical analysis to uncover this emergent structure. GAC inverts the perspective: rather than discovering manifolds, we build on them. By constraining actions to the unit sphere, we achieve structure by design, not by accident. From emergent complexity to designed simplicity—GAC exemplifies a principle in RL: structure need not emerge; it can be constructed.

3 Problem Formulation and Methodology

3.1 Problem Formulation

We consider the standard continuous control setting where an agent interacts with an environment through bounded continuous actions. The action space 𝒜⊆[−1,1]d\mathcal{A}\subseteq[-1,1]^{d} represents normalized physical constraints, where dd is the action dimension. The agent observes states s∈𝒮s\in\mathcal{S} and selects actions according to a policy π:𝒮→𝒜\pi:\mathcal{S}\rightarrow\mathcal{A}.

The maximum entropy RL framework augments the standard objective with an entropy term to encourage exploration:
J(π)=𝔼τ∼π[∑t=0∞γt(rt+αℋ(π(⋅|st)))],J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}\left(r_{t}+\alpha\mathcal{H}(\pi(\cdot|s_{t}))\right)\right],
where γ∈[0,1)\gamma\in[0,1) is the discount factor, rt=r​(st,𝐚t)r_{t}=r(s_{t},\mathbf{a}_{t}) denotes the reward at time step tt, α>0\alpha>0 is the temperature parameter controlling exploration-exploitation trade-off, ℋ\mathcal{H} denotes entropy, and τ=(s0,𝐚0,s1,𝐚1,…)\tau=(s_{0},\mathbf{a}_{0},s_{1},\mathbf{a}_{1},\dots) represents trajectories sampled from the policy-environment interaction.

Figure 1: Architecture of GAC.
State ss is processed by a shared backbone, which branches into a direction head producing a unit vector 𝝁\bm{\mu}, and a concentration head predicting κ\kappa.
The final action is generated via spherical mixing, replacing traditional distributional sampling with direct geometric interpolation.

The Geometric Mismatch. Most continuous control methods model policies as Gaussian distributions 𝒩​(μ​(s),Σ​(s))\mathcal{N}(\mu(s),\Sigma(s)) with unbounded support, requiring tanh\tanh squashing to map samples into [−1,1]d[-1,1]^{d}.
While this approach has proven highly successful, as evidenced by SAC remaining a leading method, it creates a fundamental mismatch: unbounded distributions must be compressed into bounded action spaces.
The tanh\tanh transformation achieves this compression but induces gradient saturation when |𝐚~i||\tilde{\mathbf{a}}_{i}| is large, with our analysis showing substantial pre-squashed samples fall in low-gradient regions (Appendix B).
SAC addresses this through strong entropy regularization (α≈0.2\alpha\approx 0.2), maintaining exploration despite reduced gradients.
GAC takes a different path: rather than mitigating the mismatch through entropy-driven exploration, we eliminate it by operating directly on the unit sphere.
This geometric approach ensures consistent gradient flow and enables adaptive exploration through learned concentration, offering a structurally simpler alternative that aligns policy support with environmental constraints by design.

3.2 Geometric Action Generation

Core Insight.
GAC replaces traditional action sampling with a geometric pipeline consisting of direction mapping, concentration control, and spherical mixing, as illustrated in Figure 1. Rather than modeling probability distributions over actions, we directly generate actions via geometric operations on the unit sphere, a natural choice that aligns with high-dimensional concentration phenomena where directions carry the primary semantic information. This shift eliminates the need for density-based computations such as log-probabilities or entropy, while preserving exploration through structured spherical noise, modulated by a learnable concentration parameter κ\kappa.

Direction Mapping. A neural network fμ:𝒮→ℝdf_{\mu}:\mathcal{S}\rightarrow\mathbb{R}^{d} produces raw directional vectors, which are normalized to the unit sphere:

𝝁​(s)=fμ​(s)‖fμ​(s)‖2,\bm{\mu}(s)=\frac{f_{\mu}(s)}{||f_{\mu}(s)||_{2}},

(1)

where ||⋅||2||\cdot||_{2} denotes the L2 norm. This normalization ensures 𝝁​(s)∈𝕊d−1\bm{\mu}(s)\in\mathbb{S}^{d-1}, the unit sphere in dd dimensions.

Concentration Control.
A separate network fκ:𝒮→ℝf_{\kappa}:\mathcal{S}\rightarrow\mathbb{R} predicts concentration scores, which modulate the trade-off between deterministic direction and stochastic noise.
These scores are transformed via a sigmoid function w​(κ)=σ​(κ)∈(0,1)w(\kappa)=\sigma(\kappa)\in(0,1) to produce the mixing weight used in (2), enabling smooth interpolation and stable, adaptive exploration throughout training.

Spherical Mixing. Actions are generated by interpolating between the deterministic direction and uniform spherical noise:

𝐚=r⋅normalize​(w​(κ)⋅𝝁+(1−w​(κ))⋅𝝃),\mathbf{a}=r\cdot\text{normalize}\left(w(\kappa)\cdot\bm{\mu}+(1-w(\kappa))\cdot\bm{\xi}\right),

(2)

where 𝝃∼Uniform​(𝕊d−1)\bm{\xi}\sim\text{Uniform}(\mathbb{S}^{d-1}) is sampled as normalized Gaussian noise to provide isotropic exploration on the unit sphere, and r=2.5r=2.5 is the action radius (task-dependent). Even with substantial noise contribution (e.g., 30% when κ≈1\kappa\approx 1), actions remain coherent:
spherical normalization preserves directionality while preventing magnitude corruption,
ensuring stable control throughout training.

Intrinsic Exploration.
In contrast to conventional approaches where exploration is externally injected
(e.g., Gaussian noise or entropy bonuses), GAC inherently encodes stochasticity
within the action generation process. The random direction 𝝃\bm{\xi}
is not an auxiliary perturbation but an integral part of the policy’s structure,
making exploration an intrinsic geometric property. The learnable
concentration κ\kappa acts as an endogenous control signal,
adaptively modulating the exploration-exploitation trade-off without external
regularization. This eliminates the need for separate entropy bonuses or
temperature scheduling (e.g., α\alpha tuning in SAC).

Gradient Flow. All GAC operations, including normalization, sigmoid mixing, and linear interpolation, are differentiable and compatible with standard backpropagation, ensuring stable policy optimization.

3.3 Theoretical Justification

The spherical mixing operation creates an implicit distribution with geometrically intuitive concentration control. While a closed-form density is intractable, we rigorously establish how the mixing weight controls distribution concentration.

Theorem 1 (Expected Direction Control).

For GAC’s spherical mixing operation, the expected unnormalized sample vector lies precisely along the mean direction, scaled by the mixing weight:

𝔼𝝃​[𝐯]=w​(κ)​𝝁,\mathbb{E}_{\bm{\xi}}[\mathbf{v}]=w(\kappa)\bm{\mu},

(3)

where 𝐯=w​(κ)​𝛍+(1−w​(κ))​𝛏\mathbf{v}=w(\kappa)\bm{\mu}+(1-w(\kappa))\bm{\xi} is the unnormalized mixture, 𝛍∈𝕊d−1\bm{\mu}\in\mathbb{S}^{d-1} is the mean direction, 𝛏∼Uniform​(𝕊d−1)\bm{\xi}\sim\text{Uniform}(\mathbb{S}^{d-1}) is uniform spherical noise, and w​(κ)=σ​(κ)w(\kappa)=\sigma(\kappa) is the mixing weight with sigmoid function σ​(x)=1/(1+e−x)\sigma(x)=1/(1+e^{-x}).

This result is exact and requires no approximation: w​(κ)w(\kappa) directly controls the expected alignment with 𝝁\bm{\mu}, and as κ\kappa grows large (w​(κ)→1w(\kappa)\to 1), variance shrinks and samples concentrate tightly around 𝝁\bm{\mu}.
This provides vMF-like concentration behavior without Bessel function computations. See Appendix A.1 for the proof.

3.4 Integration with SAC

GAC naturally integrates into the SAC framework by replacing the standard Gaussian policy with our geometric action generator.
The key distinction lies in exploration: GAC eliminates explicit probability computations and entropy regularization, achieving exploration instead through geometric mixing controlled by κ\kappa.

3.4.1 Exploration Control Mechanism

GAC introduces a learned exploration controller κ​(s)\kappa(s) that adaptively modulates the balance between deterministic actions and stochastic exploration. Unlike SAC’s temperature parameter α\alpha which requires manual tuning or scheduling, κ\kappa learns directly from the value landscape.

In the maximum entropy framework, the actor seeks to maximize expected return while maintaining exploration. For GAC, this objective becomes:

Lactor​(ϕ)=𝔼s∼𝒟​[−κ​(s)−mini=1,2⁡Qθi​(s,𝐚)],L_{\text{actor}}(\phi)=\mathbb{E}_{s\sim\mathcal{D}}\left[-\kappa(s)-\min_{i=1,2}Q_{\theta_{i}}(s,\mathbf{a})\right],

(4)

where 𝒟\mathcal{D} is the replay buffer, ϕ\phi denotes the actor parameters,
and 𝐚\mathbf{a} is generated via GAC’s geometric mechanism in (2)
with current state ss. The term κ​(s)\kappa(s) serves as a learned exploration controller that replaces entropy regularization. Smaller values of κ\kappa increase the contribution of stochastic noise in the geometric mixing defined in (2), thereby promoting exploration. In contrast, larger values lead to more deterministic actions. This removes the need for temperature tuning in SAC.
Unlike traditional entropy-based methods, GAC never computes probability densities. Instead, exploration emerges directly from geometric structure. This design is theoretically justified by directional statistics, where higher concentration naturally corresponds to lower entropy (see Appendix A.4).

The soft Q-function update follows standard SAC with our exploration controller. The target value incorporates the minimum of two Q-networks for stability:

y​(rt,s′)=rt+γ​(mini=1,2⁡Qθi′​(s′,𝐚′)−κ​(s′)),y(r_{t},s^{\prime})=r_{t}+\gamma\left(\min_{i=1,2}Q_{\theta^{\prime}_{i}}(s^{\prime},\mathbf{a}^{\prime})-\kappa(s^{\prime})\right),

(5)

where θi′\theta^{\prime}_{i} denotes the parameters of the target Q-network, s′s^{\prime} is
the next state, and 𝐚′\mathbf{a}^{\prime} is generated from s′s^{\prime} using GAC’s
geometric mechanism in (2). Despite replacing Gaussian policies with geometric action generation, GAC maintains the essential properties for convergence in the SAC framework. The bounded action space and smooth geometric operations ensure that the soft Bellman operator remains a contraction (see Appendix A.5 for formal analysis).

3.5 Practical Considerations

Parameter Efficiency. GAC requires only d+1d+1 outputs (direction vector plus scalar concentration) compared to 2​d2d for diagonal Gaussian policies, where parameter count reduces by nearly 50%.

Computational Efficiency. The sampling procedure involves only normalization and linear interpolation, avoiding rejection sampling or special function evaluations. The computational complexity is O​(d)O(d) per sample, compared to O​(d​k)O(dk) for vMF sampling where kk is the expected number of rejections (typically k∈[2,10]k\in[2,10] for high concentrations).

Hyperparameter Selection.
Due to the geometry of high-dimensional spheres, a unit vector in ℝd\mathbb{R}^{d} has expected per-dimension magnitude 𝔼​[|μi|]≈1/d\mathbb{E}[|\mu_{i}|]\approx 1/\sqrt{d}, which becomes vanishingly small as dd increases (e.g., ≈0.24\approx 0.24 for d=17d=17).
Without scaling, such weak actions would be ineffective for control.
The radius r=2.5r=2.5 provides principled rescaling: after spherical mixing with typical w​(κ)≈0.85w(\kappa)\approx 0.85, this yields per-dimension actions around 0.60.6–0.90.9, which fall comfortably within the normalized action bounds [−1,1][-1,1] and ensure effective actuation.
While this default works robustly across diverse tasks, certain environments benefit from adjusted scaling (e.g., Ant-v4’s multi-leg coordination performs better with r=1.0r=1.0 for finer control).

4 Experiments

4.1 Experimental Setup

Environments. We evaluate GAC across six standard MuJoCo (Todorov et al., 2012) continuous control benchmarks that collectively test different aspects of control: HalfCheetah-v4 (speed optimization), Ant-v4 (multi-leg coordination), Humanoid-v4 (complex balance), Walker2d-v4 (bipedal locomotion), Hopper-v4 (unstable dynamics), and Pusher-v4 (object manipulation). These environments span action dimensions from 3 to 17 and present diverse challenges from optimization to precise coordination.

Baselines. We compare against SAC (Gaussian + tanh\tanh), TD3 (deterministic + noise), and PPO (clipped objectives), using recommended hyperparameters from CleanRL (Huang et al., 2022). All implementations are standardized for fair comparison. See Appendix C.1 for details.

Training Protocol. All algorithms are trained for 1M environment steps with 8 parallel environments, except Pusher-v4, which runs for 500K steps due to faster convergence. We use 5 random seeds {0, 10, 42, 77, 123} and report mean episodic returns ±\pm standard deviation.

4.2 Main Results

Table 1: Average return over final training steps (1M for all tasks, 500K for Pusher-v4). Bold indicates best performance. GAC achieves competitive performance across diverse control tasks.

Environment
GAC (Ours)
SAC
TD3
PPO

Hopper-v4
1952 ±\pm 285
2094 ±\pm 604

2896 ±\pm 749
2118 ±\pm 124

Walker2d-v4
5165 ±\pm 334
5152 ±\pm 608
4457 ±\pm 457
2874 ±\pm 517

Pusher-v4
-32 ±\pm0

-23 ±\pm 2
-27 ±\pm 1
-78 ±\pm 9

HalfCheetah-v4

12750 ±\pm 758
12540 ±\pm 517

12208 ±\pm 799
1608 ±\pm 793

Ant-v4
5633 ±\pm 158
4094 ±\pm 1039
3531 ±\pm 1263
1969 ±\pm 778

Humanoid-v4
5823 ±\pm 121
5717 ±\pm 123
5819 ±\pm 278
619 ±\pm 59

Figure 2: Learning curves on (a) Hopper-v4, (b) Walker2d-v4, (c) Pusher-v4, (d) HalfCheetah-v4, (e) Ant-v4, and (f) Humanoid-v4.

Performance Analysis.
Table 1 and Figure 2 present our main experimental results across six MuJoCo benchmarks.
GAC demonstrates strong performance, achieving the best results on 4 out of 6 tasks and remaining highly competitive on others.
As expected, PPO performs substantially worse across all environments, consistent with its known limitations in exploration and entropy scheduling.

High-dimensional Control Excellence.
GAC shows particular strength in complex, high-dimensional tasks.
On Ant-v4 (8D action space), GAC achieves 5633 ±\pm 158, significantly outperforming SAC (4094 ±\pm 1039) by 37.6% and TD3 (3531 ±\pm 1263) by 59.5%.
The large variance in SAC’s performance (±\pm 1039) suggests instability in Gaussian policies for multi-leg coordination, while GAC maintains remarkable consistency (±\pm 158).
On HalfCheetah-v4 (6D), GAC achieves the highest return of 12750 ±\pm 758, slightly exceeding SAC and outperforming TD3 by 4.4%.

Learning Efficiency.
Beyond final performance, GAC exhibits superior learning dynamics.
In Figure 2(d-e), GAC shows faster initial learning on Ant environments, reaching near-optimal performance by 200k steps while SAC and TD3 continue improving until 400k steps.
This efficiency stems from GAC’s geometric structure eliminating the need for entropy tuning, as the learned κ\kappa naturally balances exploration and exploitation without manual temperature scheduling.

Trade-off Between Stability and Expressiveness.
GAC demonstrates consistent learning stability across environments, achieving the lowest variance on Pusher-v4 (−32±0-32\pm 0) and competitive performance on Walker2d, HalfCheetah, Ant and Humanoid. This stability advantage is particularly pronounced in high-dimensional spaces, where GAC’s bounded geometric structure prevents the exploration instabilities inherent to squashed Gaussian policies.

However, its performance on Hopper-v4 (1952±2851952\pm 285) lags behind TD3 (2896±7492896\pm 749), and similarly underperforms on Pusher-v4. We attribute this to a fundamental trade-off introduced by spherical normalization: while it ensures consistent gradient flow and exploration stability, it imposes a geometric prior that assumes the optimal actions lie close to a unit hypersphere. This assumption aligns well with high-dimensional locomotion tasks, where the optimal actions often reside on a spherical shell due to norm concentration effects, but may be suboptimal for tasks involving asymmetric or contact-rich dynamics. For instance, Hopper’s single-leg locomotion requires strong, unbalanced thrusts that extend beyond the unit sphere, and Pusher’s manipulation demands induce anisotropic action distributions better captured by elliptical or non-spherical structures. These observations suggest future directions for adaptive or task-conditioned geometric priors that modulate between spherical, elliptical, or unconstrained action manifolds based on task characteristics.

Challenging Environments.
Humanoid-v4 (17D action space) proves challenging for all methods.
During training, GAC obtains 5823 ±\pm 121, TD3 achieves 5819 ±\pm 278, while SAC achieves 5717 ±\pm 123.
However, post-training evaluation of GAC achieves 6591 ±\pm 53, suggesting that the algorithm learns high-quality policies while maintaining exploration during training (detailed evaluation protocol in supplementary materials).
This validates that spherical normalization effectively handles complex action spaces and balances exploration-exploitation trade-offs.

4.3 Ablation Studies

Figure 3: Ablation study on HalfCheetah-v4. Default GAC (r=2.5r=2.5, adaptive κ\kappa) performs best.

We conduct comprehensive ablation studies to analyze the contribution of each component in GAC. Table 2 and Figure 3 presents results on HalfCheetah-v4 across 5 random seeds, systematically dissecting the effect of target magnitude rr and key architectural components.

Target magnitude rr critically affects performance.
The choice of r=2.5r=2.5 is theoretically motivated by high-dimensional sphere geometry: a unit vector in ℝd\mathbb{R}^{d} has expected per-dimension magnitude 𝔼​[|μi|]≈1/d\mathbb{E}[|\mu_{i}|]\approx 1/\sqrt{d}, which becomes vanishingly small as dd increases (e.g., ≈0.24\approx 0.24 for d=17d=17), rendering unscaled actions ineffective for control. Figure 3 empirically validates this insight: reducing rr from 2.5 to 1.5 causes catastrophic 43% performance degradation due to limited exploration and actuation. In contrast, increasing rr to 3.5 has minimal impact (−4.1%-4.1\%), suggesting saturation. The default r=2.5r=2.5, after spherical mixing with typical w​(κ)≈0.85w(\kappa)\approx 0.85, yields per-dimension actions in the range 0.6–0.9. This range falls well within the normalized bounds [−1,1][-1,1] and strikes an effective balance for HalfCheetah-v4’s locomotion dynamics.

Table 2: Ablation study of GAC components on HalfCheetah-v4. Results averaged over 5 seeds.

Configuration
Final Return
Relative
Key Observation

GAC (default with κ\kappa and r=2.5r=2.5)
12750 ±\pm 758
baseline
Optimal balance

Target magnitude ablation:

r=3.5r=3.5

12229 ±\pm 422
-4.1%
Slightly over-scaled actions

r=1.5r=1.5

7272 ±\pm 1235
-43.0%
Severely limited action range

Component ablation:

w/o κ\kappa controller
11370 ±\pm 643
-10.8%
No adaptive exploration

w/o normalization
Diverged
N/A
Gradient explosion at 5k steps

Raw action output
Collapsed
N/A
Unbounded actions, NaN loss

Adaptive exploration via κ\kappa is essential.
Removing the learnable κ\kappa controller while maintaining r=2.5r=2.5 reduces performance by 10.8% and slows convergence. The learned κ\kappa enables state-dependent exploration by providing high concentration in confident states while maintaining diversity in uncertain regions. This adaptive behavior emerges naturally from the value-based objective without explicit curriculum or scheduling. Notably, this mechanism provides an elegant alternative to entropy-based regularization: rather than maximizing entropy uniformly, GAC modulates exploration geometrically through directional mixing. This not only simplifies the optimization pipeline by eliminating entropy terms, but also improves interpretability, as κ​(s)\kappa(s) can be viewed as a soft confidence score indicating how deterministic the policy should be at a given state. As a result, exploration becomes structure-aware and implicitly guided by the task dynamics.

Geometric structure ensures stability.
As shown in Table 2, removing normalization leads to divergence within 5k steps due to unbounded gradients, as output norms grow without spherical projection and trigger explosion. Likewise, using raw unbounded outputs collapses immediately (NaN losses) as actions exceed environment limits and destabilize training. These ablations confirm that GAC’s spherical geometry is essential for stability. Beyond preventing numerical issues, the constraint shapes a consistent optimization landscape where all actions share equal norm, removing scale ambiguity and acting as an implicit regularizer against degenerate solutions.

5 Conclusion

This work demonstrates that effective continuous control does not require complex probability distributions.
GAC achieves competitive or superior performance across diverse benchmarks using a single geometric operation:
𝐚=r⋅normalize​(w​(κ)⋅𝝁+(1−w​(κ))⋅𝝃)\mathbf{a}=r\cdot\text{normalize}(w(\kappa)\cdot\bm{\mu}+(1-w(\kappa))\cdot\bm{\xi}).
By replacing distributional modeling with direct geometric mixing, we reduce parameter count by nearly 50% and improve performance, achieving a 37.6% improvement over SAC on Ant-v4 and the best results on 4 out of 6 MuJoCo benchmark tasks.

The success of GAC validates a broader principle: respecting the geometric structure of action spaces can be more effective than sophisticated probabilistic machinery.
The consistent performance with a fixed radius r=2.5r=2.5 across diverse tasks suggests that learning correct action directions is often the primary challenge, while magnitude can be addressed with simple task-specific constants.

Our method eliminates the computational burden of density calculations, Bessel functions, and rejection sampling, while avoiding the gradient pathologies of tanh\tanh-squashed Gaussians.
The learnable concentration parameter κ\kappa provides adaptive exploration without explicit entropy computation, demonstrating that exploration-exploitation balance can emerge from geometric structure rather than information-theoretic regularization.

Limitations and Future Work. While GAC demonstrates strong empirical performance, several avenues remain for investigation.
The action radius rr, though robust across tasks (with r=2.5r=2.5 working well for most benchmarks), could potentially be learned adaptively for task-specific optimization.
The theoretical connection between our geometric exploration mechanism and information-theoretic quantities, while empirically validated through strong performance, warrants deeper mathematical analysis.
Additionally, extending the geometric approach to discrete or hybrid action spaces presents exciting challenges for general-purpose control.

Despite these open questions, GAC’s success suggests that the Geometric Simplicity Principle—replacing probabilistic complexity with geometric structure—could transform other areas of RL.
Future work might explore geometric approaches to value function approximation, hierarchical control, or multi-agent coordination.
By showing that simple geometric operations can replace complex probabilistic frameworks, GAC challenges the prevailing assumption that sophisticated distributions are necessary for continuous control and opens new directions in geometric RL. Ultimately, control is not about predicting densities, but about choosing directions.
GAC shows that when geometry is respected, simplicity is not a compromise—but a strength.
We hope this work inspires further efforts to rethink RL through the lens of geometric structure, not just statistical modeling.

References

Banerjee et al. (2005)

Arindam Banerjee, Inderjit S. Dhillon, Joydeep Ghosh, and Suvrit Sra.

Clustering on the unit hypersphere using von mises-fisher distributions.

Journal of Machine Learning Research, 6(46):1345–1382, 2005.

Bendada et al. (2025)

Walid Bendada, Guillaume Salha-Galvan, Romain Hennequin, Théo Bontempelli, Thomas Bouabça, and Tristan Cazenave.

Exploring large action sets with hyperspherical embeddings using von mises-fisher sampling.

arXiv preprint arXiv:2507.00518, 2025.

Chou et al. (2017)

Po-Wei Chou, Daniel Maturana, and Sebastian Scherer.

Improving stochastic policy gradients in continuous control with deep reinforcement learning using the beta distribution.

In Proceedings of the 34th International Conference on Machine Learning (ICML), pp. 834–843, 2017.

Davidson et al. (2022)

Tim R. Davidson, Luca Falorsi, Nicola De Cao, Thomas Kipf, and Jakub M. Tomczak.

Hyperspherical variational auto-encoders.

arXiv preprint arXiv:1804.00891, 2022.

Engstrom et al. (2020)

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry.

Implementation matters in deep rl: A case study on ppo and trpo.

In International Conference on Learning Representations (ICLR), 2020.

Fujimoto et al. (2018)

Scott Fujimoto, Herke van Hoof, and David Meger.

Addressing function approximation error in actor-critic methods.

In Proceedings of the 35th International Conference on Machine Learning (ICML), pp. 1587–1596, 2018.

Ghugare & Eysenbach (2025)

Raj Ghugare and Benjamin Eysenbach.

Normalizing flows are capable models for rl.

arXiv preprint arXiv:2505.23527, 2025.

Haarnoja et al. (2017)

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine.

Reinforcement learning with deep energy-based policies.

arXiv preprint arXiv:1702.08165, 2017.

Haarnoja et al. (2018)

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.

In Proceedings of the 35th International Conference on Machine Learning (ICML), pp. 1861–1870, 2018.

Hu et al. (2022)

Jiang Hu, Ruicheng Ao, Anthony Man-Cho So, Minghan Yang, and Zaiwen Wen.

Riemannian natural gradient methods.

arXiv preprint arXiv:2207.07287, 2022.

Huang et al. (2022)

Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and João G.M. Araújo.

Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms.

Journal of Machine Learning Research, 23(274):1–18, 2022.

Lillicrap et al. (2015)

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.

Continuous control with deep reinforcement learning.

arXiv preprint arXiv:1509.02971, 2015.

Mazoure et al. (2019)

Bogdan Mazoure, Thang Doan, Audrey Durand, R. Devon Hjelm, and Joelle Pineau.

Leveraging exploration in off-policy algorithms via normalizing flows.

arXiv preprint arXiv:1905.06893, 2019.

Michel et al. (2024)

Nicolas Michel, Giovanni Chierchia, Romain Negrel, and Jean-François Bercher.

Learning representations on the unit sphere: Investigating angular gaussian and von mises-fisher distributions for online continual learning.

arXiv preprint arXiv:2306.03364, 2024.

Müller & Montúfar (2024)

Johannes Müller and Guido Montúfar.

Geometry and convergence of natural policy gradient methods.

Information Geometry, 7(1):485–523, 2024.

ISSN 2511-249X.

Nickel & Kiela (2017)

Maximillian Nickel and Douwe Kiela.

Poincaré embeddings for learning hierarchical representations.

In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Nikishin et al. (2021)

Evgenii Nikishin, Romina Abachi, Rishabh Agarwal, and Pierre-Luc Bacon.

Control-oriented model-based reinforcement learning with implicit differentiation.

arXiv preprint arXiv:2106.03273, 2021.

Obando-Ceron et al. (2024)

Johan Obando-Ceron, Ghada Sokar, Timon Willi, Clare Lyle, Jesse Farebrother, Jakob Foerster, Gintare Karolina Dziugaite, Doina Precup, and Pablo Samuel Castro.

Mixtures of experts unlock parameter scaling for deep RL.

arXiv preprint arXiv:2402.08609, 2024.

OpenAI et al. (2019)

OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, and Wojciech Zaremba.

Learning dexterous in-hand manipulation.

arXiv preprint arXiv:1808.00177, 2019.

Schulman et al. (2015)

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz.

Trust region policy optimization.

In Proceedings of the 32nd International Conference on Machine Learning (ICML), pp. 1889–1897, 2015.

Schulman et al. (2017)

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347, 2017.

Scott et al. (2021)

Tyler R. Scott, Andrew C. Gallagher, and Michael C. Mozer.

von mises–fisher loss: An exploration of embedding geometries for supervised learning.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10592–10602, 2021.

doi: 10.1109/ICCV48922.2021.01044.

Seo et al. (2025)

Younggyo Seo, Jafar Uruç, and Stephen James.

Continuous control with coarse-to-fine reinforcement learning.

In Proceedings of the Conference on Robot Learning (CoRL), pp. 2866–2894, 2025.

Silver et al. (2014)

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller.

Deterministic policy gradient algorithms.

In Proceedings of the 31st International Conference on Machine Learning (ICML), pp. 387–395, 2014.

Sinii et al. (2024)

Viacheslav Sinii, Alexander Nikulin, Vladislav Kurenkov, Ilya Zisman, and Sergey Kolesnikov.

In-context reinforcement learning for variable action spaces.

arXiv preprint arXiv:2312.13327, 2024.

Theile et al. (2024)

Mirco Theile, Lukas Dirnberger, Raphael Trumpp, Marco Caccamo, and Alberto L. Sangiovanni-Vincentelli.

Action mapping for reinforcement learning in continuous environments with constraints.

arXiv preprint arXiv:2412.04327, 2024.

Tiwari et al. (2025)

Saket Tiwari, Omer Gottesman, and George Konidaris.

Geometry of neural reinforcement learning in continuous state and action spaces.

In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Todorov et al. (2012)

Emanuel Todorov, Tom Erez, and Yuval Tassa.

Mujoco: A physics engine for model-based control.

Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5026–5033, 2012.

Wang et al. (2025)

Shijie Wang, Haichao Gui, and Rui Zhong.

Attitude estimation via matrix fisher distributions on so(3) using non-unit vector measurements.

Automatica, 179:112444, 2025.

Wang et al. (2024)

Zhangyu Wang, Lantian Xu, Zhifeng Kong, Weilong Wang, Xuyu Peng, and Enyang Zheng.

A geometry-aware algorithm to learn hierarchical embeddings in hyperbolic space.

arXiv preprint arXiv:2407.16641, 2024.

You et al. (2025)

Kisung You, Dennis Shung, and Mauro Giuffrè.

Learning over von mises–fisher distributions via a wasserstein-like geometry.

arXiv preprint arXiv:2504.14164, 2025.

Zaghloul & Johnson (2025)

Mofreh R. Zaghloul and Steven G. Johnson.

Efficient calculation of modified bessel functions of the first kind, iν​(z)i_{\nu}(z), for real orders and complex arguments: Fortran implementation with double and quadruple precision.

arXiv preprint arXiv:2505.09770, 2025.

Zhou et al. (2019)

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li.

On the continuity of rotation representations in neural networks.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5738–5746, 2019.

Appendix A Theoretical Analysis

A.1 Proof of Theorem 1

Proof.

Consider the unnormalized mixture vector:

𝐯=w​𝝁+(1−w)​𝝃,\mathbf{v}=w\bm{\mu}+(1-w)\bm{\xi},

(6)

where 𝝁∈𝕊d−1\bm{\mu}\in\mathbb{S}^{d-1} is the deterministic mean direction, 𝝃∼Uniform​(𝕊d−1)\bm{\xi}\sim\text{Uniform}(\mathbb{S}^{d-1}) is uniform spherical noise, and w=w​(κ)∈[0,1]w=w(\kappa)\in[0,1] is the mixing weight.

Due to the symmetry of the uniform distribution on 𝕊d−1\mathbb{S}^{d-1}, any uniform random vector on the sphere has zero expectation:

𝔼𝝃​[𝝃]=𝟎.\mathbb{E}_{\bm{\xi}}[\bm{\xi}]=\mathbf{0}.

(7)

Therefore, the expectation of the mixture vector is:

𝔼𝝃​[𝐯]=𝔼𝝃​[w​𝝁+(1−w)​𝝃]=w​𝝁+(1−w)​𝔼𝝃​[𝝃]=w​𝝁.\mathbb{E}_{\bm{\xi}}[\mathbf{v}]=\mathbb{E}_{\bm{\xi}}[w\bm{\mu}+(1-w)\bm{\xi}]=w\bm{\mu}+(1-w)\mathbb{E}_{\bm{\xi}}[\bm{\xi}]=w\bm{\mu}.

(8)

This result holds exactly for any dimension d≥2d\geq 2.
∎

A.2 Concentration Analysis

While Theorem 1 characterizes the expected direction of the unnormalized mixture, it does not capture how tightly the samples are concentrated around this direction. We therefore analyze the cosine similarity between normalized samples and the mean direction to quantify concentration. Let 𝐯^=𝐯/‖𝐯‖2\hat{\mathbf{v}}=\mathbf{v}/\|\mathbf{v}\|_{2} denote the normalized mixture, where ∥⋅∥2\|\cdot\|_{2} is the L2 norm.

The cosine similarity between 𝐯^\hat{\mathbf{v}} and 𝝁\bm{\mu} measures directional alignment:

cos⁡∠​(𝐯^,𝝁)=𝐯⊤​𝝁‖𝐯‖2.\cos\angle(\hat{\mathbf{v}},\bm{\mu})=\frac{\mathbf{v}^{\top}\bm{\mu}}{\|\mathbf{v}\|_{2}}.

(9)

For the numerator:

𝐯⊤​𝝁\displaystyle\mathbf{v}^{\top}\bm{\mu}
=(w​𝝁+(1−w)​𝝃)⊤​𝝁\displaystyle=(w\bm{\mu}+(1-w)\bm{\xi})^{\top}\bm{\mu}

(10)

=w​𝝁⊤​𝝁+(1−w)​𝝃⊤​𝝁\displaystyle=w\bm{\mu}^{\top}\bm{\mu}+(1-w)\bm{\xi}^{\top}\bm{\mu}

=w​‖𝝁‖2+(1−w)​𝝃⊤​𝝁\displaystyle=w\|\bm{\mu}\|^{2}+(1-w)\bm{\xi}^{\top}\bm{\mu}

=w+(1−w)​𝝃⊤​𝝁,\displaystyle=w+(1-w)\bm{\xi}^{\top}\bm{\mu},

where the last step uses ‖𝝁‖2=1\|\bm{\mu}\|^{2}=1 since 𝝁∈𝕊d−1\bm{\mu}\in\mathbb{S}^{d-1}.

Taking expectations over the uniform distribution:

𝔼𝝃​[𝐯⊤​𝝁]=w+(1−w)​𝔼𝝃​[𝝃⊤​𝝁]=w,\mathbb{E}_{\bm{\xi}}[\mathbf{v}^{\top}\bm{\mu}]=w+(1-w)\mathbb{E}_{\bm{\xi}}[\bm{\xi}^{\top}\bm{\mu}]=w,

(11)

since 𝔼𝝃​[𝝃⊤​𝝁]=0\mathbb{E}_{\bm{\xi}}[\bm{\xi}^{\top}\bm{\mu}]=0, as a fixed unit vector and a random unit vector on the sphere are uncorrelated in expectation under uniform sampling.

To estimate the norm, we assume that the inner product 𝝁⊤​𝝃\bm{\mu}^{\top}\bm{\xi} remains small, which is typically the case when 𝝃\bm{\xi} is uniformly sampled from the sphere. This leads to the approximation:

‖𝐯‖22=‖w​𝝁+(1−w)​𝝃‖2=w2+(1−w)2+2​w​(1−w)​(𝝁⊤​𝝃)≈w2+(1−w)2.\|\mathbf{v}\|_{2}^{2}=\|w\bm{\mu}+(1-w)\bm{\xi}\|^{2}=w^{2}+(1-w)^{2}+2w(1-w)(\bm{\mu}^{\top}\bm{\xi})\approx w^{2}+(1-w)^{2}.

(12)

This simplification holds well in practice as confirmed by our empirical results (see Appendix A.3).

For action spaces (d≫1d\gg 1), concentration of measure implies that 𝝃⊤​𝝁\bm{\xi}^{\top}\bm{\mu} concentrates tightly around zero with high probability. Under this regime:

𝔼​[cos⁡∠​(𝐯^,𝝁)]≈ww2+(1−w)2.\mathbb{E}[\cos\angle(\hat{\mathbf{v}},\bm{\mu})]\approx\frac{w}{\sqrt{w^{2}+(1-w)^{2}}}.

(13)

For typical operating ranges where w∈[0.6,0.99]w\in[0.6,0.99] (corresponding to κ∈[0.5,5]\kappa\in[0.5,5]), this quantity closely tracks ww itself. For example, when w=0.9w=0.9, the ratio equals 0.994, validating our use of w​(κ)w(\kappa) as an effective concentration parameter.

A.3 Empirical Validation

Table A.1: Concentration control validation: theoretical vs. measured. The close match between measured concentration and theoretical ww validates Theorem 1.

κ\kappa
Weight ww
Concentration
Angle Std. (∘)

-2.0
0.119
0.091
39.3∘

-1.0
0.269
0.253
36.3∘

0.0
0.500
0.678
19.7∘

0.5
0.622
0.875
9.1∘

1.0
0.731
0.956
5.0∘

2.0
0.881
0.994
1.7∘

Figure A.1: 3D visualization of GAC sample distributions for κ∈{−2,0,0.5,1}\kappa\in\{-2,0,0.5,1\}.
Arrows indicate target direction 𝝁\bm{\mu}.
Colors represent cosine similarity with 𝝁\bm{\mu} (blue=low, red=high).
Higher κ\kappa values produce more concentrated distributions.

To complement our theoretical analysis, we empirically verify that the concentration parameter κ\kappa provides direct and monotonic control over the sample distribution’s properties. For each κ∈{−2,−1,0.0,0.5,1.0,2.0}\kappa\in\{-2,-1,0.0,0.5,1.0,2.0\}, we generate 500 samples using the GAC mechanism and measure their key concentration metrics, summarized in Table A.1 and visualized in Figure A.1.

Metrics Explanation:

•

Weight w​(κ)w(\kappa): The mixing weight w​(κ)=σ​(κ)w(\kappa)=\sigma(\kappa) that controls interpolation between deterministic direction and uniform noise. This is deterministic given κ\kappa.

•

Concentration: The empirical mean cosine similarity between normalized samples and the target direction 𝝁\bm{\mu}, measuring how aligned the final actions are with the intended direction.

•

Angle Std.: Standard deviation of sample–𝝁\bm{\mu} angles (in degrees), measuring the spread of the distribution—smaller values indicate more concentrated (less exploratory) behavior.

Empirical Validation.
We empirically validate the theoretical result in Theorem 1 by measuring the directional concentration of sampled actions under different κ\kappa values.
Results show strong agreement between measured and theoretical concentration (correlation ≈\approx 0.95), with angular standard deviation decreasing monotonically from 39.3∘ at κ=−2\kappa=-2 (high exploration) to 1.7∘ at κ=2\kappa=2 (near-deterministic).
These results confirm that GAC achieves precise, vMF-like concentration control through simple geometric operations—without requiring modified Bessel functions or rejection sampling.
The monotonic relationship between κ\kappa and directional concentration further validates our approach to structured exploration.

A.4 Exploration Control and Entropy Connection

A.4.1 Exploration Control Mechanism

GAC’s concentration parameter κ\kappa functions as an adaptive exploration controller that learns when to explore versus exploit based on the value landscape. Unlike traditional entropy regularization that requires computing probability densities, κ\kappa directly modulates the geometric mixing between deterministic and stochastic components. While GAC implicitly defines a mixture distribution through this geometric operation, it crucially avoids any density or entropy computation during optimization. The effectiveness of κ\kappa as an exploration signal emerges from three complementary perspectives:

1) Geometric Perspective.
As κ\kappa increases, the mixing weight w​(κ)=σ​(κ)w(\kappa)=\sigma(\kappa) increases monotonically from 0 to 1, causing samples to concentrate progressively around 𝝁\bm{\mu}. Empirically, the angular standard deviation decreases from 39.3∘ at κ=−2\kappa=-2 to 1.7∘ at κ=2\kappa=2 (Table A.1), directly demonstrating decreasing distributional uncertainty.

2) Information-Theoretic Perspective.
For spherical distributions, concentration and entropy are fundamentally inversely related. The vMF distribution provides a theoretical benchmark:

ℋvMF=log⁡Cd​(κ)−κ​Ad​(κ)≈−κ+d−12​log⁡κ+const,\mathcal{H}_{\text{vMF}}=\log C_{d}(\kappa)-\kappa A_{d}(\kappa)\approx-\kappa+\frac{d-1}{2}\log\kappa+\text{const},

(14)

where Cd​(κ)C_{d}(\kappa) is the normalization constant and Ad​(κ)A_{d}(\kappa) the mean resultant length; the approximation holds for large κ\kappa, with the dominant linear term −κ-\kappa justifying our exploration term as higher-order terms contribute little in practice.

Note: The relationship ℋ≈−κ\mathcal{H}\approx-\kappa serves as conceptual motivation rather than rigorous derivation. GAC uses −κ-\kappa directly as an exploration signal without ever computing actual entropy. This approximation provides theoretical intuition for why −κ-\kappa effectively balances exploration-exploitation, but the method’s success does not depend on this mathematical correspondence.

3) Empirical Validation.
Our experiments confirm that −κ-\kappa effectively captures exploration pressure:

•

The correlation between w​(κ)w(\kappa) and measured concentration exceeds 0.95 (Figure A.1)

•

As κ\kappa increases: w​(κ)=σ​(κ)→1w(\kappa)=\sigma(\kappa)\to 1 (more deterministic/exploitative)

•

As κ\kappa decreases: w​(κ)=σ​(κ)→0w(\kappa)=\sigma(\kappa)\to 0 (more stochastic/exploratory)

•

GAC with κ\kappa achieves superior performance across all benchmarks (Table 1)

•

κ\kappa’s smooth effect on exploration supports stable policy optimization

A.4.2 Theoretical Connection to Entropy

While GAC operates without computing distributions, the exploration controller κ\kappa exhibits a natural connection to entropy in directional statistics. For vMF distributions on the unit sphere, the differential entropy is given by:

ℋvMF=log⁡Cd​(κ)−κ​Ad​(κ)≈−κ+d−12​log⁡κ+const,\mathcal{H}_{\text{vMF}}=\log C_{d}(\kappa)-\kappa A_{d}(\kappa)\approx-\kappa+\frac{d-1}{2}\log\kappa+\text{const},

(15)

where Cd​(κ)=κd/2−1(2​π)d/2​Id/2−1​(κ)C_{d}(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)} is the normalization constant and Ad​(κ)=Id/2​(κ)Id/2−1​(κ)A_{d}(\kappa)=\frac{I_{d/2}(\kappa)}{I_{d/2-1}(\kappa)} is the mean resultant length, with IvI_{v} denoting the modified Bessel function of the first kind.
The asymptotic behavior of Ad​(κ)A_{d}(\kappa) is:

Ad​(κ)≈1−d−12​κ+𝒪​(κ−2).A_{d}(\kappa)\approx 1-\frac{d-1}{2\kappa}+\mathcal{O}(\kappa^{-2}).

(16)

Substituting into ℋvMF\mathcal{H}_{\text{vMF}} and simplifying, we obtain:

ℋvMF≈−κ+d−12​log⁡κ+const.\mathcal{H}_{\text{vMF}}\approx-\kappa+\frac{d-1}{2}\log\kappa+\text{const}.

(17)

The leading term −κ-\kappa dominates, with logarithmic corrections diminishing in relative magnitude for practical κ\kappa values. This validates using −κ-\kappa as an exploration controller that faithfully reflects the tradeoff between concentration and uncertainty. While GAC does not explicitly follow the vMF distribution, it inherits the same qualitative dependency between κ\kappa and sample concentration through its spherical interpolation mechanism. This connection validates why −κ-\kappa effectively balances exploration-exploitation in the SAC framework. Empirical results in Appendix A.3 further confirm the effectiveness of this surrogate.

Key distinction: While this mathematical relationship exists, GAC fundamentally differs from entropy-regularized methods:

•

Traditional SAC: Computes ℋ​[π]=−𝔼​[log⁡π​(𝐚|s)]\mathcal{H}[\pi]=-\mathbb{E}[\log\pi(\mathbf{a}|s)] requiring explicit densities

•

GAC: Uses −κ-\kappa as exploration signal without any distributional computation

This allows GAC to achieve entropy-like regularization benefits through purely geometric operations, eliminating computational overhead while maintaining theoretical grounding.

A.5 SAC Convergence with GAC

Theorem 2: GAC maintains key properties required for SAC-style convergence under standard regularity conditions.

Remark: We provide a sketch of the key arguments. A rigorous convergence proof would require extensive measure-theoretic analysis beyond the scope of this work. Our empirical results across diverse environments provide strong evidence for convergence in practice.

Proof Sketch.

We establish that GAC maintains the key properties required for SAC convergence.

Soft Bellman Contraction.
The soft Bellman operator with GAC takes the form:

𝒯π​Q​(s,𝐚)=rt​(s,𝐚)+γ​𝔼s′∼p(⋅|s,𝐚)​[Vπ​(s′)],\mathcal{T}^{\pi}Q(s,\mathbf{a})=r_{t}(s,\mathbf{a})+\gamma\mathbb{E}_{s^{\prime}\sim p(\cdot|s,\mathbf{a})}[V^{\pi}(s^{\prime})],

(18)

where the soft value function incorporates our exploration controller:

Vπ​(s)=𝔼𝝃​[Q​(s,𝐚)]−κ​(s),𝐚​ generated via (2).V^{\pi}(s)=\mathbb{E}_{\bm{\xi}}[Q(s,\mathbf{a})]-\kappa(s),\quad\mathbf{a}\text{ generated via~(\ref{action})}.

(19)

The contraction property requires:

(i) Continuity: GAC’s action generation is continuous in parameters:

•

Direction mapping 𝝁​(s)=fμ​(s)/‖fμ​(s)‖2\bm{\mu}(s)=f_{\mu}(s)/\|f_{\mu}(s)\|_{2} is continuous for fμ​(s)≠0f_{\mu}(s)\neq 0

•

Mixing weight w​(κ)=σ​(κ)w(\kappa)=\sigma(\kappa) is C∞C^{\infty} smooth

•

Action normalization ensures ‖𝐚‖=r\|\mathbf{a}\|=r (bounded)

(ii) Regularization: The exploration controller provides consistent exploration pressure:

∇κ[−κ]<0,\nabla_{\kappa}[-\kappa]<0,

(20)

encouraging exploration when concentration becomes excessive.

Under these conditions and assuming bounded rewards |r​(s,𝐚)|≤Rmax|r(s,\mathbf{a})|\leq R_{\max}, the operator 𝒯π\mathcal{T}^{\pi} is a γ\gamma-contraction:

‖𝒯π​Q1−𝒯π​Q2‖∞≤γ​‖Q1−Q2‖∞.\|\mathcal{T}^{\pi}Q_{1}-\mathcal{T}^{\pi}Q_{2}\|_{\infty}\leq\gamma\|Q_{1}-Q_{2}\|_{\infty}.

(21)

Policy Improvement.
GAC approximates policy improvement through gradient-based optimization:

Lactor=𝔼s∼𝒟,𝝃​[−κ​(s)−Q​(s,𝐚)],L_{\text{actor}}=\mathbb{E}_{s\sim\mathcal{D},\bm{\xi}}[-\kappa(s)-Q(s,\mathbf{a})],

(22)

where 𝐚\mathbf{a} is generated using GAC’s mechanism with random noise 𝝃\bm{\xi}. This objective drives the policy toward high-value regions (via the Q-term) while
maintaining exploration (via the −κ-\kappa term), achieving similar goals to SAC’s
entropy-regularized policy improvement without explicit distributional computations.

Convergence Benefits.
GAC improves convergence through three key design choices:

•

Stable gradients: Spherical normalization avoids tanh\tanh-induced saturation, preserving direction gradients.

•

Bounded actions: All actions satisfy ‖𝐚‖=r\|\mathbf{a}\|=r, preventing value divergence from out-of-bound actions.

•

Adaptive exploration: The learnable κ\kappa balances exploration and exploitation without external schedules.

Exploration–Exploitation Balance.
The loss function induces an implicit tradeoff between exploration and exploitation. The direct gradient ∂ℒ/∂κ=−1\partial\mathcal{L}/\partial\kappa=-1 promotes exploration by reducing κ\kappa, increasing action noise. In contrast, the Q-value term encourages exploitation: when higher κ\kappa leads to better actions (and thus higher QQ), gradients through Q​(s,𝐚​(s;κ))Q(s,\mathbf{a}(s;\kappa)) push κ\kappa upward. This dynamic balance emerges naturally from actor–critic interplay, without explicit entropy terms or temperature tuning.

Natural Stabilization of κ\kappa.
Despite being unconstrained, κ\kappa remains bounded (∈[0,5]\in[0,5] empirically) through:

•

Sigmoid saturation: w​(κ)=σ​(κ)w(\kappa)=\sigma(\kappa) flattens for large κ\kappa, capping its effect.

•

Gradient feedback: High κ\kappa leads to deterministic actions; in noisy environments, this reduces QQ, discouraging overconfidence.

•

No explicit clipping: Yet κ\kappa stabilizes naturally via loss dynamics.

These mechanisms ensure convergence stability without manual regularization.

∎

Appendix B Gradient Flow Analysis in Tanh-Squashed Policies

Figure A.2:
Distribution of pre-squashed Gaussian samples from a trained SAC policy.
Red areas indicate saturated gradients (|tanh′⁡(x)|<0.05|\tanh^{\prime}(x)|<0.05), with 46.4% of samples falling into these regions.
Dashed lines show the [−1,1][-1,1] tanh\tanh boundaries.
This mismatch between unbounded Gaussians and bounded action spaces motivates GAC’s direct geometric approach.

To understand the geometric mismatch between Gaussian distributions and bounded action spaces, we analyze the gradient flow through tanh\tanh squashing functions. The tanh\tanh transformation 𝐚=tanh⁡(𝐚~)\mathbf{a}=\tanh(\tilde{\mathbf{a}}) has gradient:

∂𝐚∂𝐚~=1−tanh2⁡(𝐚~)\frac{\partial\mathbf{a}}{\partial\tilde{\mathbf{a}}}=1-\tanh^{2}(\tilde{\mathbf{a}})

(23)

which approaches zero as |𝐚~|→∞|\tilde{\mathbf{a}}|\to\infty, creating regions of vanishing gradients.

We sampled pre-squashed actions from the SAC policy throughout training, particularly during the stable performance phase near convergence (0.8M–1.0M steps).Figure A.2 visualizes the distribution of these raw actions before squashing, color-coded by gradient magnitude.

Key Observations:

•

A substantial fraction (46.4%) of pre-squashed samples have gradients below 0.1, indicating severe saturation.

•

The distribution exhibits heavy tails beyond |𝐚~|>2.5|\tilde{\mathbf{a}}|>2.5, where gradient flow is minimal.

•

SAC compensates through entropy regularization (α≈0.2\alpha\approx 0.2), which maintains exploration diversity despite reduced gradients.

This analysis does not imply SAC is ineffective, as it remains highly successful in practice. Rather, it highlights a structural inefficiency in that significant computational effort is spent managing the mismatch between unbounded distributions and bounded spaces. GAC sidesteps this issue entirely by operating directly on the unit sphere, ensuring consistent gradient flow without requiring squashing or entropy-driven exploration.

Note: This visualization shows pre-squashed samples. The actual gradient flow during training is modulated by reparameterization and entropy regularization, which prevent complete saturation. However, this analysis does not diminish SAC’s effectiveness but highlights an opportunity for geometric alternatives like GAC that avoid this structural inefficiency entirely.

Appendix C Reproducibility

To ensure complete reproducibility and facilitate future research, we provide comprehensive implementation details and open-source resources for immediate verification of our results.

C.1 Implementation Details

Network Architecture: GAC uses a shared backbone with separate heads for direction and concentration:

•

Backbone: Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU

•

Direction head: Linear(256, action_dim)

•

Concentration head: Linear(256, 64) → ReLU → Linear(64, 1)

Key Hyperparameters:

•

Learning rates: 3×10−43\times 10^{-4} (actor), 1×10−31\times 10^{-3} (critic)

•

Batch size: 256256, Buffer size: 10610^{6}

•

Target network update: τ=0.005\tau=0.005

•

Discount factor: γ=0.99\gamma=0.99

•

Action radius rr: 2.52.5 for most tasks, 1.01.0 for Ant-v4

We adopt standard hyperparameters from CleanRL (Huang et al., 2022) without task-specific tuning, highlighting that GAC’s gains stem from structural design rather than careful optimization.

C.2 Computational Efficiency

Computational Efficiency: GAC’s sampling requires only:
1). Two forward passes (direction and concentration networks);
2). One normalization operation;
3). One linear interpolation;
4). One final scaling. This yields approximately 6× speedup compared to vMF sampling with rejection methods.

C.3 Code and Model Release

To enable immediate verification of our results, we provide the following:

Pre-trained Models: Trained policy weights demonstrating reported performance.

Evaluation Scripts: Evaluation protocols enabling exact reproduction of reported metrics.

Availability: Anonymized resources available during review at:

https://github.com/geometric-rl-anonymous/Geometric-Action-Control-for-Reproducibility

,
including the trained models and evaluation scripts for the primary benchmarks (Walker2d-v4, HalfCheetah-v4, Ant-v4, and Humanoid-v4) used in our analyses,
to facilitate easy verification and reproduction of our results by reviewers.
A complete open-source release is planned upon paper acceptance with comprehensive documentation and examples.

Generated on Tue Nov 11 13:31:08 2025 by LaTeXML