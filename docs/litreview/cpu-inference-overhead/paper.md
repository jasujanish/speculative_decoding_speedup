Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference

Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub
Submit in GitHub

Back to arXiv

Why HTML?

Report Issue

Back to Abstract

Download PDF

Abstract

I Introduction

II Background & Motivation

II-A CPU’s Job in LLM Inference

II-B Real-World CPU Under-Provisioning in HPC Clusters

III Multi-GPU System Evaluation Setup

IV CPU Bottleneck in LLM Inference

IV-A Tokenization Latency Evaluation in LLM Inference

IV-B Impact of Tokenization Load in LLM Serving

V Understanding the CPU Bottlenecks in Multi-GPU Systems

V-A Synchronization and CPU Oversubscription in Communication Kernels

V-B Shared Memory Broadcast Contention

VI Discussion

VI-A CPU Under-Provisioning in Cloud Compute Platforms

VI-B Emerging Trends That May Intensify CPU Bottlenecks

VI-C Limitations

VII Related Work

VII-A CPU Overhead in Agentic AIs

VII-B GPU-Driven Control and Device-Initiated Communications

VII-C LLM Serving Frameworks

VII-D Mitigating the CPU Overhead in LLMs

VII-E Mitigating Kernel Launch Overhead

VIII Conclusion

References

License: arXiv.org perpetual non-exclusive license

arXiv:2603.22774v1 [cs.AR] 24 Mar 2026

Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference

Euijun Chung,
Yuxiao Jia,
Aaron Jezghani,
Hyesoon Kim

Abstract

Large-scale machine learning workloads increasingly rely on multi-GPU systems, yet their performance is often limited by an overlooked component: the CPU.
Through a detailed study of modern large language model (LLM) inference and serving workloads, we find that multi-GPU performance frequently degrades not because GPUs are saturated, but because CPUs fail to keep the GPUs busy.
Under limited CPU allocations, systems exhibit symptoms such as delayed kernel launch, stalled communication, and increased tokenization latency, leading to severe GPU underutilization even when ample GPU resources are available.
This work presents a systematic analysis of CPU-induced slowdowns in multi-GPU LLM inference. We show that these bottlenecks persist even in serving stacks that employ process-level separation and modern GPU-side optimizations such as CUDA Graphs.
Since the marginal cost of additional CPU cores is small relative to GPU instance pricing, our evaluation indicates that increasing the number of CPU cores can substantially improve performance and stability at minimal additional cost.
Under moderate serving load, we observe that CPU-starved configurations frequently time out, while providing adequate CPU resources restores responsiveness and reduces time-to-first-token (TTFT) latency by 1.36–5.40×\times across configurations, all without requiring additional GPUs.
This work shows that CPU provisioning is a crucial factor in multi-GPU LLM inference configuration, helping prevent control-side bottlenecks.

I Introduction

Modern machine learning (ML) workloads require massive computational power, driving the widespread adoption of multi-GPU servers, such as the DGX H100 and DGX B200 [32, 33]. GPU servers are expensive to purchase and operate, requiring substantial electricity and cooling infrastructure. To ensure efficient use of valuable resources, organizations operate them in a shared, multi-tenant manner, allowing multiple users to work on their own allocated virtual instances. Through resource schedulers like Slurm or cloud platforms (e.g., AWS and Azure), users request manual and static allocations (e.g., 4 GPUs and 8 CPU cores) at a given cost per unit time [48, 2, 53]. While this approach improves GPU utilization across many users, it creates a critical blind spot: the CPU is often underallocated to users. In this paper, we reveal that this imbalance between CPU and GPU in such scenarios can silently make the CPU a significant bottleneck in multi-GPU systems.

In multi-GPU ML workloads, contrary to the common assumption that GPUs are the primary bottleneck, CPUs perform numerous critical tasks, and systems with limited CPU resources can experience severe slowdowns when CPUs are oversubscribed. The CPU’s tasks include data I/O and processing, inter-GPU synchronization, request scheduling, and kernel launches [31, 25, 26]. A significant portion of a CPU’s job involves feeding sufficient data to GPUs, i.e., retrieving data from host memory and performing preprocessing steps such as tokenization for language models [14], or image decoding and augmentation for vision models [21, 35, 55, 49]. Such input-processing stages demand substantial CPU computation, especially for long input sequences and large batched inputs in LLM serving. To achieve the required performance, these stages typically rely on multithreaded or multi-process implementations, such as tokenizer implementations in Rust [15] or PyTorch’s Dataloader with multiple workers.

Figure 1: Overview of the CPU’s jobs in multi-GPU LLM serving. Multithreaded input processing and per-GPU host processes prepare tokenized inputs, manage kernel launches, and handle synchronization.

Moreover, CPUs are responsible for coordinating computation and communication across multiple GPUs by issuing kernel launches, managing stream dependencies, and orchestrating synchronization. Modern ML frameworks assign one process per GPU, and optimized inference systems spawn additional host processes for scheduling and request management [23, 54].

However, once such processes contend for limited CPU cores, oversubscription increases host-side latency and slows the launch of both compute and communication kernels. This behavior is especially pronounced during LLM serving with multiple incoming requests, where a high CPU load from these requests can substantially increase kernel launch latency from microseconds to milliseconds, thereby degrading overall performance and risking SLO (service-level objective) violations. Strong single-core CPU performance is necessary to minimize kernel launch overhead, emphasizing the CPU’s crucial role as a coordinator in multi-GPU workloads [57].

Collective communication libraries, such as NCCL [31], can lead to additional slowdowns in scenarios with limited CPU resources, as their collectives require all participating GPUs to synchronize. If the CPU is late to dispatch on any rank, the entire device group stalls. While CUDA Graphs [30] amortize launch overhead for static kernel sequences, they cannot capture dynamic scheduling decisions required at every decode step of LLM inference [7], leaving CPU-side bottlenecks as a persistent limiting factor.

The goal of this work is to challenge a common misconception among multi-GPU system users: the assumption that GPU performance alone determines the efficiency of modern ML workloads. While prior studies have noted that adequate CPU performance is necessary to keep GPUs saturated, such as data augmentation in vision training pipelines [29, 20], to our knowledge, detailed characterizations of CPU bottlenecks specifically in multi-GPU LLM inference workloads remain limited. Moreover, although practitioners and recent reports [39, 57] informally acknowledge the importance of CPU performance in multi-GPU systems, the community lacks clear guidance on identifying CPU issues, resource allocation, and understanding why inadequate CPU capacity causes substantial slowdowns.

To better understand CPU bottlenecks in multi-GPU LLM workloads, we designed and evaluated the system’s performance under varying CPU resource configurations in LLM inference and online serving scenarios (Section IV). Across these experiments, we find that the CPU can dominate end-to-end latency when it contends for limited CPU cores. By demonstrating that performance can be substantially improved by allocating more CPU resources, we claim that a multi-GPU system’s performance depends not only on accelerator compute capacity but also on appropriately balanced CPU provisioning.

Furthermore, Section V analyzes why these slowdowns occur, identifying two key factors: \scriptsize{1}⃝ compounded contention between CPU overheads and communication coordination, which leaves GPUs idle, and \scriptsize{2}⃝ contention on the shared-memory broadcast queue used for inter-process coordination, which introduces additional delays on the critical path even under lock-free designs. Section VI evaluates how frequently such slowdowns arise in real deployments, using production resource-allocation logs to illustrate the prevalence of CPU underprovisioning across multi-tenant clusters. Section VII reviews prior work on CPU overhead in emerging ML and LLM workloads, as well as existing mitigation techniques, and Section VIII concludes.

Our contributions are as follows:

•

We present a detailed characterization of CPU-intensive components in multi-GPU LLM inference and serving, and quantify how they induce CPU-side bottlenecks that propagate into end-to-end performance degradation.

•

We design and evaluate LLM-serving experiments that illustrate how CPU tokenization overhead can degrade end-to-end performance across the system and demonstrate that additional CPU resources effectively mitigate this slowdown.

•

We identify the root causes of CPU slowdowns, showing how CPU oversubscription compounds with barrier-based GPU synchronization to leave GPUs idle, and identify contention on the lock-free shared-memory broadcast queue. This structural bottleneck scales with the degree of tensor parallelism.

•

We analyze real-world cluster allocation logs to demonstrate the prevalence of CPU under-provisioning, and provide practical insights for CPU resource allocation in modern multi-GPU LLM systems.

II Background & Motivation

II-A CPU’s Job in LLM Inference

In modern LLM serving systems, the CPU is responsible for several latency-sensitive operations that directly influence end-to-end performance. As shown in Figure 1, the CPU’s job in LLM inference includes: \scriptsize{1}⃝ processing and tokenizing input text, \scriptsize{2}⃝ managing HTTP request handling and scheduling, and \scriptsize{3}⃝ issuing GPU kernel launches to sustain steady execution on the accelerators. Although GPUs perform the bulk of the model’s numerical computation, the CPU remains responsible for coordinating each step of computation, and limitations in these control-plane tasks can become the dominant bottleneck in multi-GPU inference.

Figure 2: Overview of the LLM inference pipeline, illustrating the CPU-intensive tokenization stage and GPU-intensive model computation stage.

\scriptsize{1}⃝Tokenization adds a significant amount of work to the CPU during LLM inference. As shown in Figure 2, tokenization converts raw text strings into integer token IDs that the model can process [44]. This process occurs at the beginning of every inference request for prompt processing, while the reverse operation (detokenization) converts generated token IDs back to text during output. The tokenizers apply subword segmentation algorithms, such as byte pair encoding (BPE) [44] or SentencePiece [22], which map segments to vocabulary entries and format the output as tensor inputs. Tokenization consumes substantial CPU cycles, particularly for long prompts or batched requests, where preprocessing costs scale linearly with the input size. Because tokenization is a prerequisite to any LLM computation, it lies on the critical path and directly influences end-to-end latency.

Widely used tokenizer implementations employ multiprocessing or multithreading to amortize the overhead of computationally intensive text processing. In particular, the HuggingFace Tokenizers library [15] enables its Rust-based tokenizer to spawn multiple parallel threads by default, setting TOKENIZERS_PARALLELISM=true. This design accelerates subword segmentation and Unicode parsing by utilizing several CPU cores, but it also increases contention when many requests are processed concurrently.

\scriptsize{2}⃝HTTP server request handling also adds CPU load through connection management, request parsing, and batched scheduling. This type of overhead scales with the query rate rather than with the model size, and it is typically much smaller than the tokenization cost unless query rates are very high, such as 500 requests per second. For this reason, our analysis focuses primarily on tokenization overhead and its impact on CPU contention during LLM inference.

\scriptsize{3}⃝Kernel launch overhead represents another source of CPU work. The CPU must schedule and dispatch CUDA kernels for each model layer. Physically, each launch traverses the CUDA runtime and driver stack, culminating in a PCIe MMIO write to the GPU’s doorbell register that signals the command processor to fetch new work. If the host thread responsible for this write is delayed by OS scheduling or CPU contention, the GPU idles until the doorbell is triggered, and delays in this process slow the entire pipeline. CUDA Graphs [30] can avoid most repeated launch calls by capturing and replaying complete computation sequences. However, CUDA Graphs cannot merge all kernel calls into a single operation, as long as certain operations remain dynamic, such as host-side preprocessing, stream synchronizations, or library kernels that cannot be captured. This limitation is particularly true during each decoding step of LLM inference, where dynamic control flow, such as EOS detection, stop-condition checks, and agent tool-call dispatches, cannot be captured at graph record time [7, 19].

To ensure that GPUs receive kernels on time, multi-GPU ML and LLM libraries such as torch.distributed [25] rely on multiprocessing to allow each process to launch kernels independently and provide sufficient concurrency for high-throughput GPU dispatch. As a result, multi-GPU inference frameworks, including vLLM [23], DeepSpeed [42], and HuggingFace Accelerate [10], allocate at least one process per GPU, as also depicted in Figure 1. This structure isolates kernel launch responsibilities, reduces contention on the host, and ensures that each GPU receives data and kernels with minimal delay. The key takeaway is that modern multi-GPU frameworks should operate on a system with at least one process dedicated to each GPU to ensure efficient kernel dispatch and consistent data feeding into the accelerators.

Taken together, if a CPU-GPU resource imbalance exists and CPU threads and processes interfere with the LLM engine responsible for kernel dispatch, the entire GPU workload pipeline may be substantially delayed. In particular, under concurrent LLM requests, multiple requests can keep the tokenizer heavily occupied and contend with the kernel-launching processes, introducing delays even on systems with seemingly sufficient CPU resources. This competition causes cascading slowdowns in kernel dispatch, request scheduling, and ultimately end-to-end inference performance. We present experiments in Section IV that demonstrate how CPU can cause end-to-end slowdowns, along with an analysis of latency and CPU & GPU utilization. Additionally, Section V-A examines the effect of CPU oversubscription in more detail, visualizing cases where CPU-side resource contention causes delayed kernel calls and GPU underutilization.

TABLE I: List of CPU-GPU heterogeneous system setups used in this paper’s evaluation.

System (GPU)

Architecture

(Compute Capability)

CPU Model
#CPU Cores
#GPUs per Node
Interconnect

H100
Hopper (9.0)
Intel Xeon Platinum 8480CL
64
8
NVLink 4.0 (900 GB/s)

H200
Hopper (9.0)
Intel Xeon Platinum 8480CL
64
8
NVLink 4.0 (900 GB/s)

RTX Pro 6000
Blackwell (12.0)
Dual Intel Xeon 6737P
64
8
No NVLink (PCIe 5.0, 64 GB/s)

Figure 3: CDF of users’ CPU-to-GPU allocation ratios in the instructional cluster, weighted by GPU hours. Vertical lines mark percentiles; users manually set CPU and GPU counts.

II-B Real-World CPU Under-Provisioning in HPC Clusters

Prior cluster trace studies have characterized GPU job workloads at scale [12, 53], but have not specifically examined CPU-to-GPU allocation ratios. We analyzed real-world resource allocation patterns from production HPC clusters. We found that CPUs are indeed frequently under-requested, suggesting that imbalances between CPUs and GPUs are a practical performance problem. We collected job scheduler logs from two institutional clusters operating within a university computing environment during calendar year 2024, totaling 4.65 million salloc records. The first cluster primarily serves research workloads, while the second supports instructional and educational use. Together, the clusters comprise approximately 1,400 compute nodes with around 35,000 CPU cores and several hundred GPUs, including recent high-end models such as NVIDIA H200 and RTX Pro 6000 Blackwell GPUs.

On the research cluster, the scheduler enforces a fixed CPU-to-GPU allocation policy: each GPU receives roughly 1/N1/N of the node’s total CPU cores, where NN is the number of GPUs per node. This configuration preserves CPU/GPU affinity and prevents users from monopolizing CPU cores. In contrast, the instructional cluster does not enforce a strict CPU-to-GPU ratio, reflecting its instructional purpose to allow demonstrations of performance and allocation strategies. On this cluster, the default Slurm scheduling parameter --cpus-per-task=1 allocates only a single CPU core per task, severely degrading the performance of multi-GPU workloads.

Figure 4: CDF of users’ CPU-to-GPU allocation ratios in the research cluster, weighted by GPU hours. Vertical lines mark percentiles; CPU cores are set proportionally to GPUs unless users override the setting.

Figure 3 summarizes the aggregated user-allocation data from the instructional cluster. The figure plots the cumulative distribution (CDF) of CPU-to-GPU allocation ratios weighted by GPU hours. We observe that a significant fraction of users allocate far fewer CPU cores than GPUs. In most cases, users either forget or are not aware that they must explicitly request additional CPU cores. The median (P50) CPU-to-GPU ratio lies around 1–2 for both A100 and H100 nodes, while the P25 percentile across all device types falls at or below 2. Notably, H100 nodes, accounting for 34.3k GPU hours out of 50.9k total, show cases where users request as little as 1 CPU core for 4 or 8 GPUs, resulting in a P25 ratio of 0.25. These findings suggest that many users remain unaware that inadequate CPU allocation can introduce severe bottlenecks in multi-GPU workloads.

As shown in Figure 4, this behavior is less pronounced on the research cluster because its scheduler enforces proportional CPU allocation. Nevertheless, around 60% of jobs on certain GPU types still exhibit CPU-to-GPU ratios below 8. As our experiments in Section IV show, ratios below 4–8 cores per GPU risk measurable slowdown for LLM inference workloads, leaving these jobs vulnerable to CPU-side performance bottlenecks.

In summary, while CPU scarcity is rare for large jobs that occupy an entire multi-GPU node (e.g., a full DGX box), it is common in shared-node workloads where users run multi-GPU jobs with limited CPU allocations. The problem becomes more pronounced when the server operator fails to force the system to allocate sufficient CPU resources per requested GPU. These observations corroborate the CPU-induced bottlenecks identified in Sections IV and Sections V.

III Multi-GPU System Evaluation Setup

Table I summarizes the systems used in our experiments, including GPU architecture, number of GPUs, and available CPU cores. Our evaluation spans multiple GPU generations (from Hopper to Blackwell) and a range of configurations that combine different CPU and GPU core counts, and high-bandwidth interconnects, such as NVLink. We test multiple systems to determine whether our experiment’s CPU slowdown occurs universally. Simultaneous multithreading (SMT) is disabled for all experiments to ensure a consistent and accurate characterization of CPU-side effects. We note that production cloud instances typically expose vCPUs (hyperthreads) rather than physical cores; our physical-core measurements, therefore, represent a conservative baseline, as SMT would provide additional logical cores but with shared execution resources.

Our evaluations are conducted using virtualized or restricted CPU allocations on the listed machines to examine how multi-GPU setups in multi-tenant clusters can lead to unexpected performance degradation due to limited CPU resources. For example, on a server equipped with four H100 GPUs, we may create a constrained execution environment with only 16 CPU cores to emulate realistic cloud or cluster provisioning. As shown in our cluster log analysis (Section II-B) and cloud pricing survey (Section VI-A), such CPU-to-GPU imbalances are common in practice, confirming that this evaluation reflects real-world deployment conditions.

All LLM inference experiments use vLLM v0.11.1 with the V1 engine architecture, which separates the API server process (handling tokenization and request management) from the EngineCore process (handling scheduling) via ZeroMQ (ZMQ) inter-process communication (IPC); the EngineCore then dispatches execution commands to GPU worker processes via shared memory broadcast. Notably, the following optimizations are enabled by default: CUDA Graphs in full-and-piecewise mode for reduced kernel launch overhead, chunked prefill for overlapping prefill and decode, prefix caching, torch.compile with the Inductor backend, and custom all-reduce kernels for intra-node communication. Our results, therefore, reflect the performance of a fully optimized serving stack; the CPU bottlenecks we identify persist despite these optimizations.

Moreover, we use a fixed CPU & GPU resource allocation per job, reflecting common practice in cluster schedulers such as Slurm [48] and in cloud service offerings where users receive a predetermined number of CPU cores per GPU [2]. While such isolation supports fair sharing across users, it also introduces the risk of mismatched CPU and GPU allocations, as these allocations are typically chosen manually or determined by simple heuristics (e.g., assigning 4 CPU cores per GPU). This setup enables us to systematically investigate how insufficient CPU provisioning contributes to slowdowns in LLM inference and serving workloads.

IV CPU Bottleneck in LLM Inference

In this section, we analyze how CPU-side tasks during LLM inference and serving, such as tokenization, runtime orchestration, and communication coordination, can become significant bottlenecks when multiple jobs or requests execute concurrently. We demonstrate that contention for limited CPU resources not only delays host-side processing but also slows GPU execution, resulting in increased per-token latency and reduced overall throughput.

IV-A Tokenization Latency Evaluation in LLM Inference

Figure 5: Relative latency breakdown of tokenization and the time-to-first-token (TTFT) across varying batch sizes and sequence lengths (SL). CPU-side tokenization accounts for up to half of the total latency. Llama 3.1 8B on 4×\timesH200 system.

Figure 5 visualizes how much of the end-to-end LLM inference latency is attributed to tokenization. We use vLLM [23] to systematically quantify the relative cost of CPU-side tokenization compared to GPU-side model execution. At the same time, similar CPU-side bottlenecks are expected in other frameworks, such as SGLang [59], which also uses a multi-process architecture. Experiments are conducted using the Llama 3.1 8B model on a 4×\timesH200 system with 16 CPU cores, configured with tensor parallelism [46] and HuggingFace Tokenizers [14] integrated in vLLM. For each trial, we vary batch size and input sequence length (SL) to measure tokenization latency and the model’s time-to-first-token (TTFT).

Tokenization accounts for up to 50% of total latency in long-sequence configurations. Notably, this fraction does not diminish at longer sequence lengths because modern serving stacks use chunked prefill and FlashAttention [5], which cause prefill time to scale near-linearly rather than quadratically with input length. As a result, tokenization remains a comparable fraction of TTFT even with modern high-performance GPUs. As context lengths grow, e.g., tokenizing a 1M-token prompt at current throughput would require multiple seconds of CPU time per request, the overhead associated with tokenization is likely to become an increasingly significant factor [52]. While not included in the figure, we observe a similar breakdown across different numbers of CPU cores (e.g., five or eight), though absolute tokenization latency increases by approximately 5% and TTFT increases by around 10% due to thread scheduling delays and context-switching overhead. We note that the reported fractions are relative to TTFT (prefill latency), not end-to-end generation latency; the relative impact of tokenization decreases when token generation is factored in. Nonetheless, the results demonstrate that preprocessing can substantially reduce throughput in realistic LLM use cases.

IV-B Impact of Tokenization Load in LLM Serving

Next, we evaluate how a busy tokenizer can impact the LLM engine, potentially slowing kernel launches and increasing overall latency. Our experiment demonstrates that under heavy load, multiple tokenizer threads compete with the LLM engine for CPU resources, leading to frequent context switches, delayed kernel launches, and reduced GPU utilization. Such a slowdown substantially degrades latency as communication kernels rely on barrier-based synchronization, as discussed further in Section V-A.

Figure 6: (a) Baseline LLM inference with no load. (b) LLM inference under concurrent ”attacker” load. 1. Tokenizer spawns multiple threads, 2. CPU cores are saturated, 3. Delayed kernel launches underutilize GPUs and degrade overall performance.

Figure 7: Latency of Llama 3.1 8B and Qwen 2.5 14B on 4 & 8 GPUs under CPU load (8 & 16 requests per second), varying attacker sequence length and CPU core counts. Red ×\times marks denote timeouts. Red arrows indicate latency improvements from the smallest to the best CPU allocation, with corresponding speedups annotated. Measured on the Blackwell system.

Evaluation methodology. We design an experiment where multiple concurrent requests are sent to the LLM server, and we measure how this background load affects the latency of a single target request. We refer to the measured request as the victim, and we generate additional periodic requests, termed attackers, that impose a controlled CPU load defined by a specified request rate per second (RPS). The overall setup is illustrated in Figure 6, which shows how four attacker requests after the victim keep the tokenizer busy, thereby delaying both tokenization and inference due to delayed kernel calls and GPU underutilization. We implement this experiment using vllm serve with the V1 async engine described in Section III, where user queries arrive asynchronously over HTTP. Despite V1’s process-level separation of tokenization, scheduling, and GPU execution, CPU contention persists because all processes compete for the same limited pool of CPU cores.

In this experiment, we quantify CPU contention by measuring the latency of the victim request under attacker load while varying the number of CPU cores. For example, suppose a victim request takes 1 second with no load and 3 seconds under attacker load, but adding cores (e.g., up to 16) reduces the latency to 1.5 seconds. In that case, we can infer that CPU resources were the main bottleneck and that extra cores help keep the CPU–GPU pipeline utilized. Conversely, if latency remains at 3 seconds regardless of the number of added cores, the workload is GPU-bound. This approach cleanly distinguishes CPU- versus GPU-limited regimes in multi-GPU LLM serving.

Experimental setup. We evaluate Llama 3.1 8B and Qwen 2.5 14B on the three machines in Table I, using 4- and 8-GPU configurations. CPU resources are provisioned at four levels: (#GPUs + 1) cores (least-CPU case), 2×\times#GPUs, 4×\times#GPUs, and 8×\times#GPUs (CPU-abundant cases). As described in Section III, vLLM V1 requires at least (#GPUs + 2) concurrent processes; our least-CPU configuration of (#GPUs + 1) cores is therefore already below this minimum, guaranteeing that at least one process must time-share a core via OS scheduling. We include this extreme configuration because our cluster log analysis (Section II-B) reveals that users frequently request such constrained allocations, making this level of oversubscription a realistic operational scenario. In practice, allocating exactly #GPUs cores caused the system to operate prohibitively slowly, so we use (#GPUs + 1) as the bare-minimum functional configuration. This range lets us characterize the workload’s CPU intensity. For each configuration, we measure the time-to-first-token latency, which includes tokenization and one model forward pass.

The attacker sequence length ranges from 1.8k to 114k tokens, while the victim’s sequence length is 2.8k tokens. There are slight differences in the exact token count between the Llama and Qwen models. We measured the average latency of five continuous victim requests after the attack commenced. The attacker’s requests per second (RPS) were set to 8 and 16.

Figure 8: Time-to-first-token (TTFT) latency of sequential victim requests under attacker load (8 & 16 requests per second, 114k tokens). As attacker requests accumulate, victim latency grows, but higher CPU allocations reduce this slowdown.

Figure 9: Heatmap showing the best speedup among CPU-abundant configurations (2×\times#GPUs, 4×\times#GPUs, 8×\times#GPUs) relative to the least-CPU case (#GPUs + 1). ∞\infty indicates timeouts in the minimal-CPU configuration.

Latency per request. Figure 8 reports the time-to-first-token (TTFT) latencies of the baseline and five victim requests under attack conditions, averaged over ten runs with standard deviation shown. Llama 3.1 8B with tensor parallelism (TP) of 4 is run on the Blackwell system, with the attacker generating 8 or 16 requests per second, each with a 114k sequence length. Victim requests are issued sequentially, so as attacker requests accumulate in the serving engine, each subsequent victim TTFT shows an increasing trend. The results show that increasing the number of CPU cores reduces TTFT per request, suggesting that more CPU resources mitigate slowdowns caused by tokenization and CPU-side contention. In particular, scaling from 5 CPU cores (the least-CPU configuration) to 32 cores (8 per GPU) consistently reduces victim request TTFT under load by over 5×\times.

Latency results. Figure 7 summarizes the TTFT of victim requests on the Blackwell system, both with and without attacker requests at a given rate per second (RPS). Figure 9 reports the speedup of each CPU-abundant configuration relative to the least-CPU case across all three systems (H100, H200, and Blackwell). The same pattern holds across platforms: increasing CPU resources consistently reduces victim TTFT, confirming that the CPU bottleneck is not specific to a single system or interconnect topology. Across Llama and Qwen on 4–8 GPUs, removing CPU scarcity improves long-sequence TTFT by 1.36–5.40×\times when we go from (#GPUs + 1) cores to a CPU-abundant configuration.

This improvement stems from two effects: \scriptsize{1}⃝ the Rayon thread pool (the Rust-based parallelism library used internally by HuggingFace Tokenizers) within the API server process faces less contention. It finishes sooner, and \scriptsize{2}⃝ host-side tasks across the EngineCore (batching, scheduling, and memory management) and GPU worker processes (kernel dispatch and data transfers) proceed without delay. With higher attacker RPS, systems with fewer cores reach CPU saturation earlier, resulting in sharp latency degradation. Under oversubscription, context switching spikes, kernel launches become serialized, and GPUs sit idle even when work is available. Larger CPU allocations tolerate much higher attacker load before throughput collapses, indicating that CPU headroom directly raises the critical throughput threshold and delays GPU starvation.

LLM engine starvation. Some configurations fail to produce latency values, marked by red ×\times’s in Figure 8. This occurs when the tokenization overhead from attacker requests drives the engine’s throughput so low that the victim’s request never begins generating tokens. As attacker load grows, tokenization saturates the CPU, slowing other host-side tasks and delaying GPU kernel launches. Higher per-token latency keeps requests active longer, allowing even more arrivals and further increasing CPU pressure. Eventually, the system enters a pathological state where CPU utilization remains at or near 100% for extended periods (as shown in Figure 10). The server cannot make enough progress on the victim request, which then times out at our 200-second limit, chosen as a generous upper bound on acceptable interactive latency.

Figure 10: CPU utilization with Llama across different core allocations for 4-GPU and 8-GPU configurations. While all cases achieve near-100% utilization, utilizing more CPU cores results in shorter periods of high utilization.

CPU and GPU utilizations. Figure 10 shows that while all configurations occasionally approach full CPU usage, the duration of saturation is what primarily drives CPU-induced latency. With only 5 cores, the system remains near 100% utilization for extended periods (up to 100 seconds in the 4-GPU setting), causing preprocessing tasks to queue and resulting in the highest end-to-end latency. Increasing the CPU allocation to 8, 16, and 32 cores still produces brief spikes in utilization, but these peaks are much shorter, allowing the threads to drain quickly and preventing starvation of the GPU launch path. The 8-GPU configuration exhibits similar behavior, and although the CPU can momentarily reach full utilization even with 64 cores, these events last only for short intervals.

Furthermore, Figure 11 confirms our claim that CPU tasks make the GPU idle with low utilization. Providing sufficient CPU resources enables the GPU to operate at maximum efficiency, allowing the work to be completed more quickly (as reflected in the shorter time span shown in the figure). Along with the latency results, the utilization traces clearly show that CPU oversubscription leads to GPU stalls, causing latency jumps during LLM request serving. Both CPU & GPU utilizations are measured under 8 RPS with 114k-token sequences. Utilization traces are shown for Llama only; Qwen exhibits qualitatively similar patterns and is omitted for brevity.

Takeaways. In multi-request LLM serving pipelines, tokenization adds substantial pressure on the CPU, and this often causes end-to-end latency to be limited not by GPU throughput but by the CPU’s ability to perform tokenization and kernel scheduling on time. Under concurrent load, these CPU bottlenecks delay kernel launches, stall preprocessing pipelines, and leave GPUs underutilized. The attacker–victim experiment demonstrates that allocating sufficient CPU resources is crucial for achieving high serving performance, particularly under high-RPS or long-context workloads where tokenization dominates the critical path.

Figure 11: CPU and GPU utilization with Llama for the 4-GPU setup across varying CPU core allocations, illustrating the relationship between CPU saturation and GPU underutilization.

Training workloads. While our focus is on inference, we note that CPU bottlenecks manifest differently in training workloads. Training batch sizes and sequence lengths are constrained by GPU memory rather than CPU preprocessing capacity, making training generally less CPU-sensitive for LLMs. Vision training with data augmentation is a known exception [29, 20]. A detailed characterization of CPU bottlenecks in training is left for future work.

V Understanding the CPU Bottlenecks in Multi-GPU Systems

This section analyzes why CPU oversubscription and resource contention among multiple processes degrade multi-GPU performance by examining the CPU’s role in kernel launches and GPU synchronization. When insufficient CPU resources are available, latency-sensitive operations accumulate delays that propagate directly into GPU idle time, as computation kernels and communication kernels are launched sequentially and alternately in ML workloads. Additionally, we find that contention on shared regions such as the Linux shared-memory segment increases launch-path latency when many processes compete for access.

V-A Synchronization and CPU Oversubscription in Communication Kernels

To examine how CPU contention affects GPU utilization, we developed a microbenchmark with torch.distributed framework [25] and profiled it using the PyTorch Profiler [40]. The experiment isolates the interaction between CPU-side kernel dispatch and GPU-side execution, allowing us to quantify the synchronization overheads induced by CPU oversubscription.

Figure 12: Illustration of CPU oversubscription causing busy-waiting GPUs in multi-GPU workloads. When multiple CPU processes contend for limited cores, kernel launches are executed sequentially, leading to severe underutilization of GPUs. Gray boxes indicate subsequent kernel calls.

Figure 12 illustrates the profiled result of communication kernels under extreme CPU contention, where one process per GPU plus additional host-side processes contend for a very small number of CPU cores (i.e., 4 GPUs and 1 or 2 CPU cores). In this configuration, the CPU can launch only a limited number of communication kernels at a time, forcing the GPU kernels to execute sequentially rather than concurrently. Since NCCL collectives such as all_reduce introduce global synchronization barriers, faster GPUs must wait until every process has reached the barrier, creating idle busy-wait periods that waste compute cycles (highlighted in the figure by black dotted lines). This produces a straggler effect: in an 8-GPU collective, if a single CPU core is delayed by 1 ms due to OS scheduling, all other GPUs busy-wait for that duration, amplifying a small per-core delay into a cluster-wide stall.

This phenomenon generalizes beyond collectives: any workload stage that involves CPU-side coordination, such as dataloader workers in training, tokenization threads in LLM inference, or request-handling processes in serving engines, can trigger similar stalls when CPU cores are oversubscribed. Context switching between these processes not only increases latency in kernel launches but can also disrupt the timing assumptions of tightly synchronized GPU workloads, amplifying end-to-end slowdown. Avoiding such oversubscription is critical, as it can delay synchronization and substantially degrade system throughput. Analytical performance models have independently identified collective communication synchronization as a fundamental bottleneck in distributed LLM inference [6]; our measurements provide empirical confirmation of this on real hardware under CPU-constrained conditions.

V-B Shared Memory Broadcast Contention

In multi-GPU LLM serving, the GPU data plane (attention, MLP, and collective communication) operates with minimal CPU involvement, but the CPU control plane (tokenization, scheduling, and inter-process communication) must complete before each GPU iteration can begin. Modern serving frameworks such as vLLM [23] and SGLang [59] share this multi-process architecture, making the following analysis broadly applicable. In vLLM’s V1 architecture, POSIX shared memory (exposed via Linux’s /dev/shm) implements a broadcast queue that distributes scheduling decisions from the engine core to each GPU worker process. NCCL similarly allocates its own shared-memory regions to support intra-node coordination for collective communications [13].

This broadcast follows a 1-writer-N-reader pattern, where NN equals the tensor parallelism degree. The writer (engine core) must verify that all NN readers have consumed the previous message before writing new data, and each reader must confirm the writer has produced a new entry before consuming it. Although the implementation uses a lock-free design based on per-entry metadata flags and memory fences rather than traditional mutexes, contention still arises: the writer polls all NN reader flags on each iteration in a busy-wait loop that never yields to sleep, while readers similarly poll the writer flag. Under CPU scarcity, the writer’s busy-wait competes with reader processes for CPU time, delaying readers’ flag updates and forcing the writer to spin longer, creating a cascading delay that amplifies with the number of GPU workers.

Figure 13: Illustration of contended dequeue() operations on the shared-memory broadcast queue, where concurrent CPU load delays scheduling metadata delivery to GPU workers.

Figure 13 illustrates how this contention can substantially delay model execution. Under a load of 5 requests/s with 100k-token inputs on an H100 system using tensor parallelism (TP) = 4, the dequeue() operation in vLLM’s shm_broadcast.py, used by each GPU worker process to receive scheduling metadata, experiences up to a 19×\times slowdown relative to the uncontended case (from approximately 12 ms to 228 ms, as shown in Figure 13). With the decode phase taking only 44 ms, the contended dequeue latency of 228 ms is roughly 5×\times the GPU compute step itself, indicating that the CPU control plane dominates the critical path. This delay directly increases time-to-first-token and risks failing to meet latency SLOs in production-serving systems. The problem is exacerbated by LLM serving’s continuous batching, which requires a new scheduling decision (and thus a new broadcast) at every decode step, potentially multiplying per-step IPC overhead across many autoregressive iterations.

Takeaways. While additional CPU cores effectively mitigate oversubscription-induced slowdowns (Section V-A), broadcast contention is a structural property of the 1-writer-N-reader pattern that additional cores alone cannot fully resolve: even with adequate CPU cores, the writer must wait for the slowest reader before proceeding, and the contention is structurally proportional to the tensor parallelism degree, as the writer must poll NN reader flags per iteration. The issue compounds in multi-tenant environments where tokenization and IPC polling compete for the same CPU cores. Mitigating these bottlenecks may require redesigned IPC mechanisms, such as asynchronous scheduling pipelines that overlap IPC with GPU execution, persistent GPU kernels that poll a device-side queue to eliminate per-step launch overhead [26], or direct GPU-to-GPU signaling that entirely bypasses the CPU control plane.

VI Discussion

VI-A CPU Under-Provisioning in Cloud Compute Platforms

Public cloud providers such as AWS offer GPU instances with pricing that highlights a significant cost disparity between CPU and GPU resources [2]. GPU instances range from $3.06/hour for a single V100 (p3.2xlarge) to $55.04/hour for an 8x H100 configuration (p5.48xlarge) [2]. In contrast, CPU cores cost approximately $0.03–$0.06/hour per vCPU based on monthly rates of $21.73–$45.86 per core, making GPU compute roughly 100–1,600×\times more expensive than CPU cores depending on generation.

Cloud instance configurations from major vendors commonly provide as few as 3–6 vCPUs per GPU [2], and our cluster log analysis (Section II-B) confirms that users frequently allocate even fewer. Without prior knowledge of CPU bottlenecks, users may overlook this imbalance by selecting minimal CPU counts to reduce costs, creating a counterproductive cycle: the workload becomes CPU-starved, GPU utilization drops, and organizations waste substantial GPU investment on idle cycles.

Since the marginal cost of additional CPU cores is small relative to GPU instance pricing, e.g., adding 16 vCPUs at $0.05/hour each ($0.80/hour) to a p5.48xlarge instance at $55.04/hour represents roughly a 1.5% cost increase, our analysis shows that adding CPU resources is remarkably cost-effective. For CPU-bound workloads, performance scales nearly linearly with additional cores at minimal cost, reducing the total GPU hours required and delivering better overall throughput per dollar than simply provisioning more GPUs.

We note that conventional mitigation strategies do not fully address the CPU bottlenecks identified in this work. Admission control and rate limiting can prevent CPU overload, but they do so by reducing the accepted query rate, leaving GPU capacity underutilized and undermining the economics of multi-GPU deployment. OS-level resource isolation via cgroup, cpuset, and CPU pinning tools such as taskset and isolcpus [9] can improve scheduling determinism by dedicating cores to latency-sensitive processes, but cannot compensate when the total number of allocated cores is fundamentally insufficient for the workload’s concurrency demands. Moreover, the default OS scheduler treats all processes equally and cannot distinguish latency-critical tasks (kernel dispatch, IPC polling) from throughput-oriented ones (tokenization), allowing long-running CPU-heavy work to preempt time-sensitive control-plane operations. Evaluating the effectiveness of such pinning strategies under varying core budgets is a promising direction for future work. While large cloud instances (e.g., AWS p5.48xlarge) provision up to 24 vCPUs per GPU by default, multi-tenant sharing and cost-driven optimization frequently reduce effective per-GPU CPU allocations well below this level.

More fundamentally, solutions to CPU bottlenecks fall into two categories: host-driven approaches that strengthen the CPU (adding cores, pinning, priority scheduling) and device-initiated approaches that remove the CPU from the critical path entirely, such as GPU-initiated networking [37, 47, 11] and persistent GPU kernels [26]. Our results quantify the benefits of adding CPU cores; the latter represents a promising but more invasive architectural direction. Notably, modern GPUs already include on-die control processors, such as NVIDIA’s RISC-V-based GPU System Processor (GSP) [1], which could potentially assume the orchestration duties currently handled by the host CPU.

VI-B Emerging Trends That May Intensify CPU Bottlenecks

Several emerging trends in LLM deployment suggest that CPU-side bottlenecks will become more pronounced rather than less. First, agentic AI workloads perform dynamic reasoning with frequent tool invocations, each requiring CPU-side processing that creates substantial GPU idle time [19, 41, 60]. Multimodal inputs such as images and video require CPU-intensive preprocessing (resizing, normalization, frame extraction) before GPU inference can begin, adding further pressure to the host. Prefill-decode disaggregation [61, 38] can shift CPU-intensive tokenization to dedicated prefill servers, altering CPU provisioning but not eliminating it. Finally, long-context inference with sequences exceeding one million tokens incurs tokenization latency that scales linearly with input length, potentially requiring seconds of CPU time per request. Additionally, speculative decoding [24] introduces additional CPU overhead for verification and tree management, with rollback on rejection adding further host-side work. As GPU compute scales faster than CPU performance, deploying even moderately sized models across multiple GPUs reduces per-GPU compute time, increasing the relative contribution of CPU coordination overhead. Together, these trends indicate that the CPU bottlenecks characterized in this work are not artifacts of current-generation systems but structural challenges that will require architectural attention.

VI-C Limitations

Our experiments use NVIDIA GPUs with Intel CPUs and SMT disabled; production cloud instances typically expose vCPUs (hyperthreads), so our physical-core measurements represent a conservative baseline. CPU-GPU coupling architectures such as Grace Hopper [34], where ARM cores show different kernel launch characteristics [50], or systems with high-core-count AMD EPYC processors may exhibit different bottleneck profiles. Additionally, while our structural analysis of control-plane contention applies broadly to multi-process serving architectures, our quantitative evaluation uses vLLM; validating these findings on other frameworks such as SGLang [59] and TensorRT-LLM [36] remains future work.

VII Related Work

VII-A CPU Overhead in Agentic AIs

As agentic AI systems perform dynamic reasoning and tool usage, their machine learning (ML) workloads increasingly become CPU-bound. Kim et al. [19] reveal that frequent tool invocations cause significant GPU idle time, shifting the workload from being GPU-bound to CPU- or I/O-bound. Similarly, Raj et al. [41] find that tool-processing components predominantly run on CPUs, accounting for up to 90.6% of total latency, underscoring the need for system-level optimizations that account for the entire hardware stack.

VII-B GPU-Driven Control and Device-Initiated Communications

A significant bottleneck in large-scale model execution is the overhead of CPU involvement in managing GPU execution and communication. ARK [16] proposes a GPU-driven framework that reduces the CPU’s role in scheduling and launching GPU kernels. Libraries built on NVSHMEM [37] enable communication to be initiated directly from the device. NCCLX [47] leverages this model, using lightweight GPU threads to parallelize the preparation of remote direct memory access (RDMA) operations. DeepEP [26], a communication library tailored for Mixture-of-Experts LLMs [17], also utilizes NVSHMEM to manage its expert parallelism efficiently. However, despite these efforts, popularly used frameworks such as vLLM and PyTorch still rely on CPU-driven coordination. LithOS [4] redesigns the OS-GPU interface by atomizing long-running kernels into smaller units, enabling preemptive scheduling that can reduce head-of-line blocking from control-plane delays. Other approaches leverage spare CPU resources for distributed training [18].

VII-C LLM Serving Frameworks

Modern LLM serving systems adopt multi-process architectures to isolate CPU-side tasks from GPU execution, including vLLM [23] (Section III), SGLang [59], and TensorRT-LLM [36]. DistServe [61] and Splitwise [38] disaggregate the prefill and decode phases onto separate GPU pools to optimize goodput, thereby concentrating CPU-intensive tokenization on dedicated prefill servers. While these frameworks optimize GPU utilization and scheduling, all rely on CPU-side processes for tokenization, request management, and inter-process coordination, leaving them vulnerable to the bottlenecks characterized in this work.

VII-D Mitigating the CPU Overhead in LLMs

Tokenization in LLMs is a key contributor to CPU bottlenecks. NetTokenizer [28] offloads tokenization from CPUs to programmable network data planes. BlockBPE [56] introduces GPU-based byte pair encoding (BPE) tokenization, effective for high-batch workloads but less so for latency-sensitive tasks. LoPT [45] achieves lossless parallel tokenization through a token-merge mechanism. Meanwhile, Recasens et al. [43] report CPU overhead reaching up to 30% of total execution time in large-batch LLM inference.

VII-E Mitigating Kernel Launch Overhead

Lustig et al. [27] propose early kernel launch to reduce CPU–GPU synchronization latency. Vellaisamy et al. [50] characterize kernel launch overhead on CPU–GPU coupled architectures including Grace Hopper, focusing on single-GPU scenarios; our study targets the compounding effects across multiple GPUs under concurrent serving load. OS-level tuning such as CPU pinning has been shown to reduce scheduling jitter [9]. CUDA Graphs [30, 8] reduce launch overhead for static sequences but cannot capture dynamic scheduling decisions in LLM decoding. Grape [58] and KTransformers [3] extend CUDA Graphs to dynamic DNNs and MoE models, respectively, but neither addresses tokenization contention under concurrent requests. FlashAttention [5] reduces kernel count via fusion but does not address launch-path overhead.

Prior works have independently noted CPU overhead in tokenization [56, 45], kernel launch paths [16, 50], and data preprocessing pipelines [29]. However, none examine how these factors compound under multi-GPU LLM serving, where tokenization contention, IPC broadcast delays, and barrier-based synchronization interact to produce cascading slowdowns far exceeding any individual overhead. Complementary GPU-side characterizations [51, 43] focus on compute and memory bottlenecks within the accelerator. Our work provides a systematic characterization of the CPU-side effects that compound with these GPU-side phenomena, and quantifies how CPU provisioning affects end-to-end performance across diverse configurations.

VIII Conclusion

Our study shows that CPU-side responsibilities (tokenization, kernel coordination, and inter-process communication) can become critical bottlenecks in multi-GPU LLM inference, even in fully optimized serving stacks. CPU oversubscription, combined with barrier-based GPU synchronization, leaves GPUs idle, while shared-memory broadcast contention introduces structural delays proportional to the degree of tensor parallelism. Providing adequate CPU resources improves TTFT by 1.36–5.40×\times, and our analysis of 4.65 million cluster job records confirms that such under-provisioning is widespread. Since the marginal cost of additional CPU cores is small relative to GPU instance pricing, adequate CPU provisioning is a highly cost-effective lever for improving multi-GPU LLM serving performance.

References

[1]

“NVIDIA GSP firmware,”
https://download.nvidia.com/XFree86/Linux-x86_64/560.28.03/README/gsp.html,
2024.

[2]

“Amazon ec2 instance types,”
https://aws.amazon.com/ec2/instance-types/, AWS.

[3]

H. Chen, W. Xie, B. Zhang, J. Tang, J. Wang, J. Dong, S. Chen, Z. Yuan, C. Lin,
C. Qiu et al., “Ktransformers: Unleashing the full potential of
cpu/gpu hybrid inference for moe models,” in Proceedings of the ACM
SIGOPS 31st Symposium on Operating Systems Principles, 2025, pp. 1014–1029.

[4]

P. H. Coppock, B. Zhang, E. H. Solomon, V. Kypriotis, L. Yang, B. Sharma,
D. Schatzberg, T. C. Mowry, and D. Skarlatos, “Lithos: An operating system
for efficient machine learning on gpus,” in Proceedings of the ACM
SIGOPS 31st Symposium on Operating Systems Principles, 2025, pp. 1–17.

[5]

T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, “Flashattention: Fast and
memory-efficient exact attention with io-awareness,” Advances in
neural information processing systems, vol. 35, pp. 16 344–16 359, 2022.

[6]

M. Davies, N. Crago, K. Sankaralingam, and C. Kozyrakis, “Liminal: Exploring
the frontiers of llm decode performance,” arXiv preprint
arXiv:2507.14397, 2025.

[7]

S. Durvasula, A. Zhao, R. Kiguru, Y. Guan, Z. Chen, and N. Vijaykumar, “Acs:
Concurrent kernel execution on irregular, input-dependent computational
graphs,” arXiv preprint arXiv:2401.12377, 2024.

[8]

J. Ekelund, S. Markidis, and I. Peng, “Boosting performance of iterative
applications on gpus: Kernel batching with cuda graphs,” in 2025 33rd
Euromicro International Conference on Parallel, Distributed, and
Network-Based Processing (PDP). IEEE,
2025, pp. 70–77.

[9]

“Os-level challenges in LLM inference and optimizations,”
https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/,
Eunomia, 2025.

[10]

S. Gugger, L. Debut, T. Wolf, P. Schmid, Z. Mueller, S. Mangrulkar, M. Sun, and
B. Bossan, “Accelerate: Training and inference at scale made simple,
efficient and adaptable.” https://github.com/huggingface/accelerate,
2022.

[11]

K. Hamidouche, J. Bachan, P. Markthub, P.-J. Gootzen, E. Agostini, S. Jeaugey,
A. Shafi, G. Theodorakis, and M. G. Venkata, “Gpu-initiated networking for
nccl,” arXiv preprint arXiv:2511.15076, 2025.

[12]

Q. Hu, Z. Ye, Z. Wang, G. Wang, M. Zhang, Q. Chen, P. Sun, D. Lin, X. Wang,
Y. Luo et al., “Characterization of large language model development
in the datacenter,” in 21st USENIX Symposium on Networked Systems
Design and Implementation (NSDI 24), 2024, pp. 709–729.

[13]

Z. Hu, S. Shen, T. Bonato, S. Jeaugey, C. Alexander, E. Spada, J. Dinan,
J. Hammond, and T. Hoefler, “Demystifying nccl: An in-depth analysis of gpu
communication protocols and algorithms,” arXiv preprint
arXiv:2507.04786, 2025.

[14]

“Huggingface tokenizers,” https://huggingface.co/docs/tokenizers/index,
HuggingFace.

[15]

“Huggingface tokenizers repository,”
https://github.com/huggingface/tokenizers, HuggingFace.

[16]

C. Hwang, K. Park, R. Shu, X. Qu, P. Cheng, and Y. Xiong,
“{\{ARK}\}:{\{GPU-driven}\} code execution for distributed deep
learning,” in 20th USENIX Symposium on Networked Systems Design and
Implementation (NSDI 23), 2023, pp. 87–101.

[17]

A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S.
Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand et al., “Mixtral of
experts,” arXiv preprint arXiv:2401.04088, 2024.

[18]

Y. Jiang, Y. Zhu, C. Lan, B. Yi, Y. Cui, and C. Guo, “A unified architecture
for accelerating distributed {\{DNN}\} training in heterogeneous
{\{GPU/CPU}\} clusters,” in 14th USENIX Symposium on Operating
Systems Design and Implementation (OSDI 20), 2020, pp. 463–479.

[19]

J. Kim, B. Shin, J. Chung, and M. Rhu, “The cost of dynamic reasoning:
Demystifying ai agents and test-time scaling from an ai infrastructure
perspective,” arXiv preprint arXiv:2506.04301, 2025.

[20]

T. Kim, C. Park, M. Mukimbekov, H. Hong, M. Kim, Z. Jin, C. Kim, J.-Y. Shin,
and M. Jeon, “Fusionflow: Accelerating data preprocessing for machine
learning with cpu-gpu cooperation,” in International Conference on
Very Large DataBases (VLDB) 2024, 2024, pp. 863–876.

[21]

A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with
deep convolutional neural networks,” Advances in neural information
processing systems, vol. 25, 2012.

[22]

T. Kudo and J. Richardson, “Sentencepiece: A simple and language independent
subword tokenizer and detokenizer for neural text processing,” arXiv
preprint arXiv:1808.06226, 2018.

[23]

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang,
and I. Stoica, “Efficient memory management for large language model serving
with pagedattention,” in Proceedings of the 29th symposium on
operating systems principles, 2023, pp. 611–626.

[24]

Y. Leviathan, M. Kalman, and Y. Matias, “Fast inference from transformers via
speculative decoding,” in International Conference on Machine
Learning. PMLR, 2023, pp.
19 274–19 286.

[25]

S. Li, Y. Zhao, R. Varma, O. Salpekar, P. Noordhuis, T. Li, A. Paszke,
J. Smith, B. Vaughan, P. Damania et al., “Pytorch distributed:
Experiences on accelerating data parallel training,” arXiv preprint
arXiv:2006.15704, 2020.

[26]

A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang,
C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint
arXiv:2412.19437, 2024.

[27]

D. Lustig and M. Martonosi, “Reducing gpu offload latency via fine-grained
cpu-gpu synchronization,” in 2013 IEEE 19th International Symposium on
High Performance Computer Architecture (HPCA). IEEE, 2013, pp. 354–365.

[28]

S. Mehta, “Accelerating llm inference with smart nic tokenization and
caching,” 2025.

[29]

J. Mohan, A. Phanishayee, J. Kulkarni, and V. Chidambaram, “Looking beyond
{\{GPUs}\} for {\{DNN}\} scheduling on {\{Multi-Tenant}\} clusters,”
in 16th USENIX Symposium on Operating Systems Design and Implementation
(OSDI 22), 2022, pp. 579–596.

[30]

“Getting started with cuda graphs,”
https://developer.nvidia.com/blog/cuda-graphs/, NVIDIA.

[31]

“Nvidia collective communications library (nccl),”
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html,
NVIDIA.

[32]

“Nvidia dgx b200,” https://www.nvidia.com/en-us/data-center/dgx-b200/,
NVIDIA.

[33]

“Nvidia dgx h100,” https://www.nvidia.com/en-gb/data-center/dgx-h100/,
NVIDIA.

[34]

“Nvidia gh200 grace hopper superchip,”
https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/,
NVIDIA.

[35]

“Nvidia dali: A gpu-accelerated data loading and pre-processing library,”
https://github.com/NVIDIA/DALI, NVIDIA, 2017.

[36]

“Tensorrt-llm,” https://github.com/NVIDIA/TensorRT-LLM, NVIDIA, 2023.

[37]

“Nvshmem,” https://developer.nvidia.com/nvshmem, NVIDIA, 2024.

[38]

P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, and
R. Bianchini, “Splitwise: Efficient generative llm inference using phase
splitting,” in 2024 ACM/IEEE 51st Annual International Symposium on
Computer Architecture (ISCA). IEEE,
2024, pp. 118–132.

[39]

“Scaling analysis,”
https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis,
Princeton Research Computing.

[40]

“torch.profiler – pytorch 2.9 documentation,”
https://docs.pytorch.org/docs/stable/profiler.html, Pytorch.

[41]

R. Raj, H. Wang, and T. Krishna, “A cpu-centric perspective on agentic ai,”
arXiv preprint arXiv:2511.00739, 2025.

[42]

J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, “Deepspeed: System
optimizations enable training deep learning models with over 100 billion
parameters,” in Proceedings of the 26th ACM SIGKDD international
conference on knowledge discovery & data mining, 2020, pp. 3505–3506.

[43]

P. G. Recasens, F. Agullo, Y. Zhu, C. Wang, E. K. Lee, O. Tardieu, J. Torres,
and J. L. Berral, “Mind the memory gap: Unveiling gpu bottlenecks in
large-batch llm inference,” arXiv preprint arXiv:2503.08311, 2025.

[44]

R. Sennrich, B. Haddow, and A. Birch, “Neural machine translation of rare
words with subword units,” in Proceedings of the 54th annual meeting
of the association for computational linguistics (volume 1: long papers),
2016, pp. 1715–1725.

[45]

W. Shao, L. Zheng, P. Wang, P. Zheng, J. Li, and Y. Fan, “Lopt: Lossless
parallel tokenization acceleration for long context inference of large
language model,” 2025. [Online]. Available:
https://arxiv.org/abs/2511.04952

[46]

M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro,
“Megatron-lm: Training multi-billion parameter language models using model
parallelism,” arXiv preprint arXiv:1909.08053, 2019.

[47]

M. Si, P. Balaji, Y. Chen, C.-H. Chu, A. Gangidi, S. Hasan, S. Iyengar,
D. Johnson, B. Liu, J. Ren et al., “Collective communication for
100k+ gpus,” arXiv preprint arXiv:2510.20171, 2025.

[48]

“Slurm documentation,” https://slurm.schedmd.com/documentation.html,
Slurm.

[49]

H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou,
“Training data-efficient image transformers & distillation through
attention,” 2021.

[50]

P. Vellaisamy, T. Labonte, S. Chakraborty, M. Turner, S. Sury, and J. P. Shen,
“Characterizing and optimizing llm inference workloads on cpu-gpu coupled
architectures,” in 2025 IEEE International Symposium on Performance
Analysis of Systems and Software (ISPASS). IEEE, 2025, pp. 49–61.

[51]

H. Wang, X. Xiao, M. Yan, Z. Zhu, D. Han, D. Wang, W. Li, X. Ye, C. Hu, H. Chen
et al., “A systematic characterization of llm inference on gpus,”
arXiv preprint arXiv:2512.01644, 2025.

[52]

X. Wang, M. Salmani, P. Omidi, X. Ren, M. Rezagholizadeh, and A. Eshaghi,
“Beyond the limits: A survey of techniques to extend the context length in
large language models,” arXiv preprint arXiv:2402.02244, 2024.

[53]

Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, and
Y. Ding, “{\{MLaaS}\} in the wild: Workload analysis and scheduling in
{\{Large-Scale}\} heterogeneous {\{GPU}\} clusters,” in 19th
USENIX Symposium on Networked Systems Design and Implementation (NSDI 22),
2022, pp. 945–960.

[54]

M. Wong, U. Butler, E. Farkash, P. Tammana, A. Sivaraman, and R. Netravali,
“Gpus, cpus, and… nics: Rethinking the network’s role in serving complex
ai pipelines,” arXiv preprint arXiv:2502.15712, 2025.

[55]

B. Wu, C. Xu, X. Dai, A. Wan, P. Zhang, Z. Yan, M. Tomizuka, J. Gonzalez,
K. Keutzer, and P. Vajda, “Visual transformers: Token-based image
representation and processing for computer vision,” 2020.

[56]

A. You, “Blockbpe: Parallel bpe tokenization,” arXiv preprint
arXiv:2507.11941, 2025.

[57]

C. Zhao, C. Deng, C. Ruan, D. Dai, H. Gao, J. Li, L. Zhang, P. Huang, S. Zhou,
S. Ma et al., “Insights into deepseek-v3: Scaling challenges and
reflections on hardware for ai architectures,” in Proceedings of the
52nd Annual International Symposium on Computer Architecture, 2025, pp.
1731–1745.

[58]

B. Zheng, C. H. Yu, J. Wang, Y. Ding, Y. Liu, Y. Wang, and G. Pekhimenko,
“Grape: Practical and efficient graphed execution for dynamic deep neural
networks on gpus,” in Proceedings of the 56th Annual IEEE/ACM
International Symposium on Microarchitecture, 2023, pp. 1364–1380.

[59]

L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis,
I. Stoica, J. E. Gonzalez et al., “Sglang: Efficient execution of
structured language model programs,” Advances in neural information
processing systems, vol. 37, pp. 62 557–62 583, 2024.

[60]

Y. Zheng, J. Fan, Q. Fu, Y. Yang, W. Zhang, and A. Quinn, “Agentcgroup:
Understanding and controlling os resources of ai agents,” arXiv
preprint arXiv:2602.09345, 2026.

[61]

Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang,
“{\{DistServe}\}: Disaggregating prefill and decoding for
goodput-optimized large language model serving,” in 18th USENIX
Symposium on Operating Systems Design and Implementation (OSDI 24), 2024,
pp. 193–210.

Experimental support, please
view the build logs
for errors. Generated by

L
A
T
E

xml

.

Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
methods listed below:

Click the "Report Issue" (

) button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
should not be a barrier to accessing research. Thank you for your continued support in championing open access for
all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

BETA