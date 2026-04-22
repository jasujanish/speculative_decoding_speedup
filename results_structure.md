**Main Structure**
- results: main directory of results
    - results/8b: directory of results for qwen3_8b
        - results/8b/LTD: directory of results for LTD using qwen3_8b
        - results/8b/SupervisedLearning: directory of results for supervised learning (base policy) using qwen3_8b
        - results/8b/SupervisedLearning/Titan: directory of results for supervised learning (titan policy) using qwen3_8b
    - results/14b: directory of results for qwen3_14b
        - results/14b/LTD: directory of results for LTD using qwen3_14b
        - results/14b/SupervisedLearning: directory of results for supervised learning (base policy) using qwen3_14b
        - results/14b/SupervisedLearning/Titan: directory of results for supervised learning (titan policy) using qwen3_14b

**Directory Naming Scheme**
- LTD: {steps_per_phase}_{coop OR notcoop}
- Supervised Learning: {steps_per_phase}_{epochs}

**Example**
- 'results/8b/LTD/10k_notcoop'
    - qwen3_8b using LTD with 10k time steps per phase before co-optimization