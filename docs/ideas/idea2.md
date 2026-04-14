## Architecture
- LTD uses 2 MLPs, a depth network that outputs 0,1 at each step to determine whether to continue the tree and a size network to prune the width of the tree after generation. LTD passes the following information in its state vector: current draft depth (D), context length (L), predicted probabilities (P1 ... Pk)
- Idea 1: We use a network that runs on each node, and the network provides a budget for each node. We take the floor of this budget to determine how many children node to generate for that node. As such, the network dynamically prunes depth and width. We will experiment with an MLP and SSM based architecture. We pass the following information in the state vector: current draft depth, current context length, budget allocated across previous layers, cumulative log probabilities, and next token probabilities.
- Idea 2: We use a network that runs on each layer. The network provides a threshold for the layer, and only tokens with a probability greater than the threshold are added as nodes. As such, the network dynamically prunes depth and width. We will experiment with an MLP and SSM based architecture. We pass the following information in the state vector: current draft depth, current context length, budget allocated across previous layers, cumulative log probabilities, and next token probabilities.

## Goal
- LTD uses the following goal: $\frac{\text{tokens accepted}}{T_{\text{drafter model}} + T_{\text{accepeter model}}}$
- We use the following goal: $\frac{\text{tokens accepted}}{T_{\text{total}}}$ to incorporate all time requirements, including time required for calling the policy

## Training Procedure
- LTD uses PPO. PPO is on-policy and requires a large training time.
- We build an oracle labeler and do offline training with DAgger. We then add PPO fine-tuning.

