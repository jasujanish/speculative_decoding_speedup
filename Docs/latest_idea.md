## LTD
- LTD uses 2 MLPs, a depth network that outputs 0,1 at each step to determine whether to continue the tree and a size network to prune the width of the tree after generation. LTD passes the following information in its state vector: current draft depth (D), context length (L), predicted probabilities (P1 ... Pk)
- There are 2 significant issues with this: we have to train and run two policies & we fail to dynamically adjust tree width (which is controlled by a hyperparameter)

## Idea
- We train a single lightweight MLP policy that takes in a more comprehensive state vector
- At each draft layer, the MLP outputs a tuple $(f_m, c_m)$ that guides tree construction
- $f_m$ is the frontier retention ratio for the current layer.
  - Let the current frontier contain nodes with cumulative probabilities $s_1, \dots, s_n$.
  - We convert these scores into a normalized distribution over the frontier, so each node receives a relative weight and the weights sum to 1.
  - We sort frontier nodes by this normalized weight and select the smallest top-$k$ set whose cumulative weight is at least $f_m$.
  - Only those selected frontier nodes are expanded.
- $c_m$ is the child-mass retention ratio for each selected parent node.
  - For each expanded parent, we rank its candidate children by next-token probability.
  - We then keep the smallest set of top children with cumulative probability mass at least $c_m$.
  - Only those children are added to the next layer of the draft tree.

## Goal
- LTD uses the following goal: $\frac{\text{tokens accepted}}{T_{\text{drafter model}} + T_{\text{accepeter model}}}$
- We use the following goal: $\frac{\text{tokens accepted}}{T_{\text{total}}}$ to incorporate all time requirements, including time required for calling the policy

## Training Procedure
- LTD uses PPO. PPO is on-policy and requires a large training time.
- We can ablate the following training procedures
1. PPO
2. Build an oracle labeler and do basic imitation learning
3. Soft Actor Critics 
- Methods 2, 3 should be faster than 1

## Improvements
- LTD only stops the entire tree construction at a certain depth. This policy is able to terminate low probability branches early while directly controlling the branching factor at each level.
- LTD requires you to train two policies. We cut policy training in half by using one unified policy
- LTD uses PPO; we may (potentially) use a more sample efficient training mechanism, resulting in training time being a fraction of the cost
