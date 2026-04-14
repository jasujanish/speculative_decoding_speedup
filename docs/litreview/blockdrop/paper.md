# BlockDrop: Dynamic Inference Paths in Residual Networks

**Paper ID:** arXiv:1711.08393

**Submission Date:** November 22, 2017 (Last revised January 28, 2019)

**Venue:** CVPR 2018

## Authors
- Zuxuan Wu
- Tushar Nagarajan
- Abhishek Kumar
- Steven Rennie
- Larry S. Davis
- Kristen Grauman
- Rogerio Feris

## Abstract

The paper presents BlockDrop, a method for optimizing deep neural network inference.
The approach "learns to dynamically choose which layers of a deep network to execute
during inference so as to best reduce total computation without degrading prediction
accuracy." By leveraging ResNets' resilience to layer dropping, the framework
intelligently selects residual blocks to evaluate per image.

The authors train a policy network using reinforcement learning to balance
computational efficiency with accuracy preservation. Testing on CIFAR and ImageNet
datasets, they demonstrate that "these learned policies not only accelerate inference
but also encode meaningful visual information." On ImageNet using ResNet-101, the
method achieves approximately 20% average speedup (up to 36% for certain images)
while maintaining 76.4% top-1 accuracy.

## Subject Classification
- Computer Vision and Pattern Recognition (cs.CV)
- Machine Learning (cs.LG)

---

Note: Full paper content is available in paper.pdf (downloaded from https://arxiv.org/pdf/1711.08393).
HuggingFace Hub does not index this 2017 paper, and arXiv HTML format is not available for it.
