# Towards Understanding Sharpness-Aware Minimization 

**Maksym Andriushchenko (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2206.06232](https://arxiv.org/abs/2206.06232)

**ICML 2022**


## Abstract
Sharpness-Aware Minimization (SAM) is a recent training method that relies on worst-case weight perturbations which significantly improves generalization in various settings. We argue that the existing justifications for the success of SAM which are based on a PAC-Bayes generalization bound and the idea of convergence to flat minima are incomplete. Moreover, there are no explanations for the success of using m-sharpness in SAM which has been shown as essential for generalization. To better understand this aspect of SAM, we theoretically analyze its implicit bias for diagonal linear networks. We prove that SAM always chooses a solution that enjoys better generalization properties than standard gradient descent for a certain class of problems, and this effect is amplified by using m-sharpness. We further study the properties of the implicit bias on non-linear networks empirically, where we show that fine-tuning a standard model with SAM can lead to significant generalization improvements. Finally, we provide convergence results of SAM for non-convex objectives when used with stochastic gradients. We illustrate these results empirically for deep networks and discuss their relation to the generalization behavior of SAM. 


## Code
We performed experiments on three different model families:
- Folder `deep_nets`: code for training deep networks with m-SAM. Note that one can use m lower than the batch size (implemented via gradient accumulation).
- Folder `diagonal_linear_nets`: code for training diagonal linear networks with 1-SAM and n-SAM.
- Folder `one_layer_relu_nets`: code for training one hidden layer ReLU networks with n-SAM.


## Dependencies
All the dependencies are collected in the `Dockerfile`.