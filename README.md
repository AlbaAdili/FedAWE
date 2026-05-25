# Federated Generative Learning under Client Unavailability
---
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

## Project Overview

This project reproduces and extends the NeurIPS 2024 paper:

> Efficient Federated Learning against Heterogeneous and Non-stationary Client Unavailability

Original paper:
- OpenReview: https://openreview.net/pdf?id=DLNOBJa7TM
- arXiv: https://arxiv.org/pdf/2409.17446

The original FedAWE paper focuses on federated classification under heterogeneous client availability. In this project, we first reproduce the original classification experiments and then extend the framework to federated generative learning using Variational Autoencoders (VAEs).

Our work investigates whether stale replay mechanisms used in FedAWE remain stable for generative models under severe client dropout.

---
# Main Contributions

## Reproduction of Original FedAWE

We reproduced the main FedAWE classification experiments on:
- CIFAR-10
- SVHN
- CINIC-10

using:
- FedAvg
- FedAWE
- FedVARP
- FedAU
- MIFA

under heterogeneous client availability dynamics.

---
## Extension to Federated Generative Learning

We extended FedAWE from classification to generative learning using VAEs trained on:
- MNIST
- FashionMNIST

This extension revealed that stale replay mechanisms become substantially less stable in generative learning.

---

## Quality Echo

We introduced Quality Echo, a replay stabilization mechanism that:
- exponentially downweights stale updates
- filters replayed updates using cosine similarity with fresh updates

This reduces the impact of outdated latent-space information.

---
## Selective Echo FedAvg-VAE

We further proposed Selective Echo FedAvg-VAE, which:
- starts from stable FedAvg aggregation
- selectively reuses stale updates
- controls replay using staleness thresholds
- stabilizes latent-space optimization under severe client dropout

This method significantly improves over vanilla FedAWE-VAE.

---