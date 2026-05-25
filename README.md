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