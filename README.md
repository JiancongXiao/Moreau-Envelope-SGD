# Uniformly Stable Algorithms for Adversarial Training and Beyond

This repository is the official PyTorch implementation of paper: Uniformly Stable Algorithms for Adversarial Training and Beyond.

**Jiancong Xiao, Jiawei Zhang, Zhi-Quan Luo, Asuman Ozdaglar**

41th International Conference on Machine Learning (ICML 2024)

**arXiv:** [https://arxiv.org/abs/2405.01817](https://arxiv.org/abs/2405.01817) 

**OpenReview:** [https://openreview.net/forum?id=odCl49tWA6](https://openreview.net/forum?id=odCl49tWA6)

## Abstract

In adversarial machine learning, neural networks suffer from a significant issue known as robust overfitting, where the robust test accuracy decreases over epochs (Rice et al., 2020). Recent research conducted by Xing et al., 2021;Xiao et al., 2022 has focused on studying the uniform stability of adversarial training. Their investigations revealed that SGD-based adversarial training fails to exhibit uniform stability, and the derived stability bounds align with the observed phenomenon of robust overfitting in experiments. This finding motivates us to develop uniformly stable algorithms specifically tailored for adversarial training. To this aim, we introduce Moreau envelope-$\mathcal{A}$ (ME-$\mathcal{A}$), a variant of the Moreau Envelope-type algorithm. We employ a Moreau envelope function to reframe the original problem as a min-min problem, separating the non-strong convexity and non-smoothness of the adversarial loss. Then, this approach alternates between solving the inner and outer minimization problems to achieve uniform stability without incurring additional computational overhead. In practical scenarios, we demonstrate the efficacy of ME-$\mathcal{A}$ in mitigating the issue of robust overfitting. Beyond its application in adversarial training, this represents a fundamental result in uniform stability analysis, as ME-$\mathcal{A}$ is the first algorithm to exhibit uniform stability for weakly-convex, non-smooth problems.

## Code

The main argument of MEA is '--rho', default=None, type=float, help='MEA, coefficient of proximal term'.

For MEA, rho>0. When rho is none, the algorithm is reduced to SWA. The code is adopted from two repositories [imrahulr](https://github.com/imrahulr/adversarial_robustness_pytorch) and [Rice et al., 2020](https://github.com/locuslab/robust_overfitting).

## Citation
```
@inproceedings{
xiao2024uniformly,
title={Uniformly Stable Algorithms for Adversarial Training and Beyond},
author={Jiancong Xiao and Jiawei Zhang and Zhi-Quan Luo and Asuman E. Ozdaglar},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=odCl49tWA6}
}
```
