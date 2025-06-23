# Generalized Sinkhorn distance-regularized DRO

This repository contains the official implementation for the paper:

> **Nested Stochastic Algorithm for Generalized Sinkhorn distance-Regularized Distributionally Robust Optimization**  
> Yufeng Yang; Yi Zhou; Zhaosong Lu
> [arXiv:2503.22923](https://arxiv.org/abs/2503.22923)

This work proposes a novel dual formulation of the **Generalized Sinkhorn distance-regularized DRO** problem and introduces a **Nested Stochastic Gradient Descent (Nested SGD) algorithm** with provable convergence guarantees.

In this repository, we provide **PyTorch implementations** of our method along with several baseline algorithms.

## 🔄 Reproducing Results

To reproduce the experimental results, please follow the steps below:

1. **Install dependencies**  
   Make sure you have the following packages installed:
   - Python 3.8
   - PyTorch
   - Matplotlib
   - NumPy
   - SciPy
   - Pandas
2. **Modify the checkpoint paths in the scripts to point to valid local directories**


If you find this work useful, please consider citing the code:

```bibtex

@software{mysinkhorndro,
  author       = {Yufeng Yang and Yi Zhou and Zhaosong Lu},
  title        = {Sinkhorn-Regularized-DRO-v1.0},
  year         = 2025,
  doi          = {10.5281/zenodo.15723973},
  url          = {https://doi.org/10.5281/zenodo.15723973},
  note         = {GitHub repository.}
}
and the paper
```bibtex
@misc{yang2025nestedstochasticgradientdescent,
      title={Nested Stochastic Gradient Descent for (Generalized) Sinkhorn Distance-Regularized Distributionally Robust Optimization}, 
      author={Yufeng Yang and Yi Zhou and Zhaosong Lu},
      year={2025},
      eprint={2503.22923},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2503.22923}, 
}
