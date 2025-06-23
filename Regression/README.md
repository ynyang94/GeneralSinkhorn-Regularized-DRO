# SinkhornDRO

This repository contains the experimental code for regression tasks. It includes implementations of various distributionally robust optimization (DRO) formulations, including Sinkhorn-based methods and f-divergence regularization.

## 📁 Project Structure

├── SinkhornDRO.py\
├── sinkhorn_base.py\
├── fDRO.py\
├── LinearERM.py\
└── Python classes for:\
• Generalized Sinkhorn Distance-regularized DRO\
• Constrained Sinkhorn DRO\
• f-divergence regularized DRO\
• Empirical Risk Minimization (ERM)

├── DataGenerator.py\
├── Regression.py\
└── Experimental code for linear regression on synthetic data

├── logistic.py\
├── logistic_test.py\
└── Training and evaluation code for logistic regression


## 🧪 Usage Notes

- **Checkpoint Loading:** To load pretrained checkpoints, ensure that all DRO class initializations in the test files match those used during training.
- **Checkpoint Saving:** Modify the checkpoint paths to valid local or virtual directories before running any training scripts.

## 🛠 Environment Requirements

- Python 3.8
- PyTorch
- Matplotlib
- Numpy
- Scipy
- Pandas
