# SinkhornDRO for LeNet training and evaluation

This repository contains the experimental code for LeNet tasks. It includes implementations of various distributionally robust optimization (DRO) formulations, including Sinkhorn-based methods and f-divergence regularization.

## 📁 Project Structure
├── image_process_mnist.py\
├── DeepSinkhornDRO.py\
├── sinkhorn_base.py\
├── DeepfDRO.py\
├── DeepERM.py\
└── Python classes for:\
• Data Preparation
• Generalized Sinkhorn Distance-regularized DRO\
• Constrained Sinkhorn DRO\
• f-divergence regularized DRO\
• Empirical Risk Minimization (ERM)
├── train_pipeline.py\
├── test_and_attack.py\
├── LeNet_train.py\
├── LeNet_test.py\
└── contains:\
• training pipeline of each formulation\
• adversarial attack methods and evaluation pipeline
• train/test code for LeNet


## 🧪 Usage Notes

- **Checkpoint Loading:** To load pretrained checkpoints, ensure that all DRO class initializations in the test files match those used during training.
- **Checkpoint Saving:** Modify the checkpoint paths to valid local or virtual directories before running any training scripts.
- **Data Collection Path:** Modify the data collection paths to valid local or virtual directories before running any training scripts.

## 🛠 Environment Requirements

- Python 3.8
- PyTorch
- Matplotlib
- Numpy
- Scipy
- Pandas

