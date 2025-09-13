## Plan

- Environment: WSL Ubuntu, Python 3.11 virtualenv.
- Libraries: PyTorch, snnTorch, torchvision, matplotlib.
- Model v1: Conv(1→12,k=5)-MaxPool-LIF(β≈0.9) → Conv(12→64,k=5)-MaxPool-LIF →
  Flatten → Linear(64×4×4→10)-LIF.
- Training: surrogate gradients with `ce_rate_loss`, Adam lr=1e-3,
  time steps `T=20`.
- Baseline: same conv stack but with ReLU activations and softmax head.
- Dropout (p≈0.2) after each convolutional block to curb overfitting.
- Optional Fashion-MNIST support and simple augmentation.
- Outputs: accuracy vs. baseline, training curves, spike raster plots,
  saved weights and metrics in `results/`.
- Layout: code under `src/`, datasets in `data/`, results in `results/`,
  planning documents under `specs/001-cnn-snn-hybrid/`.

