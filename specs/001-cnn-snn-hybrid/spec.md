## Spec

Build a hybrid image classifier for MNIST using PyTorch and snnTorch.

- Baseline CNN for reference accuracy.
- Hybrid CNN→SNN model where a CNN extracts spatial features and a LIF
  spiking head provides temporal dynamics.
- Dropout regularization on convolutional blocks.
- Training script logs loss/accuracy curves, produces spike raster plots,
  and saves model weights and metrics.
- Target: baseline ≥97% accuracy on MNIST, hybrid within ~1–3% using
  10–25 time steps.

Out of scope: neuromorphic hardware or Jetson deployment.

