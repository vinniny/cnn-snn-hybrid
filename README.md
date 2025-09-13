# CNN-SNN Hybrid for MNIST

This project demonstrates a hybrid neural network that combines a convolutional neural network (CNN) with a spiking neural network (SNN) head using [`snnTorch`](https://snntorch.readthedocs.io). The CNN extracts spatial features from MNIST images while a layer of Leaky Integrate-and-Fire (LIF) neurons introduces temporal dynamics. A conventional CNN baseline is also provided for comparison.

## Setup

Create and activate a Python 3.11 virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Run scripts from the repository root.

### CNN baseline

```bash
python src/train.py --model cnn --epochs 10
```

After roughly 10 epochs the baseline achieves >97% accuracy on MNIST.

### CNN→SNN hybrid

```bash
python src/train.py --model hybrid --epochs 10 --timesteps 20
```

The hybrid model reaches accuracy within ~1–3% of the baseline when using 10–25 time steps (`--timesteps`). Training a hybrid model automatically saves a spike raster plot under `results/spike_raster.png` for a sample test image.

## Outputs

- Trained weights are stored under `results/cnn_baseline.pt` or `results/hybrid.pt`.
- Training curves are saved to `results/train_curve_<model>.png`.
- Final metrics (`test_loss` and `test_accuracy`) are written to `results/metrics.json`.

## Notes

- Uses the GPU if available; otherwise defaults to CPU.
- Dataset downloads require an internet connection.
