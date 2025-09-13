"""Model definitions for CNN baseline and CNN→SNN hybrid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class CNN(nn.Module):
    """Baseline CNN for MNIST.

    Architecture:
    Conv(1→12, k=5) → ReLU → MaxPool → Conv(12→64, k=5) → ReLU → MaxPool →
    Flatten → Linear(64*4*4→10).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.drop1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNSNN(nn.Module):
    """CNN feature extractor with LIF spiking head.

    Follows the same convolutional stack as :class:`CNN` but replaces ReLU
    activations with LIF neurons to introduce temporal dynamics.
    """

    def __init__(self, T: int = 20, beta: float = 0.9) -> None:
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(64 * 4 * 4, 10)

        spike_grad = surrogate.fast_sigmoid()
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return spike recordings of shape ``[T, batch, 10]``."""

        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky(batch_size, (12, 24, 24), x.device)
        mem2 = self.lif2.init_leaky(batch_size, (64, 8, 8), x.device)
        mem3 = self.lif3.init_leaky(batch_size, 10, x.device)

        spk_rec = []
        for _ in range(self.T):
            cur1 = F.max_pool2d(self.conv1(x), 2)
            cur1 = self.drop1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            cur2 = self.drop2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = spk2.view(batch_size, -1)
            cur3 = self.fc(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

