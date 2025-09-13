"""Convolutional baseline and CNN→SNN hybrid models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

__all__ = ["CNNOnly", "CNNSNN"]


class CNNOnly(nn.Module):
    """Pure CNN with two conv layers followed by a fully connected head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNSNN(nn.Module):
    """CNN feature extractor with LIF spiking neurons."""

    def __init__(self, beta: float = 0.9) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.fc = nn.Linear(64 * 4 * 4, 10)

        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

    def init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        with torch.no_grad():
            cur1 = self.conv1(x)
            mem1 = torch.zeros((batch, *cur1.shape[1:]), device=x.device)
            cur1 = F.max_pool2d(cur1, 2)
            cur2 = self.conv2(cur1)
            mem2 = torch.zeros((batch, *cur2.shape[1:]), device=x.device)
            cur2 = F.max_pool2d(cur2, 2)
            cur3 = cur2.view(batch, -1)
            cur3 = self.fc(cur3)
            mem3 = torch.zeros((batch, cur3.shape[1]), device=x.device)
        return mem1, mem2, mem3

    def forward_step(
        self,
        x: torch.Tensor,
        mem1: torch.Tensor,
        mem2: torch.Tensor,
        mem3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cur1 = self.conv1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur1 = F.max_pool2d(spk1, 2)

        cur2 = self.conv2(cur1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur2 = F.max_pool2d(spk2, 2)

        cur3 = cur2.view(cur2.size(0), -1)
        cur3 = self.fc(cur3)
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem1, mem2, mem3
