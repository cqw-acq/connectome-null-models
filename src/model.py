"""A frozen-connectome reservoir.

The recurrent matrix is the connectome (or one of its null models) and never
changes. Only the input projection onto sensory neurons and the readout from
descending neurons are trained — the setup described by the Biological
Processing Units paper (arXiv:2507.10951).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConnectomeReservoir(nn.Module):
    def __init__(
        self,
        A: np.ndarray,
        inp_idx: np.ndarray,
        out_idx: np.ndarray,
        n_in: int = 784,
        n_classes: int = 10,
        steps: int = 8,
        leak: float = 0.4,
    ):
        super().__init__()
        self.steps, self.leak = steps, leak
        self.register_buffer("A", torch.from_numpy(A))
        self.register_buffer("inp_idx", torch.from_numpy(inp_idx.astype(np.int64)))
        self.register_buffer("out_idx", torch.from_numpy(out_idx.astype(np.int64)))
        self.n = A.shape[0]

        # The only trainable parameters in the whole model.
        self.inject = nn.Linear(n_in, len(inp_idx))
        self.readout = nn.Linear(len(out_idx) * steps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        drive = self.inject(x)                       # [b, n_sensory]
        h = x.new_zeros(b, self.n)
        taps = []
        for _ in range(self.steps):
            pre = h @ self.A.T
            pre = pre.index_add(1, self.inp_idx, drive)
            h = (1 - self.leak) * h + self.leak * torch.tanh(pre)
            taps.append(h[:, self.out_idx])
        return self.readout(torch.cat(taps, dim=1))


class DenseBaseline(nn.Module):
    """Same trainable budget, no connectome: a plain MLP through a random frozen layer."""

    def __init__(self, n_hidden: int, n_in: int = 784, n_classes: int = 10, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n_hidden, n_hidden, generator=g) / np.sqrt(n_hidden)
        self.register_buffer("W", w)
        self.inject = nn.Linear(n_in, n_hidden)
        self.readout = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(torch.tanh(self.inject(x) @ self.W.T))
