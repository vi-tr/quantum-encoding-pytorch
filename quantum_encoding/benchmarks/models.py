from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from quantum_encoding import QuantumEncodingLayer


def make_head(
    feature_dim: int, num_classes: int, hidden_dim: int = 128
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, num_classes),
    )


class PlainMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.head = make_head(input_dim, num_classes, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class QuantumFeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        num_params = input_dim // 2
        state_dim = num_params + 1
        num_qubits = int(math.log2(state_dim))
        if 2**num_qubits != state_dim:
            raise ValueError(
                "Quantum classifier requires a quantum-compatible input_dim."
            )
        self.encoder = QuantumEncodingLayer(num_qubits=num_qubits)
        self.head = make_head(2 * self.encoder.state_dim, num_classes, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder.forward_features(x))


class FixedReservoirLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 64,
        *,
        leak_rate: float = 0.5,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if reservoir_dim < 1:
            raise ValueError("reservoir_dim must be positive.")

        generator = torch.Generator().manual_seed(seed)
        w_in = torch.randn(reservoir_dim, 1, generator=generator) * input_scale
        w = torch.randn(reservoir_dim, reservoir_dim, generator=generator)
        radius = torch.linalg.eigvals(w).abs().max().real.clamp_min(1e-6)
        w = w * (spectral_radius / radius)

        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.register_buffer("w_in", w_in)
        self.register_buffer("w", w)
        self.register_buffer("bias", torch.zeros(reservoir_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input shape (batch, features).")

        batch_size, seq_len = x.shape
        state = torch.zeros(
            batch_size, self.reservoir_dim, device=x.device, dtype=x.dtype
        )
        running_state = torch.zeros_like(state)

        for step in range(seq_len):
            u = x[:, step : step + 1]
            preact = torch.nn.functional.linear(
                state, self.w
            ) + torch.nn.functional.linear(u, self.w_in, self.bias)
            updated = torch.tanh(preact)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * updated
            running_state = running_state + state

        mean_state = running_state / seq_len
        return torch.cat((state, mean_state), dim=-1)


class ReservoirFeatureClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        reservoir_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = FixedReservoirLayer(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            seed=seed,
        )
        self.head = make_head(2 * reservoir_dim, num_classes, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))
