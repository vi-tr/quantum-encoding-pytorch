from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


def _infer_num_qubits(data_width: int) -> int:
    if data_width <= 0 or data_width % 2 != 0:
        raise ValueError("Input width must be a positive even integer.")

    num_params = data_width // 2
    state_dim = num_params + 1
    num_qubits = int(math.log2(state_dim))
    if 2**num_qubits != state_dim:
        raise ValueError(
            "Input width must be of the form 2 * (2**n - 1) for some integer n."
        )
    return num_qubits


def _gray_code_indices(state_dim: int) -> Tensor:
    indices = torch.arange(state_dim, dtype=torch.long)
    return indices ^ (indices >> 1)


class QuantumEncodingLayer(nn.Module):
    """Non-trainable layer for the quantum-encoding transform."""

    def __init__(self, num_qubits: int) -> None:
        super().__init__()
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")

        self.num_qubits = int(num_qubits)
        self.state_dim = 1 << self.num_qubits
        self.num_params = self.state_dim - 1
        self.input_dim = 2 * self.num_params

        self.register_buffer("gray_indices", _gray_code_indices(self.state_dim))
        self.register_buffer(
            "_sqrt2_inv",
            torch.tensor(1.0 / math.sqrt(2.0), dtype=torch.float64),
        )

        self.requires_grad_(False)

    @staticmethod
    def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
        return torch.complex128 if real_dtype == torch.float64 else torch.complex64

    def _encode_to_quantum_state(self, data: Tensor) -> Tensor:
        delta = data[..., : self.num_params] * (math.pi / 2.0)
        gamma = data[..., self.num_params :] * (2.0 * math.pi)

        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)

        suffix_cos = torch.cumprod(cos_delta.flip(-1), dim=-1).flip(-1)
        phase = torch.polar(torch.ones_like(gamma), gamma)

        complex_dtype = self._complex_dtype_for(data.dtype)
        encoded = torch.empty(
            (*data.shape[:-1], self.state_dim),
            dtype=complex_dtype,
            device=data.device,
        )

        encoded[..., 0] = suffix_cos[..., 0]
        if self.num_params > 1:
            encoded[..., 1 : self.num_params] = (
                sin_delta[..., :-1] * suffix_cos[..., 1:] * phase[..., :-1]
            )
        encoded[..., self.num_params] = sin_delta[..., -1] * phase[..., -1]
        return encoded

    def _apply_circuit(self, state: Tensor) -> Tensor:
        permuted = state.index_select(-1, self.gray_indices)
        half = self.state_dim // 2
        first = permuted[..., :half]
        second = permuted[..., half:]

        out = torch.empty_like(permuted)
        scale = self._sqrt2_inv.to(device=permuted.device, dtype=permuted.real.dtype)
        out[..., :half] = (first + second) * scale
        out[..., half:] = (first - second) * scale
        return out

    def forward(self, data: Tensor) -> Tensor:
        if not torch.is_tensor(data):
            raise TypeError("data must be a torch.Tensor")
        if data.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension to be {self.input_dim}, got {data.shape[-1]}."
            )
        if not data.is_floating_point():
            raise TypeError("data must have a floating-point dtype")

        input_tensor = data.to(
            dtype=torch.float64 if data.dtype == torch.float64 else torch.float32
        )
        quantum_state = self._encode_to_quantum_state(input_tensor)
        return self._apply_circuit(quantum_state)

    def forward_features(self, data: Tensor) -> Tensor:
        """Return concatenated real/imag features for real-valued ML pipelines."""
        encoded_state = self.forward(data)
        return torch.cat((encoded_state.real, encoded_state.imag), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"num_qubits={self.num_qubits}, input_dim={self.input_dim}, "
            f"state_dim={self.state_dim}"
        )


def quantum_encode(data: Tensor | Sequence[float]) -> tuple[Tensor, Tensor]:
    if torch.is_tensor(data):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float64)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)

    num_qubits = _infer_num_qubits(int(tensor.shape[-1]))
    layer = QuantumEncodingLayer(num_qubits=num_qubits)
    result = layer(tensor)
    return result.real, result.imag


def run_layer(data: Tensor | Sequence[float]) -> tuple[Tensor, Tensor]:
    return quantum_encode(data)
