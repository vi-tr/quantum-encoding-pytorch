from __future__ import annotations

import torch

from quantum_encoding import QuantumEncodingLayer


def main() -> None:
    torch.manual_seed(0)

    num_qubits = 3
    layer = QuantumEncodingLayer(num_qubits=num_qubits)

    batch_size = 2
    input_dim = layer.input_dim
    x = torch.rand(batch_size, input_dim, dtype=torch.float32)

    complex_state = layer(x)
    real_imag_features = layer.forward_features(x)

    print(f"input shape: {tuple(x.shape)}")
    print(f"complex output shape: {tuple(complex_state.shape)}")
    print(f"real/imag feature shape: {tuple(real_imag_features.shape)}")
    print(
        f"trainable parameters: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}"
    )


if __name__ == "__main__":
    main()
