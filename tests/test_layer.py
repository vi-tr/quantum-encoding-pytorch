from __future__ import annotations

import torch

from quantum_encoding import QuantumEncodingLayer, quantum_encode
from tests.reference import run_layer as reference_run_layer


def test_layer_matches_reference_small_cases():
    for num_qubits in (1, 2, 3, 4):
        layer = QuantumEncodingLayer(num_qubits=num_qubits)
        x = torch.rand(layer.input_dim, dtype=torch.float64)

        expected_real, expected_imag = reference_run_layer(x.tolist())
        expected = torch.complex(
            torch.tensor(expected_real, dtype=torch.float64),
            torch.tensor(expected_imag, dtype=torch.float64),
        )

        actual = layer(x)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_layer_supports_batched_inputs_and_features():
    layer = QuantumEncodingLayer(num_qubits=3)
    x = torch.rand(5, layer.input_dim, dtype=torch.float32)

    output = layer(x)
    assert output.shape == (5, layer.state_dim)
    assert output.dtype == torch.complex64

    features = layer.forward_features(x)
    assert features.shape == (5, 2 * layer.state_dim)
    assert torch.allclose(features[..., : layer.state_dim], output.real)
    assert torch.allclose(features[..., layer.state_dim :], output.imag)


def test_no_trainable_parameters():
    layer = QuantumEncodingLayer(num_qubits=2)
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 0


def test_quantum_encode_matches_reference_signature():
    x = torch.rand(14, dtype=torch.float64)

    expected_real, expected_imag = reference_run_layer(x.tolist())
    actual_real, actual_imag = quantum_encode(x)

    assert torch.allclose(actual_real, torch.tensor(expected_real, dtype=torch.float64))
    assert torch.allclose(actual_imag, torch.tensor(expected_imag, dtype=torch.float64))
