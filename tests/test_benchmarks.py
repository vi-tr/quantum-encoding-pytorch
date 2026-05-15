from __future__ import annotations

import torch

from quantum_encoding.benchmarks.models import (
    FixedReservoirLayer,
    PlainMLPClassifier,
    QuantumFeatureClassifier,
    ReservoirFeatureClassifier,
)


def test_plain_mlp_forward_shape() -> None:
    model = PlainMLPClassifier(input_dim=4, num_classes=3)
    x = torch.randn(5, 4)
    y = model(x)
    assert y.shape == (5, 3)


def test_quantum_classifier_forward_shape() -> None:
    model = QuantumFeatureClassifier(input_dim=14, num_classes=3)
    x = torch.randn(5, 14)
    y = model(x)
    assert y.shape == (5, 3)


def test_reservoir_layer_and_classifier_forward_shape() -> None:
    reservoir = FixedReservoirLayer(input_dim=4, reservoir_dim=8, seed=0)
    x = torch.randn(5, 4)
    features = reservoir(x)
    assert features.shape == (5, 16)

    model = ReservoirFeatureClassifier(
        input_dim=4, num_classes=3, reservoir_dim=8, seed=0
    )
    y = model(x)
    assert y.shape == (5, 3)
