# quantum-encoding-pytorch-module

PyTorch implementation of a fixed, non-trainable quantum-encoding layer.

## Install

```bash
uv sync --extra test
```

## Use

```python
import torch

from quantum_encoding import QuantumEncodingLayer, quantum_encode

layer = QuantumEncodingLayer(num_qubits=3)
x = torch.rand(2, layer.input_dim)
complex_state = layer(x)
real_part, imag_part = quantum_encode(x[0])
```

## Benchmark

```bash
uv sync --extra test --extra benchmark
quantum-encoding-benchmark
```

The default benchmark suite uses `moons`, `breast_cancer`, `digits`, and the local `creditcard.csv` file at the project root.
