"""Plain python implementations to compare against the optimized ones."""
from __future__ import annotations

from cmath import exp
from math import cos, pi, sin
from typing import Sequence


def encode_to_quantum_state(data: Sequence[float]) -> list[complex]:
    num_params = len(data) // 2

    delta = [value * pi / 2 for value in data[:num_params]]
    gamma = [value * 2 * pi for value in data[num_params:]]

    amplitudes = [0j] * (num_params + 1)

    amplitudes[-1] = sin(delta[-1]) * exp(1j * gamma[-1])
    running_cos_product = cos(delta[-1])
    for index in range(num_params - 2, -1, -1):
        amplitudes[index + 1] = (
            sin(delta[index]) * running_cos_product * exp(1j * gamma[index])
        )
        running_cos_product *= cos(delta[index])

    amplitudes[0] = running_cos_product
    return amplitudes


def inverse_gray_code(index: int) -> int:
    value = index
    while index:
        index >>= 1
        value ^= index
    return value


def simulate_optimized_circuit(psi: Sequence[complex]) -> list[complex]:
    num_amplitudes = len(psi)
    half = num_amplitudes // 2

    permuted = [0j] * num_amplitudes
    for index, amplitude in enumerate(psi):
        permuted[inverse_gray_code(index)] = amplitude

    transformed = [0j] * num_amplitudes
    scale = 1 / 2**0.5
    for index in range(half):
        first = permuted[index]
        second = permuted[index + half]
        transformed[index] = (first + second) * scale
        transformed[index + half] = (first - second) * scale

    return transformed


def run_layer(data: Sequence[float]) -> tuple[list[float], list[float]]:
    result = simulate_optimized_circuit(encode_to_quantum_state(data))
    return [value.real for value in result], [value.imag for value in result]
