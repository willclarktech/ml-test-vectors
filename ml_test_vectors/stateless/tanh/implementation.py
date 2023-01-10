import numpy as np

from ml_test_vectors.types import Tensor


def tanh(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return np.tanh(x, dtype="float32")


def tanh_(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return 1 - x**2
