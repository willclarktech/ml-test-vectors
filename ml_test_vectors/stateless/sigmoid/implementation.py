import numpy as np

from ml_test_vectors.types import Tensor


def sigmoid(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return 1 / (1 + np.exp(-x, dtype="float32"))


def sigmoid_(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return x * (1 - x)
