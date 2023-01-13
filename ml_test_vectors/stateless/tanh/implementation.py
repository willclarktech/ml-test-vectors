from typing import Optional

import numpy as np

from ml_test_vectors.core import Tensor


def forward(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return np.tanh(x, dtype="float32").tolist()


def backward(inp: Tensor, output: Optional[Tensor]) -> Tensor:
    outp = forward(inp) if output is None else output
    x = np.array(outp, dtype="float32")
    return (1 - x**2).tolist()
