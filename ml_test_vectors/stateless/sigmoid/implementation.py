from typing import Optional

import numpy as np

from ml_test_vectors.types import Tensor


def forward(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return 1 / (1 + np.exp(-x, dtype="float32"))


def backward(inp: Tensor, output: Optional[Tensor]) -> Tensor:
    outp = forward(inp) if output is None else output
    x = np.array(outp, dtype="float32")
    return x * (1 - x)
