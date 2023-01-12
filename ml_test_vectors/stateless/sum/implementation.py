from typing import Optional

import numpy as np

from ml_test_vectors.types import Tensor


def forward(inp: Tensor) -> Tensor:
    x = np.array(inp, dtype="float32")
    return x.sum()


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp)
