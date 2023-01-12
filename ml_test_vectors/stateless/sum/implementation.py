from functools import reduce
from typing import Optional

import numpy as np

from ml_test_vectors.types import Tensor


def forward(inp: Tensor) -> Tensor:
    return (
        inp
        if isinstance(inp, float)
        else reduce(lambda subtotal, n: subtotal + forward(n), inp, 0)
    )


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp)
