from functools import reduce
from typing import Optional

import numpy as np

from ml_test_vectors.types import Scalar, Tensor


def forward(inp: Tensor) -> Scalar:
    return (
        inp
        if isinstance(inp, float)
        else reduce(lambda subtotal, m: subtotal + forward(m), inp, 0)
    )


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp)
