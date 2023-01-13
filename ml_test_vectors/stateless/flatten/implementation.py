from functools import reduce
from typing import Optional

import numpy as np

from ml_test_vectors.core import Tensor


def forward(inp: Tensor) -> Tensor:
    def reducer(flattened: Tensor, t: Tensor) -> Tensor:
        if isinstance(flattened, float):
            raise ValueError("Invalid tensor")
        flattened_t = forward(t)
        if isinstance(flattened_t, float):
            raise ValueError("Invalid tensor")
        return flattened + flattened_t

    initial_tensor: Tensor = []
    return (
        inp
        if isinstance(inp, float) or len(inp) == 0 or isinstance(inp[0], float)
        else reduce(reducer, inp, initial_tensor)
    )


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp).tolist()
