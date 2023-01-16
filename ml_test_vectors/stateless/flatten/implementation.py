from functools import reduce
from typing import Optional

import numpy as np

from ml_test_vectors.core import Tensor, Vector


def forward(inp: Tensor) -> Vector:
    def reducer(flattened: Vector, t: Tensor) -> Vector:
        return flattened + forward(t)

    initial_vector: Vector = []
    return [inp] if isinstance(inp, float) else reduce(reducer, inp, initial_vector)


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp).tolist()
