from functools import reduce
from typing import Optional

import numpy as np

from ml_test_vectors.types import Tensor


def forward(inp: Tensor) -> Tensor:
    return (
        inp
        if isinstance(inp, float) or isinstance(inp[0], float)
        else reduce(lambda flattened, t: flattened + forward(t), inp, [])
    )


def backward(inp: Tensor, _output: Optional[Tensor]) -> Tensor:
    return np.ones_like(inp)
