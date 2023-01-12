import os
from functools import partial
from typing import List

import numpy as np
import torch

from ml_test_vectors.types import Tensor


def get_test_vector_file_path(sibling_file_path: str) -> str:
    dirname = os.path.dirname(os.path.realpath(sibling_file_path))
    return os.path.join(dirname, "test_vector.json")


def detorch(tensors: List[torch.Tensor]) -> Tensor:
    return [tensor.tolist() for tensor in tensors]


assert_allclose = partial(
    np.testing.assert_allclose, atol=1.0e-7, rtol=1.0e-7, equal_nan=True
)
