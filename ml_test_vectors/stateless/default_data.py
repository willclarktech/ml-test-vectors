from typing import List

from ml_test_vectors.types import Tensor

default_input_data: List[Tensor] = [
    [0.0],
    [-0.1],
    [0.1],
    [-1.0],
    [1.0],
    [-2.0],
    [2.0],
    [-10.0],
    [10.0],
    [-1.0, 0.0, 1.0],
    [
        [-1.0, 0.0, 1.0],
        [2.0, 4.0, 8.0],
    ],
    [
        [
            [-1.0, 0.0, 1.0],
            [2.0, 4.0, 8.0],
        ],
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    ],
]
