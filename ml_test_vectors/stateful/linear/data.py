from typing import Tuple, TypedDict

import torch

from ml_test_vectors.core import Matrix

LinearState = TypedDict("LinearState", {"shape": Tuple[int, int], "weights": Matrix})

empty_weights: Matrix = []
pre_state: LinearState = {"shape": (1, 1), "weights": []}
post_state: LinearState = {"shape": (1, 1), "weights": []}


def torch_fn(state: LinearState, inp: Tensor) -> Tensor:
    # Include bias
    linear_layer = torch.nn.Linear(*state["shape"], bias=False)
    with torch.no_grad():
        linear_layer.weight = torch.nn.Parameter(torch.tensor(state))
    return linear_layer(inp)
