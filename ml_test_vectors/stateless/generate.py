from typing import List

import torch

from ml_test_vectors.types import Tensor, TestVector, TorchStatelessFunction
from ml_test_vectors.utils import detorch


def generate_stateless_function_test_vector(
    forward: TorchStatelessFunction, raw_forward_inputs: List[Tensor]
) -> TestVector:
    forward_inputs = [
        torch.tensor(input, requires_grad=True) for input in raw_forward_inputs
    ]
    forward_outputs = [forward(input) for input in forward_inputs]
    # A simple way to generate valid inputs for the backward pass
    for output in forward_outputs:
        output.sum().backward()
    backward_outputs = [input.grad for input in forward_inputs]
    return TestVector(
        detorch(forward_inputs),
        detorch(forward_outputs),
        detorch(forward_outputs),
        detorch(backward_outputs),
    )
