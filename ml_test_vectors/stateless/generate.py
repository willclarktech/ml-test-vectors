from typing import List

import torch

from ml_test_vectors.core import Tensor, TestVector, TorchStatelessFunction


def generate_stateless_function_test_vector(
    forward: TorchStatelessFunction, raw_inputs: List[Tensor]
) -> TestVector:
    inputs = [torch.tensor(inp, requires_grad=True) for inp in raw_inputs]
    outputs = [forward(inp) for inp in inputs]
    # A simple way to generate gradients
    for output in outputs:
        output.sum().backward()
    gradients = [inp.grad for inp in inputs]
    return TestVector(
        inputs=[input.tolist() for input in inputs],
        outputs=[output.tolist() for output in inputs],
        gradients=[
            None if gradient is None else gradient.tolist() for gradient in gradients
        ],
    )
