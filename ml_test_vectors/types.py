from typing import Callable, List, Sequence, TypedDict, Union

import torch

Scalar = float
Tensor = Union[Scalar, Sequence["Tensor"]]

StatelessFunction = Callable[[Tensor], Tensor]
StatelessDerivativeFunction = Callable[[Tensor, Tensor], Tensor]
TorchStatelessFunction = Callable[[torch.Tensor], torch.Tensor]


class TestVector(TypedDict):
    inputs: List[Tensor]
    outputs: List[Tensor]
    gradients: List[Tensor]
