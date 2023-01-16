from typing import Callable, List, Optional, TypedDict, Union

import torch

Scalar = float
Vector = List[float]
Tensor = Union[Scalar, List["Tensor"]]

StatelessFunction = Callable[[Tensor], Tensor]
StatelessDerivativeFunction = Callable[[Tensor, Tensor], Tensor]
TorchStatelessFunction = Callable[[torch.Tensor], torch.Tensor]


class TestVector(TypedDict):
    inputs: List[Tensor]
    outputs: List[Tensor]
    gradients: List[Optional[Tensor]]
