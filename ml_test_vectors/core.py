from typing import Callable, List, Optional, TypedDict, Union

import torch

# TODO: Figure out if dimensions can be parameterized in mypy so we can escape a maximum
Scalar = float
Vector = List[float]
Matrix = List[Vector]
Tensor4 = List[Matrix]
Tensor5 = List[Tensor4]
Tensor = Union[Scalar, Vector, Matrix, Tensor4, Tensor5]

StatelessFunction = Callable[[Tensor], Tensor]
StatelessDerivativeFunction = Callable[[Tensor, Optional[Tensor]], Tensor]
TorchStatelessFunction = Callable[[torch.Tensor], torch.Tensor]


class TestVector(TypedDict):
    inputs: List[Tensor]
    outputs: List[Tensor]
    gradients: List[Optional[Tensor]]
