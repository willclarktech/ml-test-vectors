from typing import Callable, List, Optional, TypeAlias, TypedDict, Union

import torch

TorchStatelessFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

# TODO: Figure out if dimensions can be parameterized in mypy so we can escape a maximum
Scalar: TypeAlias = float
Vector: TypeAlias = List[float]
Matrix: TypeAlias = List[Vector]
Tensor4: TypeAlias = List[Matrix]
Tensor5: TypeAlias = List[Tensor4]
Tensor: TypeAlias = Union[Scalar, Vector, Matrix, Tensor4, Tensor5]

# TODO: Make generic with more specific input/output types
StatelessFunction: TypeAlias = Callable[[Tensor], Tensor]
StatelessDerivativeFunction: TypeAlias = Callable[[Tensor, Optional[Tensor]], Tensor]

StatelessLayer = TypedDict(
    "StatelessLayer",
    {
        "forward": StatelessFunction,
        "backward": StatelessDerivativeFunction,
    },
)

TestVector = TypedDict(
    "TestVector",
    {
        "inputs": List[Tensor],
        "outputs": List[Tensor],
        "gradients": List[Optional[Tensor]],
    },
)
