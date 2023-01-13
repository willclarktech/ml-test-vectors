import json
from operator import itemgetter
from typing import Callable, List, Sequence, Union

import torch

Scalar = float
Tensor = Union[Scalar, Sequence["Tensor"]]

StatelessFunction = Callable[[Tensor], Tensor]
StatelessDerivativeFunction = Callable[[Tensor, Tensor], Tensor]
TorchStatelessFunction = Callable[[torch.Tensor], torch.Tensor]


class TestVector:
    def __init__(
        self,
        inputs: List[Tensor],
        outputs: List[Tensor],
        gradients: List[Tensor],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.gradients = gradients

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


def test_vector_from_json(serialized: str) -> TestVector:
    parsed = json.loads(serialized)
    inputs, outputs, gradients, = itemgetter(
        "inputs", "outputs", "gradients"
    )(parsed)
    return TestVector(
        inputs,
        outputs,
        gradients,
    )
