import json
from operator import itemgetter
from typing import Callable, List, Sequence, Union

import torch

Tensor = Union[Sequence[float], Sequence["Tensor"]]

StatelessFunction = Callable[[Tensor], Tensor]
TorchStatelessFunction = Callable[[torch.Tensor], torch.Tensor]


class TestVector:
    def __init__(
        self,
        forward_inputs: List[Tensor],
        forward_outputs: List[Tensor],
        backward_inputs: List[Tensor],
        backward_outputs: List[Tensor],
    ) -> None:
        self.forward_inputs = forward_inputs
        self.forward_outputs = forward_outputs
        self.backward_inputs = backward_inputs
        self.backward_outputs = backward_outputs

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


def test_vector_from_json(serialized: str) -> TestVector:
    parsed = json.loads(serialized)
    forward_inputs, forward_outputs, backward_inputs, backward_outputs = itemgetter(
        "forward_inputs", "forward_outputs", "backward_inputs", "backward_outputs"
    )(parsed)
    return TestVector(
        forward_inputs, forward_outputs, backward_inputs, backward_outputs
    )
