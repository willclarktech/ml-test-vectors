import os

import ml_test_vectors.stateless.flatten.implementation as flatten
import ml_test_vectors.stateless.sigmoid.implementation as sigmoid
import ml_test_vectors.stateless.sum.implementation as my_sum
import ml_test_vectors.stateless.tanh.implementation as tanh
from ml_test_vectors.types import (
    StatelessDerivativeFunction,
    StatelessFunction,
    TestVector,
    test_vector_from_json,
)
from ml_test_vectors.utils import assert_allclose


def test(
    test_vector: TestVector,
    forward: StatelessFunction,
    backward: StatelessDerivativeFunction,
) -> None:
    for inp, expected in zip(test_vector.inputs, test_vector.outputs):
        output = forward(inp)
        assert_allclose(
            output,
            expected,
            err_msg=f"Forward input was {inp}",
        )
    for inp, output, expected in zip(
        test_vector.inputs, test_vector.outputs, test_vector.gradients
    ):
        gradient = backward(inp, output)
        assert_allclose(
            gradient,
            expected,
            err_msg=f"Backward input was {inp}; output was {output}",
        )


def test_fn(fn_name: str, fn) -> None:
    print(f"Testing {fn_name}")
    dirname = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dirname, fn_name, "test_vector.json")
    with open(file_path, "r", encoding="utf8") as file:
        test_vector = test_vector_from_json(file.read())
        test(test_vector, fn.forward, fn.backward)


functions = [
    ("flatten", flatten),
    ("sigmoid", sigmoid),
    ("sum", my_sum),
    ("tanh", tanh),
]


def main() -> None:
    for fn_name, fn in functions:
        test_fn(fn_name, fn)


main()
