from ml_test_vectors.types import StatelessDerivativeFunction, StatelessFunction
from ml_test_vectors.utils import assert_allclose, test_vector_from_json


def check_forward(
    test_vector_file_path: str,
    forward: StatelessFunction,
) -> None:
    with open(test_vector_file_path, "r", encoding="utf8") as file:
        test_vector = test_vector_from_json(file.read())
        for inp, expected in zip(test_vector["inputs"], test_vector["outputs"]):
            output = forward(inp)
            assert_allclose(
                output,
                expected,
                err_msg=f"Forward input was {inp}",
            )


def check_backward(
    test_vector_file_path: str,
    backward: StatelessDerivativeFunction,
) -> None:
    with open(test_vector_file_path, "r", encoding="utf8") as file:
        test_vector = test_vector_from_json(file.read())
        for inp, output, expected in zip(
            test_vector["inputs"], test_vector["outputs"], test_vector["gradients"]
        ):
            gradient = backward(inp, output)
            assert_allclose(
                gradient,
                expected,
                err_msg=f"Backward input was {inp}; output was {output}",
            )
