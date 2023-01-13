from ml_test_vectors.stateless.sigmoid.implementation import backward, forward
from ml_test_vectors.stateless.testutils import check_backward, check_forward
from ml_test_vectors.utils import get_test_vector_file_path

test_vector_file_path = get_test_vector_file_path(__file__)


def test_forward() -> None:
    check_forward(test_vector_file_path, forward)


def test_backward() -> None:
    check_backward(test_vector_file_path, backward)
