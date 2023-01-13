import json
import os
from functools import partial
from operator import itemgetter

import numpy as np

from ml_test_vectors.core import TestVector


def test_vector_from_json(serialized: str) -> TestVector:
    parsed = json.loads(serialized)
    inputs, outputs, gradients, = itemgetter(
        "inputs", "outputs", "gradients"
    )(parsed)
    return TestVector(
        inputs=inputs,
        outputs=outputs,
        gradients=gradients,
    )


def get_test_vector_file_path(sibling_file_path: str) -> str:
    dirname = os.path.dirname(os.path.realpath(sibling_file_path))
    return os.path.join(dirname, "test_vector.json")


assert_allclose = partial(
    np.testing.assert_allclose, atol=1.0e-7, rtol=1.0e-7, equal_nan=True
)
