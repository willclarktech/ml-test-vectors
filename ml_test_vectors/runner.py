import json
import os
from typing import Any, Dict

from ml_test_vectors.stateless import (
    flatten,
    generate_stateless_function_test_vector,
    my_sum,
    sigmoid,
    tanh,
)

functions: Dict[str, Any] = {
    "flatten": flatten,
    "sigmoid": sigmoid,
    "sum": my_sum,
    "tanh": tanh,
}


def run(options: Dict[str, Any]) -> None:
    fn_name = options.get("function") or ""
    fn = functions.get(fn_name)
    if fn is None:
        raise ValueError(f"Function {fn_name or '(none)'} not recognized")
    test_vector = generate_stateless_function_test_vector(fn.torch_fn, fn.input_data)
    dirname = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dirname, "stateless", fn_name, "test_vector.json")
    with open(file_path, "w", encoding="utf8") as file:
        json.dump(
            json.loads(test_vector.to_json()),
            file,
        )
    print(f"Run! {options} {fn_name} {fn}")
