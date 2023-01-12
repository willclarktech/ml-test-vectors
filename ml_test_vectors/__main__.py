import sys

from ml_test_vectors.parser import create_parser
from ml_test_vectors.runner import functions, run

parser = create_parser(functions)
options = vars(parser.parse_args())
show_trace = options.pop("trace", False)

try:
    run(options)
# pylint: disable=broad-except
except Exception as exception:
    if show_trace:
        raise exception
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
