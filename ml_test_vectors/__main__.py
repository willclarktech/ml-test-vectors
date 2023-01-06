import sys

from ml_test_vectors.lib import hello

try:
    hello()
# pylint: disable=broad-except
except Exception as exception:
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
