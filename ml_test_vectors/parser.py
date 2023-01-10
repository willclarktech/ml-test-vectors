import argparse
from typing import Iterable


def create_parser(functions: Iterable[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "ml_test_vectors",
        description="Generate test vectors for various stateless/stateful machine learning functions",
    )
    parser.add_argument("function", help=f"choose from: {', '.join(functions)}")
    return parser
