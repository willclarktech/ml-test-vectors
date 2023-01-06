# ML Test Vectors

Test vectors for ML functions.

## Prerequisites

- Python3.11

## Installation

Using poetry:

```sh
poetry install
```

Using pip:

```sh
pip install -r requirements.txt
```

Using pip to install development dependencies too:

```sh
pip install -r requirements.dev.txt
```

On Google Colab to avoid conflicts with preinstalled packages:

```sh
pip install -r requirements.colab.txt
```

## Running code

### CLI

An executable is provided in `./bin`. From the root directory run:

```sh
./bin/ml_test_vectors
```

This will also pass on additional arguments.

### Programmatic API

Use the exposed `hello` function:

```py
import ml_test_vectors

ml_test_vectors.hello()
```

### Notebooks

A notebook is provided in `./notebooks` which demonstrates how to use the programmatic API. The notebook provides a link to open in Google Colab. To run locally start a Jupyter notebook server and open the notebook in the browser window which should open automatically:

```sh
jupyter notebook
```

## Development

The following scripts assume the requirements have been installed. If using poetry, they assume `poetry shell` has already been run or else they should be prefixed with `poetry run`.

### Lint

```sh
pylint ./ml_test_vectors
```

### Typecheck

```sh
mypy
```

### Format

```sh
black ./ml_test_vectors
```

### Generating requirements files

```sh
./scripts/generate_requirements.sh
```
