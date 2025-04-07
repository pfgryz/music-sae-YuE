default: prepare-env

prepare-env:
    uvx pre-commit install
    uv sync