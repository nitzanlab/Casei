# Contributing to Casei

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/casei-dev/casei.git
cd casei
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check casei/
ruff format casei/
```

## Pull Requests

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Make sure all tests pass and linting is clean.
4. Open a PR against `main` with a clear description of the change.

## Reporting Issues

Open an issue on GitHub with a minimal reproducible example and your environment details (`python --version`, `pip list`).
