# Development

## Setup

```bash
uv sync
```

## Development Commands

```bash
# Type check
uv run pyright .

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Auto-fix linting issues
uv run ruff check --fix .
```

## Before Committing

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright .
```
