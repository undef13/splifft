check:
    uv run ruff check src tests
    uv run ruff format --check src tests
    # uv run --with pydantic mypy src tests

fmt:
    uv run ruff check src tests --fix
    uv run ruff format src tests
    pnpm run fmt:json src/splifft/data/registry.json

docs:
    uv run mkdocs serve

gen-schema:
    uv run scripts/gen_schema.py
