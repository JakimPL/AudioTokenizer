.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run mypy
	uv run pylint audiotokenizer

.PHONY: format
format:
	uv run isort .
	uv run black .
