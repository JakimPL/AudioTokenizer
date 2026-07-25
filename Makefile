.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run mypy
	uv run pylint audiotokenizer trackmod

.PHONY: format
format:
	uv run isort .
	uv run black .
