"""Run the converter as a module: ``python -m audiotokenizer`` is an alias for the console script."""

from __future__ import annotations

from audiotokenizer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
