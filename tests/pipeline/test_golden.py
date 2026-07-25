from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig

GOLDEN = Path(__file__).parent.parent / "data"


def golden_signal() -> NDArray[np.float64]:
    """The fixed signal both reference modules were compiled from."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def golden_configs() -> dict[str, TokenizerConfig]:
    """The two configurations pinned byte-for-byte: a plain IT module and a 16-bit-tempo XM module."""
    return {
        "golden.it": TokenizerConfig.load(profile="hacked", format="it", tempo=125, n_atoms=16),
        "golden.xm": TokenizerConfig.load(profile="hacked", format="xm", tempo=441, n_atoms=16),
    }


@pytest.mark.parametrize("name", sorted(golden_configs()))
def test_compiled_bytes_match_the_reference_module(name: str) -> None:
    """Pin the compiler's output so refactoring the writers cannot silently change what it emits.

    The references were captured before the `trackmod` extraction; any diff here means the migration
    changed the bytes rather than only their provenance.
    """
    compiled = compile_signal(golden_signal(), golden_configs()[name])
    assert compiled.to_bytes() == (GOLDEN / name).read_bytes()
