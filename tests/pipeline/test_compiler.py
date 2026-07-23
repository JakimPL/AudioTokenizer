from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def test_compiled_size_matches_the_written_file(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16))
    assert compiled.cost.total == len(compiled.to_bytes())


def test_compilation_respects_channel_and_atom_caps(signal: NDArray[np.float64]) -> None:
    config = TokenizerConfig(profile="strict", tempo=150, n_atoms=24)
    compiled = compile_signal(signal, config)
    assert compiled.n_atoms_used <= config.n_atoms
    assert compiled.n_channels_used <= config.max_channels
    assert np.isfinite(compiled.metrics.mel_distance_db)
    assert compiled.metrics.mel_distance_db > 0.0


def test_hacked_profile_allows_more_than_64_channels() -> None:
    assert TokenizerConfig(profile="hacked", tempo=125, n_atoms=100).max_channels == 127
    assert TokenizerConfig(profile="strict", tempo=125, n_atoms=100).max_channels == 64


def test_too_many_atoms_is_rejected() -> None:
    # 128 atoms would need 256 stored samples, over the 255 the 1-byte note map can route.
    with pytest.raises(ValueError):
        TokenizerConfig(profile="hacked", tempo=125, n_atoms=128)
