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


def test_polarity_sticky_saves_note_bytes_and_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    plain = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16))
    sticky = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16, polarity_sticky=4))
    # Suppressing tiny-code sign flips can only remove note bytes, never add them.
    assert sticky.cost.pattern <= plain.cost.pattern
    # The size model stays byte-exact, and the estimate reflects what the sticky grids actually play.
    assert sticky.cost.total == len(sticky.to_bytes())
    assert np.isfinite(sticky.metrics.mel_distance_db)


def test_persistence_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16, persistence=1e-3))
    assert compiled.cost.total == len(compiled.to_bytes())
    assert np.isfinite(compiled.metrics.mel_distance_db)


def test_taper_removes_boundary_clicks_and_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    clean = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=24))
    raw = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=24, taper_alpha=0.0))
    # The default taper forces every atom to silent edges, so the boundary jump collapses far below the raw
    # codec — the artifact the spectral metrics miss, now measured.
    assert np.isfinite(clean.metrics.click_db)
    assert clean.metrics.click_db < raw.metrics.click_db - 6.0
    # Atom values never enter the size model, so the byte-exact keystone still holds with the taper on.
    assert clean.cost.total == len(clean.to_bytes())


def test_rate_levers_default_off_leave_the_reconstruction_unchanged(signal: NDArray[np.float64]) -> None:
    base = compile_signal(signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16))
    explicit = compile_signal(
        signal, TokenizerConfig(profile="strict", tempo=150, n_atoms=16, persistence=0.0, polarity_sticky=0)
    )
    assert explicit.to_bytes() == base.to_bytes()
    assert np.array_equal(explicit.estimate, base.estimate)
