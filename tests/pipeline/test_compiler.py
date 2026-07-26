from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance

from audiotokenizer.module.format import Format
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig

CANONICAL = Compliance.CANONICAL
EXTENDED = Compliance.EXTENDED


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


@pytest.mark.parametrize("module_format", tuple(Format))
def test_compiled_size_matches_the_written_file(signal: NDArray[np.float64], module_format: Format) -> None:
    # The keystone: the size a compilation reports is the size it writes, for both formats.
    compiled = compile_signal(
        signal, TokenizerConfig(compliance=CANONICAL, format=module_format, tempo=150, n_atoms=16)
    )
    assert compiled.size.total == len(compiled.to_bytes())


def test_compilation_respects_channel_and_atom_caps(signal: NDArray[np.float64]) -> None:
    config = TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=24)
    compiled = compile_signal(signal, config)
    assert compiled.n_atoms_used <= config.n_atoms
    assert compiled.n_channels_used <= config.max_channels
    assert np.isfinite(compiled.metrics.mel_distance_db)
    assert compiled.metrics.mel_distance_db > 0.0


def test_extended_compliance_allows_more_than_the_canonical_channels() -> None:
    assert TokenizerConfig(compliance=EXTENDED, tempo=125, n_atoms=100).max_channels == 127
    assert TokenizerConfig(compliance=CANONICAL, tempo=125, n_atoms=100).max_channels == 64


def test_too_many_atoms_is_rejected() -> None:
    # 128 atoms would need 256 stored samples, over the 255 an IT module's sample table holds.
    with pytest.raises(ValueError):
        TokenizerConfig(compliance=EXTENDED, tempo=125, n_atoms=128)


def test_a_tempo_the_format_cannot_store_is_refused_at_the_config() -> None:
    # IT's header tempo is a single byte at both compliance levels, so a 16-bit tempo is refused rather
    # than written into a field too small to hold it. XM's is a word, so the same tempo is fine there.
    with pytest.raises(ValueError):
        TokenizerConfig(compliance=EXTENDED, format=Format.IT, tempo=441, n_atoms=16)
    assert TokenizerConfig(compliance=EXTENDED, format=Format.XM, tempo=441, n_atoms=16).tempo == 441


def test_a_compiled_module_reports_no_violations(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16))
    assert compiled.violations == ()
    assert compiled.writable


def test_channels_stay_within_the_bound_the_compliance_level_states(signal: NDArray[np.float64]) -> None:
    # The cap is read off the format's own limit table rather than repeated here, so asking for a pool
    # wider than the channels allowed still writes a module the format accepts.
    config = TokenizerConfig(compliance=CANONICAL, format=Format.XM, tempo=250, n_atoms=64)
    compiled = compile_signal(signal, config)
    assert compiled.n_channels_used <= config.limits.bound(Capability.CHANNELS).maximum
    assert compiled.writable


def test_polarity_sticky_saves_note_bytes_and_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    plain = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16))
    sticky = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16, polarity_sticky=4))
    # Suppressing tiny-code sign flips can only remove note bytes, never add them.
    assert sticky.size.patterns <= plain.size.patterns
    # The size model stays byte-exact, and the estimate reflects what the sticky grids actually play.
    assert sticky.size.total == len(sticky.to_bytes())
    assert np.isfinite(sticky.metrics.mel_distance_db)


def test_persistence_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16, persistence=1e-3))
    assert compiled.size.total == len(compiled.to_bytes())
    assert np.isfinite(compiled.metrics.mel_distance_db)


def test_taper_removes_boundary_clicks_and_keeps_the_keystone(signal: NDArray[np.float64]) -> None:
    clean = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=24))
    raw = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=24, taper_alpha=0.0))
    # The default taper forces every atom to silent edges, so the boundary jump collapses far below the raw
    # codec — the artifact the spectral metrics miss, now measured.
    assert np.isfinite(clean.metrics.click_db)
    assert clean.metrics.click_db < raw.metrics.click_db - 6.0
    # Atom values never enter the size model, so the byte-exact keystone still holds with the taper on.
    assert clean.size.total == len(clean.to_bytes())


def test_rate_levers_default_off_leave_the_reconstruction_unchanged(signal: NDArray[np.float64]) -> None:
    base = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16))
    explicit = compile_signal(
        signal, TokenizerConfig(compliance=CANONICAL, tempo=150, n_atoms=16, persistence=0.0, polarity_sticky=0)
    )
    assert explicit.to_bytes() == base.to_bytes()
    assert np.array_equal(explicit.estimate, base.estimate)


def test_the_reconstruction_is_format_agnostic(signal: NDArray[np.float64]) -> None:
    # Selection, quantisation, assignment and the reconstruction all read the same slot grids, so at the
    # same compliance/tempo/pool the estimate must be identical whichever format writes it.
    common = dict(compliance=CANONICAL, tempo=250, n_atoms=16)
    it_compiled = compile_signal(signal, TokenizerConfig(format=Format.IT, **common))
    xm_compiled = compile_signal(signal, TokenizerConfig(format=Format.XM, **common))
    assert np.array_equal(it_compiled.estimate, xm_compiled.estimate)
