from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.pipeline.compiler import _xm_rows_per_pattern, compile_signal
from audiotokenizer.pipeline.config import MAX_ATOMS, XM_MAX_ATOMS, TokenizerConfig
from audiotokenizer.xm.module import XMModule
from audiotokenizer.xm.reader import read_xm_samples
from audiotokenizer.xm.spec import MAX_PATTERNS, MAX_ROWS, STRICT_MAX_CHANNELS


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


# -- config format-awareness ------------------------------------------------


def test_channel_cap_is_format_and_profile_aware() -> None:
    assert TokenizerConfig(profile="strict", format="xm", tempo=125, n_atoms=10).max_channels == 32
    assert TokenizerConfig(profile="hacked", format="xm", tempo=125, n_atoms=10).max_channels == 255
    assert TokenizerConfig(profile="strict", format="it", tempo=125, n_atoms=10).max_channels == 64
    assert TokenizerConfig(profile="hacked", format="it", tempo=125, n_atoms=10).max_channels == 127


def test_tempo_ceiling_is_the_16bit_hack_when_hacked() -> None:
    assert TokenizerConfig(profile="hacked", format="xm", tempo=125, n_atoms=10).max_tempo == 65535
    assert TokenizerConfig(profile="strict", format="xm", tempo=125, n_atoms=10).max_tempo == 255


def test_atom_pool_cap_is_wider_for_xm() -> None:
    assert TokenizerConfig(profile="hacked", format="it", tempo=125, n_atoms=10).max_atoms == MAX_ATOMS == 127
    assert TokenizerConfig(profile="hacked", format="xm", tempo=125, n_atoms=10).max_atoms == XM_MAX_ATOMS == 1024
    # A pool IT cannot route (200 atoms -> 400 samples > 255) is fine for XM (400 <= 2048).
    with pytest.raises(ValueError):
        TokenizerConfig(profile="strict", format="it", tempo=125, n_atoms=200)
    assert TokenizerConfig(profile="strict", format="xm", tempo=125, n_atoms=200).n_atoms == 200


# -- the XM compile path ----------------------------------------------------


def test_xm_compile_builds_an_xm_module_byte_exactly(signal: NDArray[np.float64]) -> None:
    config = TokenizerConfig.load(profile="hacked", format="xm", tempo=441, n_atoms=16)
    compiled = compile_signal(signal, config)
    assert isinstance(compiled.module, XMModule)
    assert compiled.timing.tempo == 441  # a tempo IT cannot store
    assert compiled.timing.row_frames == 250
    assert compiled.cost.total == len(compiled.to_bytes())


def test_xm_reconstruction_matches_it_bit_for_bit(signal: NDArray[np.float64]) -> None:
    # Selection, quantisation, assignment and the reconstruction are format-agnostic, so at the same
    # profile/tempo/pool the estimate a module reproduces must be identical whichever format writes it.
    common = dict(profile="strict", tempo=250, n_atoms=16)
    it_compiled = compile_signal(signal, TokenizerConfig(format="it", **common))
    xm_compiled = compile_signal(signal, TokenizerConfig(format="xm", **common))
    assert np.array_equal(it_compiled.estimate, xm_compiled.estimate)


def test_xm_module_round_trips_through_xmodits(signal: NDArray[np.float64]) -> None:
    config = TokenizerConfig.load(profile="hacked", format="xm", tempo=441, n_atoms=12)
    compiled = compile_signal(signal, config)
    assert isinstance(compiled.module, XMModule)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.xm"
        compiled.save(path)
        ripped = read_xm_samples(path)
    assert len(ripped) == len(compiled.module.samples) == 2 * compiled.n_atoms_used


def test_strict_xm_never_exceeds_32_channels(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(profile="strict", format="xm", tempo=250, n_atoms=40))
    assert compiled.n_channels_used <= STRICT_MAX_CHANNELS


# -- the feasibility wall ---------------------------------------------------


def test_rows_per_pattern_fits_the_order_table() -> None:
    assert _xm_rows_per_pattern(1) == 1
    assert _xm_rows_per_pattern(MAX_PATTERNS) == 1  # 256 rows -> 256 one-row patterns, exactly the wall
    assert _xm_rows_per_pattern(MAX_PATTERNS * MAX_ROWS) == MAX_ROWS  # the largest writable song


def test_rows_per_pattern_raises_past_the_wall() -> None:
    # A song needing patterns taller than 256 rows to fit 256 of them cannot be written as XM.
    with pytest.raises(ValueError):
        _xm_rows_per_pattern(MAX_PATTERNS * MAX_ROWS + 1)
