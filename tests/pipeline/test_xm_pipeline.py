from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.limits.compliance import Compliance
from trackmod.xm.module import XMModule
from trackmod.xm.spec.ranges import (
    CANONICAL_MAX_CHANNELS,
    CANONICAL_MAX_TEMPO,
    EXTENDED_MAX_CHANNELS,
    MAX_PATTERNS,
    MAX_ROWS,
    MAX_TEMPO,
)

from audiotokenizer.module.format import Format
from audiotokenizer.module.samples import POLARITIES
from audiotokenizer.module.xm import XM_BINDING
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig, max_atoms_for
from audiotokenizer.render.xmodits import read_samples

CANONICAL = Compliance.CANONICAL
EXTENDED = Compliance.EXTENDED


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


# -- config format-awareness ------------------------------------------------


def test_channel_cap_is_format_and_compliance_aware() -> None:
    assert TokenizerConfig(compliance=CANONICAL, format=Format.XM, tempo=125, n_atoms=10).max_channels == 32
    assert TokenizerConfig(compliance=EXTENDED, format=Format.XM, tempo=125, n_atoms=10).max_channels == 192
    assert TokenizerConfig(compliance=CANONICAL, format=Format.IT, tempo=125, n_atoms=10).max_channels == 64
    assert TokenizerConfig(compliance=EXTENDED, format=Format.IT, tempo=125, n_atoms=10).max_channels == 127


def test_tempo_ceiling_is_the_16bit_word_only_where_the_field_is_one() -> None:
    # This format's header tempo is a word, so extending it reaches rows the canonical byte cannot ask for.
    assert TokenizerConfig(compliance=EXTENDED, format=Format.XM, tempo=125, n_atoms=10).max_tempo == MAX_TEMPO
    assert TokenizerConfig(compliance=CANONICAL, format=Format.XM, tempo=125, n_atoms=10).max_tempo == (
        CANONICAL_MAX_TEMPO
    )
    # Impulse Tracker's is a byte at both levels, which is the bug the limit table exists to prevent.
    assert TokenizerConfig(compliance=EXTENDED, format=Format.IT, tempo=125, n_atoms=10).max_tempo == 255


def test_atom_pool_cap_is_wider_for_xm() -> None:
    assert max_atoms_for(Format.IT, EXTENDED) == 127
    assert max_atoms_for(Format.XM, EXTENDED) == 1024
    # A pool IT cannot route (200 atoms -> 400 stored samples > 255) is fine for XM (400 <= 2048).
    with pytest.raises(ValueError):
        TokenizerConfig(compliance=CANONICAL, format=Format.IT, tempo=125, n_atoms=200)
    assert TokenizerConfig(compliance=CANONICAL, format=Format.XM, tempo=125, n_atoms=200).n_atoms == 200


# -- the XM compile path ----------------------------------------------------


def test_xm_compile_builds_an_xm_module_byte_exactly(signal: NDArray[np.float64]) -> None:
    config = TokenizerConfig(compliance=EXTENDED, format=Format.XM, tempo=441, n_atoms=16)
    compiled = compile_signal(signal, config)
    assert isinstance(compiled.module, XMModule)
    assert compiled.timing.tempo == 441  # a tempo IT cannot store
    assert compiled.timing.row_frames == 250
    assert compiled.size.total == len(compiled.to_bytes())


def test_xm_module_round_trips_through_xmodits(signal: NDArray[np.float64], tmp_path: Path) -> None:
    config = TokenizerConfig(compliance=EXTENDED, format=Format.XM, tempo=441, n_atoms=12)
    compiled = compile_signal(signal, config)
    path = compiled.save(tmp_path / "out.xm")
    ripped = read_samples(path)
    assert len(ripped) == len(compiled.module.song.samples) == POLARITIES * compiled.n_atoms_used


def test_canonical_xm_never_exceeds_32_channels(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, format=Format.XM, tempo=250, n_atoms=40))
    assert compiled.n_channels_used <= CANONICAL_MAX_CHANNELS
    assert compiled.writable


def test_extended_xm_writes_more_channels_than_the_tracker_honours(signal: NDArray[np.float64]) -> None:
    # The extended bound is what the record layout holds, and 192 is where libopenmpt stops loading. The
    # same pool that the canonical level clamps to 32 channels spreads over more of them here.
    common = dict(format=Format.XM, tempo=250, n_atoms=48)
    canonical = compile_signal(signal, TokenizerConfig(compliance=CANONICAL, **common))
    extended = compile_signal(signal, TokenizerConfig(compliance=EXTENDED, **common))
    assert canonical.module.song.channels == CANONICAL_MAX_CHANNELS
    assert CANONICAL_MAX_CHANNELS < extended.module.song.channels <= EXTENDED_MAX_CHANNELS
    assert canonical.writable and extended.writable


# -- the pattern split ------------------------------------------------------


def test_pattern_count_never_needs_more_orders_than_the_table_holds() -> None:
    assert XM_BINDING.pattern_count(1) == 1
    assert XM_BINDING.pattern_count(MAX_PATTERNS) == MAX_PATTERNS  # 256 rows -> 256 one-row patterns
    assert XM_BINDING.pattern_count(MAX_PATTERNS * MAX_ROWS) == MAX_PATTERNS  # the largest canonical song


@pytest.mark.parametrize("rows", [1, 7, 255, 256, 257, 530, 4097, MAX_PATTERNS * MAX_ROWS])
def test_pattern_count_covers_every_row_within_both_ceilings(rows: int) -> None:
    # The split has to satisfy two walls at once: no more patterns than the order table indexes, and no
    # pattern taller than the row field stores. Every row of the song must still land in one of them.
    count = XM_BINDING.pattern_count(rows)
    height = -(-rows // count)
    assert 1 <= count <= MAX_PATTERNS
    assert 1 <= height <= MAX_ROWS
    assert count * height >= rows


def test_a_song_past_the_wall_leaves_a_pattern_over_the_row_ceiling() -> None:
    # 256 patterns of 256 rows is the largest song the two ceilings admit together. One row past it, the
    # shortest split the order table allows is already taller than the row field, which is what the
    # module's own bounds then report rather than the split silently dropping rows.
    rows = MAX_PATTERNS * MAX_ROWS + 1
    count = XM_BINDING.pattern_count(rows)
    assert count == MAX_PATTERNS
    assert -(-rows // count) > MAX_ROWS
