from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.coding.cost import pattern_slices
from audiotokenizer.coding.xm_cost import xm_module_bytes, xm_pattern_bytes, xm_writable
from audiotokenizer.xm.instruments import XMInstrument
from audiotokenizer.xm.module import XMModule, write_xm_module
from audiotokenizer.xm.patterns import XMPattern, XMPlayback, pack_pattern
from audiotokenizer.xm.samples import XMSample
from audiotokenizer.xm.spec import MAX_PATTERNS, MAX_ROWS, PATTERN_HEADER_BYTES, SAMPLES_PER_INSTRUMENT


def _random_grids(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """A random sparse assignment: each active cell picks an atom-and-polarity slot and a 1..64 volume."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_atoms, size=(n_rows, n_channels))
    polarity = rng.integers(0, 2, size=(n_rows, n_channels))
    volume = rng.integers(0, 65, size=(n_rows, n_channels)).astype(np.int64)
    sample_no = np.where(volume > 0, 2 * picks + polarity, 0).astype(np.int64)
    return sample_no, volume


def _module_from_grids(
    sample_no: NDArray[np.int64], volume: NDArray[np.int64], *, n_atoms: int, rows_per_pattern: int = MAX_ROWS
) -> XMModule:
    """Build the module the compiler would from these grids, so its size can be checked against the model."""
    n_slots = 2 * n_atoms
    rng = np.random.default_rng(0)
    pcm = rng.standard_normal((n_slots, 8))
    pcm /= np.max(np.abs(pcm), axis=1, keepdims=True)
    samples = tuple(XMSample(name=f"s{i}", pcm=pcm[i], depth_bits=8, volume=40) for i in range(n_slots))
    instruments = tuple(
        XMInstrument(name=f"i{start // SAMPLES_PER_INSTRUMENT}", n_samples=min(SAMPLES_PER_INSTRUMENT, n_slots - start))
        for start in range(0, n_slots, SAMPLES_PER_INSTRUMENT)
    )
    slices = pattern_slices(sample_no.shape[0], rows_per_pattern=rows_per_pattern)
    patterns = tuple(
        XMPattern(rows=stop - start, sample_no=sample_no[start:stop], volume=volume[start:stop])
        for start, stop in slices
    )
    return XMModule(
        name="cost",
        samples=samples,
        instruments=instruments,
        patterns=patterns,
        orders=tuple(range(len(patterns))),
        playback=XMPlayback(speed=1, tempo=1000),
        channels=int(sample_no.shape[1]),
    )


# Half-random 256-row patterns pack to roughly 2 bytes per cell, so at 64 channels each pattern stays under
# XM's u16 length; denser or wider grids hit the feasibility wall, exercised separately below.
@pytest.mark.parametrize(
    ("seed", "n_rows", "n_channels", "n_atoms"),
    [(1, 205, 40, 30), (2, 517, 64, 50), (3, 1000, 64, 96), (4, 33, 8, 4)],
)
def test_pattern_bytes_equals_packed_stream(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> None:
    sample_no, volume = _random_grids(seed, n_rows, n_channels, n_atoms)
    for start, stop in pattern_slices(n_rows):
        pattern = XMPattern(rows=stop - start, sample_no=sample_no[start:stop], volume=volume[start:stop])
        stream = len(pack_pattern(pattern)) - PATTERN_HEADER_BYTES
        assert xm_pattern_bytes(sample_no[start:stop], volume[start:stop]) == stream


@pytest.mark.parametrize(
    ("seed", "n_rows", "n_channels", "n_atoms"),
    [(1, 205, 40, 30), (2, 517, 64, 50), (3, 1000, 64, 96)],
)
def test_module_bytes_equals_written_file(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> None:
    sample_no, volume = _random_grids(seed, n_rows, n_channels, n_atoms)
    module = _module_from_grids(sample_no, volume, n_atoms=n_atoms)
    cost = xm_module_bytes(sample_no, volume, n_stored_samples=2 * n_atoms, pcm_frames=8, bits_per_frame=8)
    assert cost.total == len(write_xm_module(module))


def test_empty_channel_costs_one_byte_and_present_cell_three_plus_instrument_change() -> None:
    # channel 0 keeps slot 0 (same instrument) every row; channel 1 is silent -> one byte each row.
    rows = 4
    sample_no = np.zeros((rows, 2), dtype=np.int64)
    volume = np.zeros((rows, 2), dtype=np.int64)
    volume[:, 0] = 32
    # ch0: first present cell 4 bytes (states instrument) + 3 kept cells * 3 = 13; ch1: 4 empty * 1 = 4.
    assert xm_pattern_bytes(sample_no, volume) == 13 + 4
    assert (
        xm_pattern_bytes(sample_no, volume)
        == len(pack_pattern(XMPattern(rows, sample_no, volume))) - PATTERN_HEADER_BYTES
    )


def test_polarity_flip_inside_one_instrument_costs_no_instrument_byte() -> None:
    # slots 0 and 1 are the two polarities of atom 0 -> same instrument; flipping between them each row
    # re-states the note (always) but never the instrument, so every present cell stays three bytes.
    rows = 6
    sample_no = np.array([[r % 2] for r in range(rows)], dtype=np.int64)
    volume = np.full((rows, 1), 20, dtype=np.int64)
    assert xm_pattern_bytes(sample_no, volume) == rows * 3 + 1  # +1: the first cell's instrument byte


def test_feasibility_wall_rejects_too_many_patterns() -> None:
    # 300 one-row patterns exceed the 256-entry order table, so the song is unwritable even though each
    # pattern is tiny (unlike IT, whose pattern count is a u16).
    sample_no, volume = _random_grids(seed=7, n_rows=300, n_channels=4, n_atoms=8)
    cost = xm_module_bytes(sample_no, volume, n_stored_samples=16, pcm_frames=8, bits_per_frame=8, rows_per_pattern=1)
    assert cost.n_rows == 300
    assert not xm_writable(cost, rows_per_pattern=1)
    assert len(pattern_slices(300, rows_per_pattern=1)) > MAX_PATTERNS


def test_feasibility_wall_rejects_a_pattern_over_the_u16_limit() -> None:
    rng = np.random.default_rng(9)
    rows, channels = MAX_ROWS, 255
    volume = rng.integers(1, 65, size=(rows, channels)).astype(np.int64)  # every cell present
    sample_no = rng.integers(0, 128, size=(rows, channels)).astype(np.int64)
    cost = xm_module_bytes(sample_no, volume, n_stored_samples=256, pcm_frames=8, bits_per_frame=8)
    assert not cost.writable
    assert not xm_writable(cost)
    with pytest.raises(ValueError):
        pack_pattern(XMPattern(rows=rows, sample_no=sample_no, volume=volume))


def test_writable_holds_for_a_realistic_song() -> None:
    sample_no, volume = _random_grids(seed=2, n_rows=517, n_channels=64, n_atoms=50)
    cost = xm_module_bytes(sample_no, volume, n_stored_samples=100, pcm_frames=8, bits_per_frame=8)
    assert xm_writable(cost)
