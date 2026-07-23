from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.coding.cost import module_bytes, pattern_bytes, pattern_slices
from audiotokenizer.it.instruments import ITInstrument, fixed_c5_note_map
from audiotokenizer.it.module import ITModule, write_it_module
from audiotokenizer.it.patterns import ITPattern, ITPlayback, pack_pattern
from audiotokenizer.it.samples import ITSample
from audiotokenizer.it.spec import KEYBOARD_NOTES, PATTERN_HEADER_BYTES


def _random_grids(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """A random sparse assignment: each active cell picks an atom-and-polarity slot and a 1..64 volume."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_atoms, size=(n_rows, n_channels))
    polarity = rng.integers(0, 2, size=(n_rows, n_channels))
    volume = rng.integers(0, 65, size=(n_rows, n_channels)).astype(np.int64)
    sample_no = np.where(volume > 0, 2 * picks + polarity, 0).astype(np.int64)
    return sample_no, volume


def _module_from_grids(sample_no: NDArray[np.int64], volume: NDArray[np.int64], *, n_atoms: int) -> ITModule:
    """Build the module the compiler would from these grids, so its size can be checked against the model."""
    n_slots = 2 * n_atoms
    rng = np.random.default_rng(0)
    pcm = rng.standard_normal((n_slots, 8))
    pcm /= np.max(np.abs(pcm), axis=1, keepdims=True)
    samples = tuple(ITSample(name=f"s{i}", pcm=pcm[i], depth_bits=8, c5speed=44100) for i in range(n_slots))
    instruments = tuple(
        ITInstrument(
            name=f"i{start // KEYBOARD_NOTES}",
            note_map=fixed_c5_note_map(
                {slot - start: slot + 1 for slot in range(start, min(start + KEYBOARD_NOTES, n_slots))}
            ),
        )
        for start in range(0, n_slots, KEYBOARD_NOTES)
    )
    patterns = tuple(
        ITPattern(rows=stop - start, sample_no=sample_no[start:stop], volume=volume[start:stop])
        for start, stop in pattern_slices(sample_no.shape[0])
    )
    return ITModule(
        name="cost",
        samples=samples,
        instruments=instruments,
        patterns=patterns,
        orders=tuple(range(len(patterns))),
        playback=ITPlayback(speed=1, tempo=125, global_volume=128, mix_volume=48),
    )


# A 200-row pattern of fully-random cells packs near its 5-byte-per-cell worst case, so channel counts
# stay at 64, where even that fits IT's u16 pattern length. Real assignments persist atoms and pack far
# smaller; the compiler keystone (tests/pipeline) exercises 96 channels on structured audio.
@pytest.mark.parametrize(
    ("seed", "n_rows", "n_channels", "n_atoms"),
    [(1, 205, 40, 30), (2, 517, 64, 50), (3, 1000, 64, 96), (4, 33, 8, 4)],
)
def test_pattern_bytes_equals_packed_stream(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> None:
    sample_no, volume = _random_grids(seed, n_rows, n_channels, n_atoms)
    for start, stop in pattern_slices(n_rows):
        pattern = ITPattern(rows=stop - start, sample_no=sample_no[start:stop], volume=volume[start:stop])
        stream = len(pack_pattern(pattern)) - PATTERN_HEADER_BYTES
        assert pattern_bytes(sample_no[start:stop], volume[start:stop]) == stream


@pytest.mark.parametrize(
    ("seed", "n_rows", "n_channels", "n_atoms"),
    [(1, 205, 40, 30), (2, 517, 64, 50), (3, 1000, 64, 96)],
)
def test_module_bytes_equals_written_file(seed: int, n_rows: int, n_channels: int, n_atoms: int) -> None:
    sample_no, volume = _random_grids(seed, n_rows, n_channels, n_atoms)
    module = _module_from_grids(sample_no, volume, n_atoms=n_atoms)
    cost = module_bytes(sample_no, volume, n_stored_samples=2 * n_atoms, pcm_frames=8, bits_per_frame=8)
    assert cost.total == len(write_it_module(module))


def test_pack_pattern_rejects_a_pattern_over_the_u16_limit() -> None:
    sample_no, volume = _random_grids(seed=9, n_rows=200, n_channels=96, n_atoms=96)
    with pytest.raises(ValueError):
        pack_pattern(ITPattern(rows=200, sample_no=sample_no, volume=volume))
