from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.limits.compliance import Compliance
from trackmod.spec.grid import EMPTY

from audiotokenizer.audio.io import SAMPLE_RATE
from audiotokenizer.coding.cost import BITS_PER_BYTE, bitrate_kbps, cost_lower_bound, kilobytes
from audiotokenizer.module.format import Format
from audiotokenizer.module.samples import POLARITIES
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def test_kilobytes_uses_the_binary_kilobyte() -> None:
    assert kilobytes(1024) == 1.0
    assert kilobytes(0) == 0.0


def test_bitrate_counts_bits_over_seconds() -> None:
    assert bitrate_kbps(1000, 1.0) == BITS_PER_BYTE
    assert bitrate_kbps(2000, 2.0) == BITS_PER_BYTE


def test_bitrate_of_an_empty_signal_is_zero_rather_than_a_division() -> None:
    # A zero-length reference reaches the summary line before anything has been played; reporting 0 kbps
    # keeps that path free of a guard at every call site.
    assert bitrate_kbps(4096, 0.0) == 0.0
    assert bitrate_kbps(4096, -1.0) == 0.0


@pytest.mark.parametrize(("tempo", "n_atoms"), [(125, 16), (150, 24), (250, 8)])
def test_lower_bound_never_exceeds_a_real_module(signal: NDArray[np.float64], tempo: int, n_atoms: int) -> None:
    # The planner prunes candidates whose lower bound overruns the budget, so an over-estimate would
    # silently drop a feasible config. The bound must sit at or below what the module actually writes.
    config = TokenizerConfig(compliance=Compliance.EXTENDED, format=Format.IT, tempo=tempo, n_atoms=n_atoms)
    compiled = compile_signal(signal, config)
    patterns = compiled.module.song.patterns
    n_rows = sum(pattern.rows for pattern in patterns)
    n_cells = sum(int(np.count_nonzero(pattern.volume != EMPTY)) for pattern in patterns)
    bound = cost_lower_bound(
        n_rows,
        n_cells,
        n_stored_samples=POLARITIES * compiled.n_atoms_used,
        pcm_frames=compiled.timing.row_frames,
        bits_per_frame=config.pcm_bits,
    )
    assert bound <= compiled.size.total


def test_lower_bound_grows_with_every_input_it_takes() -> None:
    # Each argument buys bytes in the file, so none of them may leave the bound flat: a bound blind to one
    # of them would prune identically across a whole sweep axis.
    base = dict(n_stored_samples=32, pcm_frames=882, bits_per_frame=8)
    reference = cost_lower_bound(500, 4000, **base)
    assert cost_lower_bound(600, 4000, **base) > reference
    assert cost_lower_bound(500, 5000, **base) > reference
    assert cost_lower_bound(500, 4000, **{**base, "n_stored_samples": 48}) > reference
    assert cost_lower_bound(500, 4000, **{**base, "pcm_frames": 1000}) > reference
    assert cost_lower_bound(500, 4000, **{**base, "bits_per_frame": 16}) > reference


def test_lower_bound_of_a_silent_song_is_the_fixed_overhead() -> None:
    # No rows and no cells leaves only the header, the order list, the offset tables and one pattern
    # record — a positive floor every candidate pays before it stores anything.
    assert cost_lower_bound(0, 0, n_stored_samples=0, pcm_frames=0) > 0


def test_bitrate_of_a_compiled_module_is_its_own_size_over_its_own_duration(signal: NDArray[np.float64]) -> None:
    compiled = compile_signal(
        signal, TokenizerConfig(compliance=Compliance.EXTENDED, format=Format.IT, tempo=125, n_atoms=16)
    )
    duration = signal.size / SAMPLE_RATE
    assert bitrate_kbps(compiled.size.total, duration) == pytest.approx(
        len(compiled.to_bytes()) * BITS_PER_BYTE / duration / 1000
    )
