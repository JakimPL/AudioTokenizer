"""Pin the compiled output against the modules the pipeline produced before the ``trackmod`` extraction.

The migration was allowed to change how the bytes are produced, not what they play, and these three tests
say exactly how much of "unchanged" survived. FastTracker 2 came through byte-for-byte. Impulse Tracker
did not, in four ways that a tracker cannot hear: ``trackmod`` fills the DOS-filename field each record
reserves (the old writer left it zero), it leaves the instrument's default panning unset instead of
writing centre, it maps a keymap's unused keys to themselves rather than to C-5, and its packer reuses a
channel's remembered volume where the old one always restated it — which makes the file one byte *smaller*.
Everything that sounds is identical, which is what the parse-back test and the render A/B assert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.it.module import ITModule
from trackmod.limits.compliance import Compliance

from audiotokenizer.audio.io import SAMPLE_RATE
from audiotokenizer.module.format import Format
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.render.openmpt import openmpt123_available, render_bytes

GOLDEN = Path(__file__).parent.parent / "data"

CONFIGS = {
    Format.IT: TokenizerConfig(compliance=Compliance.EXTENDED, format=Format.IT, tempo=125, n_atoms=16),
    Format.XM: TokenizerConfig(compliance=Compliance.EXTENDED, format=Format.XM, tempo=441, n_atoms=16),
}


def golden_signal() -> NDArray[np.float64]:
    """The fixed signal both reference modules were compiled from."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def reference(module_format: Format) -> bytes:
    return (GOLDEN / f"golden.{module_format}").read_bytes()


def compiled(module_format: Format) -> bytes:
    return compile_signal(golden_signal(), CONFIGS[module_format]).to_bytes()


def test_xm_bytes_are_unchanged() -> None:
    """The FastTracker 2 path reproduces its reference exactly, down to the last byte."""
    assert compiled(Format.XM) == reference(Format.XM)


def test_it_writes_the_same_song_no_larger() -> None:
    """The Impulse Tracker path writes the same content, and never spends more bytes doing it.

    Parsing both files back is the test rather than comparing them: it reads past the informational
    fields the two writers fill differently and compares what a tracker actually plays.
    """
    before = ITModule.parse(reference(Format.IT)).song
    after = ITModule.parse(compiled(Format.IT)).song

    assert after.patterns == before.patterns
    assert after.samples == before.samples
    assert after.order == before.order
    assert after.playback == before.playback
    assert after.channels == before.channels
    assert [instrument.keymap for instrument in after.instruments] == [
        instrument.keymap for instrument in before.instruments
    ]
    assert len(compiled(Format.IT)) <= len(reference(Format.IT))


@pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 is not installed")
@pytest.mark.parametrize("module_format", sorted(CONFIGS))
def test_renders_are_identical(module_format: Format) -> None:
    """The ground truth: a real tracker plays the migrated module and the reference identically."""
    expected, _ = render_bytes(reference(module_format), suffix=f".{module_format}", sample_rate=SAMPLE_RATE)
    actual, _ = render_bytes(compiled(module_format), suffix=f".{module_format}", sample_rate=SAMPLE_RATE)
    assert np.array_equal(actual, expected)
