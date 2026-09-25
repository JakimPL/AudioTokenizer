"""Play the extended bounds through a real tracker, run only when ``openmpt123`` is installed.

Every bound above what a format's own tracker honoured is a claim about somebody else's loader, and a
claim only a renderer can settle. Checking that a probe *loads* is not enough — libopenmpt will happily
read a module and then clamp what it cannot play — so each probe here asserts the rendered audio lasts as
long as the row clock says it should. A 127-channel module that quietly dropped its extra channels, or a
441 BPM module whose tempo word was truncated to a byte, would still load and would fail this.

``openmpt123`` appends a fixed tail past the last row, so a probe is measured as the row clock plus an
allowance rather than exactly; where the clock itself is the claim, two probes differing only in row count
are subtracted so the tail cancels.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import pitched_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.core.voices.voices import InstrumentVoices
from trackmod.limits.compliance import Compliance
from trackmod.module.protocol import TrackerModule
from trackmod.spec.levels import MAX_VOLUME
from trackmod.trackers.it.module import ITModule
from trackmod.trackers.it.spec.ranges import EXTENDED_MAX_CHANNELS as IT_EXTENDED_MAX_CHANNELS
from trackmod.trackers.it.timing import TIMINGS as IT_TIMINGS
from trackmod.trackers.xm.module import XMModule
from trackmod.trackers.xm.spec.ranges import EXTENDED_MAX_CHANNELS as XM_EXTENDED_MAX_CHANNELS
from trackmod.trackers.xm.timing import TIMINGS as XM_TIMINGS

from audiotokenizer.render.openmpt import openmpt123_available, render_module

pytestmark = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 is not installed")

RATE: Final = 44100
PROBE_ROWS: Final = 64
PROBE_NOTE: Final = Note(60)
#: openmpt123 keeps rendering past the last row; every probe overshoots by the same fixed margin, so an
#: allowance this wide separates "the clock is right" from "a field was truncated".
TAIL_ALLOWANCE: Final = RATE // 5


def probe_song(*, channels: int, rows: int = PROBE_ROWS, speed: int, tempo: int) -> Song:
    """A song that sounds on every channel it declares, so a dropped channel changes what is rendered."""
    frames = 64
    builder = PatternBuilder(rows=rows, channels=channels)
    for channel in range(channels):
        for row in range(0, rows, 4):
            builder.place(row, channel, Cell(note=PROBE_NOTE, instrument=0, volume=MAX_VOLUME))
    return Song(
        name="probe",
        channels=channels,
        patterns=(builder.build(),),
        order=OrderList(entries=(0,)),
        voices=InstrumentVoices(
            instruments=(Instrument(name="probe", keymap=pitched_keymap(sample=0)),),
            samples=(Sample(name="probe", pcm=np.sin(2 * np.pi * np.arange(frames) / frames), rate=RATE),),
        ),
        playback=Playback(speed=speed, tempo=tempo),
    )


def rendered_frames(module: TrackerModule) -> int:
    rendered, rate = render_module(module, sample_rate=RATE)
    assert rate == RATE
    assert float(np.sqrt(np.mean(rendered**2))) > 1e-4  # a clamped-to-silence load would pass a length check
    return int(rendered.size)


def assert_plays_the_clock(module: TrackerModule, *, rows: int, row_frames: int) -> None:
    expected = rows * row_frames
    assert expected <= rendered_frames(module) <= expected + TAIL_ALLOWANCE


def test_it_plays_127_channels() -> None:
    # 64 is what Impulse Tracker itself honoured; the pattern stream's channel marker reaches 127.
    speed, tempo = 6, 125
    song = probe_song(channels=IT_EXTENDED_MAX_CHANNELS, speed=speed, tempo=tempo)
    module = ITModule.from_song(song, compliance=Compliance.EXTENDED)
    assert module.violations() == ()
    assert_plays_the_clock(module, rows=PROBE_ROWS, row_frames=IT_TIMINGS.row_frames(speed, tempo, frame_rate=RATE))


def test_xm_plays_127_channels() -> None:
    # 32 is what FastTracker 2 honoured; 127 is what the players descended from it read, which is the
    # extended bound the limit table states.
    speed, tempo = 6, 125
    song = probe_song(channels=XM_EXTENDED_MAX_CHANNELS, speed=speed, tempo=tempo)
    module = XMModule.from_song(song, compliance=Compliance.EXTENDED)
    assert module.violations() == ()
    assert_plays_the_clock(module, rows=PROBE_ROWS, row_frames=XM_TIMINGS.row_frames(speed, tempo, frame_rate=RATE))


def test_xm_plays_a_tempo_past_the_byte_a_canonical_module_stores() -> None:
    # 441 BPM is the whole point of this format's 16-bit tempo word: a row of 250 frames, five times
    # shorter than the 255 ceiling reaches. A truncated tempo would render a far longer file.
    speed, tempo = 1, 441
    row_frames = XM_TIMINGS.row_frames(speed, tempo, frame_rate=RATE)
    assert row_frames == 250
    song = probe_song(channels=8, speed=speed, tempo=tempo)
    module = XMModule.from_song(song, compliance=Compliance.EXTENDED)
    assert module.violations() == ()
    assert_plays_the_clock(module, rows=PROBE_ROWS, row_frames=row_frames)


def test_a_high_xm_tempo_scales_the_render_exactly_with_the_rows() -> None:
    # Subtracting two probes that differ only in row count cancels the renderer's tail, so what is left is
    # the row length itself — the strongest statement that the tempo word arrived intact.
    speed, tempo = 1, 441
    row_frames = XM_TIMINGS.row_frames(speed, tempo, frame_rate=RATE)
    short = XMModule.from_song(
        probe_song(channels=8, rows=32, speed=speed, tempo=tempo), compliance=Compliance.EXTENDED
    )
    long = XMModule.from_song(probe_song(channels=8, rows=96, speed=speed, tempo=tempo), compliance=Compliance.EXTENDED)
    assert rendered_frames(long) - rendered_frames(short) == (96 - 32) * row_frames
