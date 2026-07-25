from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from trackmod.core.effects.effect import Effect
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.point import EnvelopePoint
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, pitched_keymap, routed_keymap
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.loop import Loop, LoopMode
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.spec.pitch import NOTE_COUNT

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class GridShape:
    """The dimensions and density of a randomly generated pattern grid."""

    rows: int
    channels: int
    instruments: int
    seed: int


def make_sample(name: str, *, frames: int = 32, depth: BitDepth = BitDepth.SIXTEEN, seed: int = 0) -> Sample:
    """A short deterministic waveform, used wherever a test needs sample data rather than silence."""
    rng = np.random.default_rng(seed)
    return Sample(name=name, pcm=rng.uniform(-1.0, 1.0, frames), rate=SAMPLE_RATE, depth=depth)


def random_pattern(shape: GridShape) -> Pattern:
    """A grid whose cells cover every combination of present and absent columns.

    Randomising column presence independently is what exercises the reuse bits: a packer that only ever
    sees fully populated cells never has to decide what an absent column costs.
    """
    rng = np.random.default_rng(shape.seed)
    builder = PatternBuilder(rows=shape.rows, channels=shape.channels)
    for row in range(shape.rows):
        for channel in range(shape.channels):
            draw = rng.random()
            if draw < 0.25:
                continue

            note = Note(int(rng.integers(0, NOTE_COUNT))) if draw < 0.85 else NoteCommand.OFF
            builder.place(
                row,
                channel,
                Cell(
                    note=note,
                    instrument=int(rng.integers(0, shape.instruments)) if draw < 0.8 else None,
                    volume=int(rng.integers(0, 65)) if draw < 0.7 else None,
                    effect=Effect(command=1, parameter=int(rng.integers(0, 256))) if draw > 0.9 else None,
                ),
            )

    return builder.build()


def rescaled(song: Song, channels: int) -> Song:
    """The same song widened to a new channel count, which is how the channel limits are exercised."""
    return Song(
        name=song.name,
        channels=channels,
        patterns=tuple(pattern.widened(channels) for pattern in song.patterns),
        order=song.order,
        instruments=song.instruments,
        samples=song.samples,
        playback=song.playback,
    )


@pytest.fixture
def waveform() -> NDArray[np.float64]:
    """A deterministic waveform for sample round-trip checks."""
    return np.sin(2 * np.pi * 4 * np.linspace(0.0, 1.0, 64, endpoint=False))


@pytest.fixture
def fade_envelope() -> Envelope:
    """A two-point volume envelope with a sustain point, the shape a released voice needs."""
    return Envelope(
        points=(EnvelopePoint(tick=0, value=64), EnvelopePoint(tick=32, value=0)),
        sustain=EnvelopeSpan(begin=0, end=0),
    )


@pytest.fixture
def song(fade_envelope: Envelope) -> Song:
    """A small song exercising every part of the model both formats have to carry.

    Both patterns clear the canonical row floor, so the song is writable at either compliance level.
    """
    samples = (
        make_sample("lead", seed=1),
        make_sample("bass", depth=BitDepth.EIGHT, seed=2),
        Sample(
            name="looped",
            pcm=np.linspace(-1.0, 1.0, 48),
            rate=22050,
            loop=Loop(begin=8, end=40, mode=LoopMode.FORWARD),
        ),
    )
    instruments = (
        Instrument(name="piano", keymap=pitched_keymap(sample=0), volume_envelope=fade_envelope, fadeout=256),
        Instrument(
            name="router",
            keymap=routed_keymap(
                {
                    Note(60): KeyAssignment(sample=1, note=Note(60)),
                    Note(61): KeyAssignment(sample=2, note=Note(60)),
                }
            ),
        ),
    )
    patterns = (
        random_pattern(GridShape(rows=32, channels=4, instruments=len(instruments), seed=7)),
        random_pattern(GridShape(rows=48, channels=4, instruments=len(instruments), seed=8)),
    )
    return Song(
        name="trackmod",
        channels=4,
        patterns=patterns,
        order=OrderList(entries=(0, 1, 0)),
        instruments=instruments,
        samples=samples,
        playback=Playback(speed=6, tempo=125),
    )
