"""What one tracker format decides about turning a coded signal into a module.

Everything up to the channel grids is format-agnostic — selection, quantisation, assignment and the
reconstruction all read the same ``(sample_no, volume)`` slots — and ``trackmod`` carries the whole song
model. What is left is this handful of choices, and they are exactly where the two formats disagree:

* **How a dictionary is routed.** Impulse Tracker gives an instrument 120 keys and starts at the lowest;
  FastTracker 2 gives it 16 and starts partway up its 96-key range.
* **Where an atom's gain lives.** Impulse Tracker multiplies the volume column by the sample's own
  global-volume lattice, so the gain rides in the sample. FastTracker 2's volume column *overrides* the
  sample volume byte rather than scaling it, so the gain has to be baked into the waveform.
* **Whether a cell has to name its instrument again**, which follows from where the gain lives: a note
  played without one keeps the channel's current volume settings instead of taking the new sample's.
* **How a song is cut into patterns.** Impulse Tracker keeps per-channel memory that resets at a pattern
  boundary, so it fills each pattern to the ceiling. FastTracker 2 keeps none and instead has a small
  order table, so it spreads the song across as many patterns as that table holds.
* **Which module class the song binds to**, and which limit table it answers to.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.song import Song
from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.module.protocol import TrackerModule

from audiotokenizer.module.format import Format
from audiotokenizer.module.routing import Routing


class Binding(Protocol):
    """One tracker format, bound to this codec."""

    @property
    def format(self) -> Format:
        """Which format this binding writes."""

    @property
    def routing(self) -> Routing:
        """How this format's instruments and keys reach the stored samples."""

    @property
    def restates_instrument(self) -> bool:
        """Whether every played cell names its instrument again rather than only on a change.

        Naming it costs nothing in either format — one keeps a reuse flag, the other a mask bit — and
        what it buys is the instrument's own volume being re-applied when the note starts. A format that
        keeps an atom's gain in the sample needs that on every cell, or a channel carries the previous
        atom's gain into the new one; a format that bakes the gain into the waveform does not.
        """

    def limits(self, compliance: Compliance) -> Limits:
        """The bounds this format holds a song to, at one compliance level."""

    def sample(self, name: str, pcm: NDArray[np.float64], *, rate: int, gain: int, depth: BitDepth) -> Sample:
        """One stored atom, with its gain carried wherever this format has room for it."""

    def pattern_count(self, rows: int) -> int:
        """How many patterns this format cuts a song of ``rows`` rows into."""

    def module(self, song: Song, *, compliance: Compliance) -> TrackerModule:
        """Bind a song to this format at one compliance level."""


def gained(pcm: NDArray[np.float64], gain: int, *, levels: int) -> NDArray[np.float64]:
    """A waveform scaled by a gain drawn from a ``0..levels`` lattice."""
    return np.asarray(pcm * (gain / levels), dtype=np.float64)
