"""Compiling the codec's output into an Impulse Tracker module.

This format multiplies two 0..64 lattices on the way to a channel's amplitude — the sample's own
global-volume and the row's volume column — so an atom's static gain rides in the sample and the column is
left to carry the coefficient alone. That is also why every played cell names its instrument: the sample's
volume is taken at a note that states one, and a note without one would sound the new atom's waveform at
the previous atom's gain.

Its instruments route 120 keys each, one per stored sample, and its packed patterns keep per-channel
memory that resets at every pattern boundary, so a song is cut into as few patterns as the 200-row ceiling
allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.song import Song
from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.trackers.it.limits import it_limits
from trackmod.trackers.it.module import ITModule
from trackmod.trackers.it.spec.ranges import MAX_ROWS

from audiotokenizer.module.format import Format
from audiotokenizer.module.routing import Routing

FIRST_KEY: Final = 0  # an instrument's first sample answers to the lowest key this format numbers
ROUTING: Final = Routing(samples_per_instrument=NOTE_COUNT, first_key=FIRST_KEY)


@dataclass(frozen=True)
class ITBinding:
    """Impulse Tracker as this codec writes it."""

    format: Format = Format.IT
    routing: Routing = field(default=ROUTING)
    restates_instrument: bool = True

    def limits(self, compliance: Compliance) -> Limits:
        """The bounds this format holds a song to, at one compliance level."""
        return it_limits(compliance)

    def sample(self, name: str, pcm: NDArray[np.float64], *, rate: int, gain: int, depth: BitDepth) -> Sample:
        """One stored atom, its gain riding in the multiplier this format applies under the volume column."""
        return Sample(name=name, pcm=np.asarray(pcm, dtype=np.float64), rate=rate, depth=depth, gain=gain)

    def pattern_count(self, rows: int) -> int:
        """How many patterns a song of ``rows`` rows fills, at this format's row ceiling."""
        return max(1, -(-rows // MAX_ROWS))

    def module(self, song: Song, *, compliance: Compliance) -> ITModule:
        """Bind a song to this format at one compliance level."""
        return ITModule.from_song(song, compliance=compliance)


IT_BINDING: Final = ITBinding()
