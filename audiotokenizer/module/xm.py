"""Compiling the codec's output into a FastTracker 2 module.

Two of this format's decisions shape everything here. Its volume column *overrides* the sample volume byte
rather than scaling it, so an atom's static gain has to be baked into the stored waveform — the byte would
be erased the moment a row states its coefficient — at the cost of resolution in a quiet atom, which is
the price of having one lattice where Impulse Tracker has two. With nothing left in the instrument for a
note to re-apply, a cell names one only when the channel is not already carrying it. And its instruments
own 16 samples each, keyed from C-2 upward: low enough that the whole block of 16 sits inside the 96 keys
this format numbers, and high enough that the transposition tuning each sample back to its recorded rate
fits a signed byte.

Its packed patterns keep no memory at all, so a pattern's height costs nothing in cell bytes and the song
is spread over as many patterns as the 256-entry order table holds. That keeps every packed pattern well
under the 16-bit length its header stores, which is the wall a wide, densely played song would otherwise
hit.
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
from trackmod.spec.levels import MAX_VOLUME
from trackmod.trackers.xm.limits import xm_limits
from trackmod.trackers.xm.module import XMModule
from trackmod.trackers.xm.settings import XMSettings
from trackmod.trackers.xm.spec.ranges import CANONICAL_MAX_PATTERNS, CANONICAL_SAMPLES_PER_INSTRUMENT

from audiotokenizer.module.binding import gained
from audiotokenizer.module.format import Format
from audiotokenizer.module.routing import Routing

FIRST_KEY: Final = 24  # C-2, leaving room below for the transposition and above for the whole block
TRACKER_NAME: Final = "AudioTokenizer"
ROUTING: Final = Routing(samples_per_instrument=CANONICAL_SAMPLES_PER_INSTRUMENT, first_key=FIRST_KEY)
SETTINGS: Final = XMSettings(tracker=TRACKER_NAME)


@dataclass(frozen=True)
class XMBinding:
    """FastTracker 2 as this codec writes it."""

    format: Format = Format.XM
    routing: Routing = field(default=ROUTING)
    restates_instrument: bool = False

    def limits(self, compliance: Compliance) -> Limits:
        """The bounds this format holds a song to, at one compliance level."""
        return xm_limits(compliance)

    def sample(self, name: str, pcm: NDArray[np.float64], *, rate: int, gain: int, depth: BitDepth) -> Sample:
        """One stored atom, its gain baked into the waveform, which is the only place this format keeps it."""
        return Sample(name=name, pcm=gained(pcm, gain, levels=MAX_VOLUME), rate=rate, depth=depth)

    def pattern_count(self, rows: int) -> int:
        """How many patterns a song of ``rows`` rows is spread over: the shortest the order table allows.

        A packed pattern's length is a 16-bit field and this format stores every cell of every row, so the
        shorter a pattern is the further a wide, densely played song stays from that wall. Cutting at the
        height the 256-entry order table forces is therefore the most feasible split, and it costs nothing
        in cell bytes because the packing keeps no per-pattern memory to reset.
        """
        height = max(1, -(-rows // CANONICAL_MAX_PATTERNS))
        return max(1, -(-rows // height))

    def module(self, song: Song, *, compliance: Compliance) -> XMModule:
        """Bind a song to this format at one compliance level."""
        return XMModule.from_song(song, compliance=compliance, settings=SETTINGS)


XM_BINDING: Final = XMBinding()
