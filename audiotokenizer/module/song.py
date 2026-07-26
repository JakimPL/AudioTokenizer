"""Assembling one compiled signal into the format-agnostic song a tracker module is written from.

This is the seam between the codec and ``trackmod``: everything above it reasons in atoms, coefficients
and channels, and everything below it in notes, instruments and patterns. The binding supplies the four
choices the two formats disagree on — the routing, where an atom's gain lives, how many patterns the song
is cut into, and which module class the song binds to — so the assembly itself is written once.
"""

from __future__ import annotations

from trackmod.core.samples.depth import BitDepth
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song

from audiotokenizer.coding.assignment import Assignment
from audiotokenizer.coding.quantisation import StoredAtoms
from audiotokenizer.module.binding import Binding
from audiotokenizer.module.instruments import dictionary_instruments
from audiotokenizer.module.patterns import dictionary_patterns
from audiotokenizer.module.samples import stored_samples
from audiotokenizer.module.slicing import pattern_slices


def build_song(
    binding: Binding,
    stored: StoredAtoms,
    assignment: Assignment,
    *,
    name: str,
    playback: Playback,
    rate: int,
    depth: BitDepth,
) -> Song:
    """The song a compiled signal plays: its stored atoms, the keys that reach them, and the channel grids."""
    samples = stored_samples(stored, binding, rate=rate, depth=depth)
    rows = int(assignment.sample_no.shape[0])
    slices = pattern_slices(rows, count=binding.pattern_count(rows))
    patterns = dictionary_patterns(
        assignment.sample_no,
        assignment.volume,
        binding.routing,
        slices=slices,
        restate=binding.restates_instrument,
    )
    return Song(
        name=name,
        channels=assignment.n_channels,
        patterns=patterns,
        order=OrderList.sequential(len(patterns)),
        instruments=dictionary_instruments(binding.routing, slots=len(samples)),
        samples=samples,
        playback=playback,
    )
