"""The pattern grids a channel assignment becomes.

The assignment hands over two aligned ``(rows, channels)`` grids — which stored sample each cell plays and
at what volume — and a cell is present where its volume is positive. Turning that into a tracker's five
columns is where the routing is spent: the note column names the key that selects the sample, the
instrument column names the instrument that key belongs to, and the volume column carries the coefficient.

Whether a cell names its instrument at all is the binding's call, not a saving to be taken wherever it is
available. Leaving the column empty is what a tracker means by "keep playing the one you have", and a
format that stores an atom's gain in the instrument re-applies that gain only where a note states one —
so :class:`~audiotokenizer.module.it.ITBinding` restates it on every played cell and
:class:`~audiotokenizer.module.xm.XMBinding`, whose gain is already in the waveform, does not. Either way
the memory a channel keeps runs to the pattern boundary, so each grid is built from one pattern's rows
alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from trackmod.core.patterns.column import Column, Columns
from trackmod.core.patterns.grid import Pattern
from trackmod.spec.grid import EMPTY, GRID_DTYPE

from audiotokenizer.module.routing import Routing


def restated(instrument: NDArray[np.int64], present: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """A mask over the grid, true where a present cell names an instrument its channel is not carrying.

    A channel remembers across the rows it sits out, so the comparison runs over its present cells alone
    and its first one always states what it plays.
    """
    marks = np.zeros_like(present)
    for channel in range(present.shape[1]):
        rows = np.flatnonzero(present[:, channel])
        if rows.size == 0:
            continue

        values = instrument[rows, channel]
        stated = np.empty(rows.size, dtype=bool)
        stated[0] = True
        stated[1:] = values[1:] != values[:-1]
        marks[rows, channel] = stated
    return marks


def pattern_columns(
    sample_no: NDArray[np.int64],
    volume: NDArray[np.int64],
    routing: Routing,
    *,
    restate: bool,
) -> Columns:
    """The five column planes one pattern's slice of the assignment grids fills."""
    present = np.asarray(volume > 0)
    keys = routing.first_key + sample_no % routing.samples_per_instrument
    instruments = sample_no // routing.samples_per_instrument
    stated = present if restate else restated(instruments, present)
    blank = np.full(sample_no.shape, EMPTY, dtype=GRID_DTYPE)
    return {
        Column.NOTE: np.where(present, keys, EMPTY).astype(GRID_DTYPE),
        Column.INSTRUMENT: np.where(stated, instruments, EMPTY).astype(GRID_DTYPE),
        Column.VOLUME: np.where(present, volume, EMPTY).astype(GRID_DTYPE),
        Column.EFFECT: blank,
        Column.PARAMETER: blank.copy(),
    }


def dictionary_patterns(
    sample_no: NDArray[np.int64],
    volume: NDArray[np.int64],
    routing: Routing,
    *,
    slices: list[tuple[int, int]],
    restate: bool,
) -> tuple[Pattern, ...]:
    """One pattern per row span, each built from that span's rows alone."""
    return tuple(
        Pattern.from_columns(pattern_columns(sample_no[start:stop], volume[start:stop], routing, restate=restate))
        for start, stop in slices
    )
