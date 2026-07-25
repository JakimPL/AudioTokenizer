from typing import Final

import numpy as np
from numpy.typing import NDArray

from trackmod.core.patterns.grid import Pattern
from trackmod.it.patterns.cost import ColumnCost
from trackmod.it.spec.cells import CHANNEL_BYTE, COLUMN_BYTE, EFFECT_BYTES, MASK_BYTE, ROW_TERMINATOR_BYTE, CellMask
from trackmod.spec.grid import EMPTY
from trackmod.spec.width import BYTE_MAX

PARAMETER_RADIX: Final = BYTE_MAX + 1


def fresh_states(column: ColumnCost) -> NDArray[np.bool_]:
    """Where the column states a value the channel has not most recently stated."""
    fresh = np.zeros(column.values.shape, dtype=bool)
    rows = np.flatnonzero(column.values != EMPTY)
    if rows.size == 0:
        return fresh

    stated = column.values[rows]
    changed = np.empty(rows.size, dtype=bool)
    changed[0] = True
    changed[1:] = stated[1:] != stated[:-1]
    fresh[rows] = changed
    return fresh


def mask_bits(column: ColumnCost, fresh: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The mask bits this column contributes to each row of its channel."""
    present = column.values != EMPTY
    stated = np.where(fresh, int(column.fresh_bit), 0)
    repeated = np.where(present & ~fresh, int(column.reuse_bit), 0)
    return np.asarray(stated | repeated, dtype=np.int64)


def effect_key(commands: NDArray[np.int16], parameters: NDArray[np.int16]) -> NDArray[np.int64]:
    """One comparable value per cell for the command and parameter a channel remembers together."""
    combined = commands.astype(np.int64) * PARAMETER_RADIX + np.maximum(parameters.astype(np.int64), 0)
    return np.asarray(np.where(commands == EMPTY, EMPTY, combined))


def channel_columns(pattern: Pattern, channel: int) -> tuple[ColumnCost, ...]:
    """The four columns one channel spends bytes on, in the order a packed cell states them."""
    return (
        ColumnCost(
            values=pattern.note[:, channel].astype(np.int64),
            fresh_bit=CellMask.NOTE,
            reuse_bit=CellMask.LAST_NOTE,
            byte_cost=COLUMN_BYTE,
        ),
        ColumnCost(
            values=pattern.instrument[:, channel].astype(np.int64),
            fresh_bit=CellMask.INSTRUMENT,
            reuse_bit=CellMask.LAST_INSTRUMENT,
            byte_cost=COLUMN_BYTE,
        ),
        ColumnCost(
            values=pattern.volume[:, channel].astype(np.int64),
            fresh_bit=CellMask.VOLUME,
            reuse_bit=CellMask.LAST_VOLUME,
            byte_cost=COLUMN_BYTE,
        ),
        ColumnCost(
            values=effect_key(pattern.effect[:, channel], pattern.parameter[:, channel]),
            fresh_bit=CellMask.EFFECT,
            reuse_bit=CellMask.LAST_EFFECT,
            byte_cost=EFFECT_BYTES,
        ),
    )


def channel_bytes(pattern: Pattern, channel: int) -> int:
    """How many bytes one channel spends across a whole pattern."""
    occupied = pattern.occupied[:, channel]
    if not occupied.any():
        return 0

    masks = np.zeros(pattern.rows, dtype=np.int64)
    columns = 0
    for column in channel_columns(pattern, channel):
        fresh = fresh_states(column)
        masks |= mask_bits(column, fresh)
        columns += column.byte_cost * int(fresh.sum())

    stated = masks[occupied]
    changed = np.empty(stated.size, dtype=bool)
    changed[0] = True
    changed[1:] = stated[1:] != stated[:-1]
    return CHANNEL_BYTE * int(stated.size) + MASK_BYTE * int(changed.sum()) + columns


def packed_bytes(pattern: Pattern) -> int:
    """How many bytes a pattern's packed cell stream occupies, without building it.

    This is the exact counterpart of the packer, computed a whole channel at a time: the reuse bits make
    a column's cost depend only on that channel's previous stated value, so each column reduces to one
    run-length comparison rather than a walk over the grid.
    """
    total = ROW_TERMINATOR_BYTE * pattern.rows
    for channel in range(pattern.channels):
        total += channel_bytes(pattern, channel)

    return total
