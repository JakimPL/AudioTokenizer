from typing import Final

from trackmod.binary.cursor import Cursor
from trackmod.binary.records.values import read_int
from trackmod.core.effects.effect import Effect
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.spec.grid import EMPTY
from trackmod.spec.levels import MAX_VOLUME
from trackmod.xm.layout.pattern import PATTERN_HEADER
from trackmod.xm.note import decode_note
from trackmod.xm.spec.cells import (
    INSTRUMENT_OFFSET,
    NO_EFFECT,
    NO_INSTRUMENT,
    RAW_CELL_COLUMNS,
    VOLUME_COLUMN_BASE,
    CellMask,
)

COLUMN_BITS: Final = (
    CellMask.NOTE,
    CellMask.INSTRUMENT,
    CellMask.VOLUME,
    CellMask.EFFECT,
    CellMask.PARAMETER,
)


def stated_columns(cursor: Cursor) -> list[int]:
    """The five column values one stored cell carries, absent where it states nothing.

    A first byte with the high bit clear is itself the note and the other four columns follow it whole;
    otherwise the byte is a mask and only the columns it names are present.
    """
    first = cursor.take(1)[0]
    if not first & CellMask.PACKED:
        return [first, *cursor.take(RAW_CELL_COLUMNS - 1)]

    return [cursor.take(1)[0] if first & bit else EMPTY for bit in COLUMN_BITS]


def decode_volume(volume: int) -> int | None:
    """The level a volume-column byte sets, or ``None`` when it holds one of the column's own effects.

    Only the setting range maps onto a level; the slides and vibrato the column also encodes have no
    place in the shared model and are dropped rather than misread as a level.
    """
    if not VOLUME_COLUMN_BASE <= volume <= VOLUME_COLUMN_BASE + MAX_VOLUME:
        return None

    return volume - VOLUME_COLUMN_BASE


def decode_effect(command: int, parameter: int) -> Effect | None:
    """The effect a cell carries, which an empty command with an empty parameter does not."""
    stated, argument = max(command, NO_EFFECT), max(parameter, NO_EFFECT)
    if stated == NO_EFFECT and argument == NO_EFFECT:
        return None

    return Effect(command=stated, parameter=argument)


def decode_cell(cursor: Cursor) -> Cell:
    """One cell, read from the cursor's position."""
    note, instrument, volume, command, parameter = stated_columns(cursor)
    return Cell(
        note=None if note == EMPTY else decode_note(note),
        instrument=None if instrument in (EMPTY, NO_INSTRUMENT) else instrument - INSTRUMENT_OFFSET,
        volume=None if volume == EMPTY else decode_volume(volume),
        effect=decode_effect(command, parameter),
    )


def unpack_cells(stream: bytes, *, rows: int, channels: int) -> Pattern:
    """Rebuild a pattern grid from a stored cell stream of a known size."""
    if not stream:
        return Pattern.empty(rows=rows, channels=channels)

    cursor = Cursor(stream)
    builder = PatternBuilder(rows=rows, channels=channels)
    for row in range(rows):
        for channel in range(channels):
            builder.place(row, channel, decode_cell(cursor))

    return builder.build()


def unpack_pattern(cursor: Cursor, *, channels: int) -> Pattern:
    """Read one pattern — its header and its cell stream — from the cursor's position."""
    header = cursor.read(PATTERN_HEADER)
    rows = read_int(header, "rows")
    stream = cursor.take(read_int(header, "packed_size"))
    return unpack_cells(stream, rows=rows, channels=channels)
