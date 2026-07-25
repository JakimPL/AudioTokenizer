from trackmod.binary.cursor import Cursor
from trackmod.binary.records.values import read_int
from trackmod.core.effects.effect import Effect
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.it.layout.pattern import PATTERN_HEADER
from trackmod.it.note import decode_note
from trackmod.it.patterns.memory import ChannelMemory
from trackmod.it.spec.cells import (
    CHANNEL_MARKER,
    END_OF_ROW,
    INSTRUMENT_OFFSET,
    NO_INSTRUMENT,
    UNSET,
    CellMask,
)
from trackmod.spec.grid import EMPTY


def decode_column(cursor: Cursor, mask: int, *, fresh: CellMask, reuse: CellMask, remembered: int) -> tuple[int, int]:
    """The value this cell carries and the value the channel goes on remembering.

    A column the mask leaves out is absent from the cell and leaves the channel's memory untouched, so a
    later cell can still reuse the value that was last stated.
    """
    if mask & fresh:
        stated = cursor.take(1)[0]
        return stated, stated

    if mask & reuse:
        return remembered, remembered

    return EMPTY, remembered


def decode_effect(cursor: Cursor, mask: int, memory: ChannelMemory) -> Effect | None:
    """The effect this cell carries, reading the command and parameter bytes only when they are stated."""
    if mask & CellMask.EFFECT:
        memory.command = cursor.take(1)[0]
        memory.parameter = cursor.take(1)[0]
    elif not mask & CellMask.LAST_EFFECT:
        return None

    if memory.command == UNSET:
        return None

    return Effect(command=memory.command, parameter=max(memory.parameter, 0))


def decode_cell(cursor: Cursor, memory: ChannelMemory) -> Cell:
    """One cell, read against the mask and the values its channel last stated."""
    mask = memory.mask
    note, memory.note = decode_column(
        cursor,
        mask,
        fresh=CellMask.NOTE,
        reuse=CellMask.LAST_NOTE,
        remembered=memory.note,
    )
    instrument, memory.instrument = decode_column(
        cursor,
        mask,
        fresh=CellMask.INSTRUMENT,
        reuse=CellMask.LAST_INSTRUMENT,
        remembered=memory.instrument,
    )
    volume, memory.volume = decode_column(
        cursor,
        mask,
        fresh=CellMask.VOLUME,
        reuse=CellMask.LAST_VOLUME,
        remembered=memory.volume,
    )
    return Cell(
        note=None if note == EMPTY else decode_note(note),
        instrument=None if instrument in (EMPTY, NO_INSTRUMENT) else instrument - INSTRUMENT_OFFSET,
        volume=None if volume == EMPTY else volume,
        effect=decode_effect(cursor, mask, memory),
    )


def unpack_cells(stream: bytes, *, rows: int) -> Pattern:
    """Rebuild a pattern grid from a packed cell stream, sized to the widest channel it reaches.

    Raises:
        ValueError: when a channel reuses a mask it has never stated, which no writer produces.
    """
    cursor = Cursor(stream)
    memories: dict[int, ChannelMemory] = {}
    placed: list[tuple[int, int, Cell]] = []
    for row in range(rows):
        marker = cursor.take(1)[0]
        while marker != END_OF_ROW:
            channel = (marker & ~CHANNEL_MARKER) - 1
            memory = memories.setdefault(channel, ChannelMemory())
            if marker & CHANNEL_MARKER:
                memory.mask = cursor.take(1)[0]
            elif memory.mask == UNSET:
                raise ValueError(f"channel {channel} reuses a mask it has not stated, at row {row}")

            placed.append((row, channel, decode_cell(cursor, memory)))
            marker = cursor.take(1)[0]

    channels = max((channel for _, channel, _ in placed), default=0) + 1
    builder = PatternBuilder(rows=rows, channels=channels)
    for row, channel, cell in placed:
        builder.place(row, channel, cell)

    return builder.build()


def unpack_pattern(cursor: Cursor) -> Pattern:
    """Read one pattern — its header and its cell stream — from the cursor's position."""
    header = cursor.read(PATTERN_HEADER)
    stream = cursor.take(read_int(header, "packed_size"))
    return unpack_cells(stream, rows=read_int(header, "rows"))
