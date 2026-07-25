import numpy as np
from numpy.typing import NDArray

from trackmod.core.effects.effect import Effect
from trackmod.core.notes.codec import decode_note, encode_note
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.column import Column, Columns
from trackmod.spec.grid import EMPTY, GRID_DTYPE


def blank_columns(*, rows: int, channels: int) -> dict[Column, NDArray[np.int16]]:
    """A full set of ``(rows, channels)`` planes with every value absent."""
    return {column: np.full((rows, channels), EMPTY, dtype=GRID_DTYPE) for column in Column}


def read_cell(columns: Columns, row: int, channel: int) -> Cell:
    """Gather the five planes at one grid position back into a cell."""
    note = int(columns[Column.NOTE][row, channel])
    instrument = int(columns[Column.INSTRUMENT][row, channel])
    volume = int(columns[Column.VOLUME][row, channel])
    command = int(columns[Column.EFFECT][row, channel])
    parameter = int(columns[Column.PARAMETER][row, channel])
    return Cell(
        note=None if note == EMPTY else decode_note(note),
        instrument=None if instrument == EMPTY else instrument,
        volume=None if volume == EMPTY else volume,
        effect=None if command == EMPTY else Effect(command=command, parameter=max(parameter, 0)),
    )


def write_cell(columns: Columns, row: int, channel: int, cell: Cell) -> None:
    """Scatter a cell's four columns across the five planes at one grid position."""
    columns[Column.NOTE][row, channel] = EMPTY if cell.note is None else encode_note(cell.note)
    columns[Column.INSTRUMENT][row, channel] = EMPTY if cell.instrument is None else cell.instrument
    columns[Column.VOLUME][row, channel] = EMPTY if cell.volume is None else cell.volume
    columns[Column.EFFECT][row, channel] = EMPTY if cell.effect is None else cell.effect.command
    columns[Column.PARAMETER][row, channel] = EMPTY if cell.effect is None else cell.effect.parameter
