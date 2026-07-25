import numpy as np

from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.codec import read_cell, write_cell
from trackmod.core.patterns.column import Column
from trackmod.core.patterns.grid import Pattern
from trackmod.spec.grid import EMPTY


class PatternBuilder:
    """Fills a pattern grid cell by cell, then freezes it into a :class:`Pattern`.

    Authoring a pattern is inherently incremental — notes land first, releases and global effects land on
    top of rows already written — so the mutable side is its own type and the finished grid stays frozen.
    """

    def __init__(self, *, rows: int, channels: int) -> None:
        blank = Pattern.empty(rows=rows, channels=channels)
        self._rows = blank.rows
        self._channels = blank.channels
        self._columns = dict(blank.columns)

    @property
    def rows(self) -> int:
        """How many rows the grid holds."""
        return self._rows

    @property
    def channels(self) -> int:
        """How many channels the grid is wide."""
        return self._channels

    def read(self, row: int, channel: int) -> Cell:
        """The cell currently written at one grid position."""
        return read_cell(self._columns, row, channel)

    def place(self, row: int, channel: int, cell: Cell) -> None:
        """Write ``cell`` at one grid position, replacing every column already there.

        Raises:
            IndexError: when the position lies outside the grid.
        """
        if not 0 <= row < self._rows or not 0 <= channel < self._channels:
            raise IndexError(f"position ({row}, {channel}) is outside the {self._rows}x{self._channels} grid")

        write_cell(self._columns, row, channel, cell)

    def free_effect_channel(self, row: int) -> int | None:
        """The lowest channel on ``row`` whose effect column is still free, or ``None`` when all are taken.

        A global effect such as a tempo change belongs to the row rather than to a voice, so it goes
        wherever there is room; scanning from the lowest channel keeps the choice deterministic.
        """
        free = np.flatnonzero(self._columns[Column.EFFECT][row] == EMPTY)
        return int(free[0]) if free.size else None

    def build(self) -> Pattern:
        """Freeze the grid written so far."""
        return Pattern.from_columns({column: plane.copy() for column, plane in self._columns.items()})
