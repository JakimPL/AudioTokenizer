from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator

from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.codec import blank_columns, read_cell
from trackmod.core.patterns.column import Column, Columns
from trackmod.schema.array import Grid
from trackmod.schema.config import FROZEN
from trackmod.spec.grid import EMPTY, GRID_DTYPE, MIN_CHANNELS, MIN_ROWS


class Pattern(BaseModel):
    """A grid of cells stored as five aligned ``(rows, channels)`` planes.

    Columns are kept apart rather than as objects so presence is decided per column — the packers and the
    size models read whole planes at once, and a cell may hold a note with no volume. The grid sentinel
    marks an absent value in every plane.
    """

    model_config = FROZEN

    note: Grid
    instrument: Grid
    volume: Grid
    effect: Grid
    parameter: Grid

    @model_validator(mode="after")
    def _aligned(self) -> Pattern:
        shapes = {column: plane.shape for column, plane in self.columns.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"pattern columns disagree on shape: {shapes}")

        if self.rows < MIN_ROWS or self.channels < MIN_CHANNELS:
            raise ValueError(f"pattern size {self.rows}x{self.channels} must be at least {MIN_ROWS}x{MIN_CHANNELS}")

        return self

    @classmethod
    def empty(cls, *, rows: int, channels: int) -> Pattern:
        """A pattern of the given size with every column absent."""
        return cls.from_columns(blank_columns(rows=rows, channels=channels))

    @classmethod
    def from_columns(cls, columns: Columns) -> Pattern:
        """A pattern built from a full set of column planes."""
        return cls(
            note=columns[Column.NOTE],
            instrument=columns[Column.INSTRUMENT],
            volume=columns[Column.VOLUME],
            effect=columns[Column.EFFECT],
            parameter=columns[Column.PARAMETER],
        )

    @property
    def columns(self) -> Columns:
        """Every column plane, keyed by the column it holds."""
        return {
            Column.NOTE: self.note,
            Column.INSTRUMENT: self.instrument,
            Column.VOLUME: self.volume,
            Column.EFFECT: self.effect,
            Column.PARAMETER: self.parameter,
        }

    @property
    def rows(self) -> int:
        """How many rows the grid holds."""
        return int(self.note.shape[0])

    @property
    def channels(self) -> int:
        """How many channels the grid is wide."""
        return int(self.note.shape[1])

    @property
    def occupied(self) -> NDArray[np.bool_]:
        """A ``(rows, channels)`` mask that is true where any column carries a value."""
        return np.asarray(
            (self.note != EMPTY) | (self.instrument != EMPTY) | (self.volume != EMPTY) | (self.effect != EMPTY)
        )

    def widened(self, channels: int) -> Pattern:
        """The same grid padded with silent channels out to ``channels``.

        Formats that store one channel count for a whole module need every pattern at that width, while
        a packed pattern only reaches as far as its widest occupied channel.

        Raises:
            ValueError: when the grid is already wider than the requested width.
        """
        if channels < self.channels:
            raise ValueError(f"cannot narrow a {self.channels}-channel pattern to {channels}")

        if channels == self.channels:
            return self

        padding = np.full((self.rows, channels - self.channels), EMPTY, dtype=GRID_DTYPE)
        return Pattern.from_columns(
            {column: np.concatenate([plane, padding], axis=1) for column, plane in self.columns.items()}
        )

    def column(self, column: Column) -> NDArray[np.int16]:
        """One column plane."""
        return self.columns[column]

    def cell(self, row: int, channel: int) -> Cell:
        """The cell at one grid position."""
        return read_cell(self.columns, row, channel)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return NotImplemented

        return all(np.array_equal(plane, other.column(column)) for column, plane in self.columns.items())

    def __hash__(self) -> int:
        return hash((self.rows, self.channels))
