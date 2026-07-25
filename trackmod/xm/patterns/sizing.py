import numpy as np
from numpy.typing import NDArray

from trackmod.core.patterns.column import Column
from trackmod.core.patterns.grid import Pattern
from trackmod.spec.grid import EMPTY
from trackmod.xm.spec.cells import COLUMN_BYTE, PACKED_BYTE, RAW_CELL_BYTES, RAW_CELL_COLUMNS


def stated_counts(pattern: Pattern) -> NDArray[np.int64]:
    """How many columns each grid position states."""
    planes = [pattern.column(column) != EMPTY for column in Column]
    return np.sum(planes, axis=0, dtype=np.int64)


def cell_bytes(stated: NDArray[np.int64]) -> NDArray[np.int64]:
    """How many bytes each cell occupies, given how many columns it states."""
    return np.where(stated == RAW_CELL_COLUMNS, RAW_CELL_BYTES, PACKED_BYTE + COLUMN_BYTE * stated)


def packed_bytes(pattern: Pattern) -> int:
    """How many bytes a pattern's stored cell stream occupies, without building it.

    This is the exact counterpart of the packer. The format keeps no memory between cells, so a cell's
    cost depends on nothing but which of its own columns are present, and the whole pattern reduces to a
    sum over the grid.
    """
    return int(cell_bytes(stated_counts(pattern)).sum())
