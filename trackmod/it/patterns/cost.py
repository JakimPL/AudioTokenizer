from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trackmod.it.spec.cells import CellMask


@dataclass(frozen=True)
class ColumnCost:
    """One column of a single channel, with what stating and repeating it cost in the stream."""

    values: NDArray[np.int64]
    fresh_bit: CellMask
    reuse_bit: CellMask
    byte_cost: int
