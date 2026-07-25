from dataclasses import dataclass

from trackmod.spec.width import BYTE_MAX
from trackmod.xm.spec.cells import RAW_CELL_COLUMNS, CellMask

StatedColumn = tuple[CellMask, int]


@dataclass(frozen=True)
class EncodedCell:
    """The columns one cell states, each carrying the mask bit that announces it."""

    columns: tuple[StatedColumn, ...]

    @property
    def mask(self) -> int:
        """The mask byte announcing exactly these columns."""
        mask = int(CellMask.PACKED)
        for bit, _ in self.columns:
            mask |= bit

        return mask

    @property
    def complete(self) -> bool:
        """Whether every column is stated, which is what the uncompressed form requires."""
        return len(self.columns) == RAW_CELL_COLUMNS

    def to_bytes(self) -> bytes:
        """The cell as it is stored, uncompressed when that is the shorter of the two forms."""
        values = bytes(value & BYTE_MAX for _, value in self.columns)
        return values if self.complete else bytes([self.mask]) + values
