from enum import IntFlag, unique
from typing import Final

NOTE_OFFSET: Final = 1
KEY_OFF: Final = 97
UNSTORABLE: Final = -1

INSTRUMENT_OFFSET: Final = 1
NO_INSTRUMENT: Final = 0

VOLUME_COLUMN_BASE: Final = 0x10
VOLUME_COLUMN_EMPTY: Final = 0x00

NO_EFFECT: Final = 0

RAW_CELL_COLUMNS: Final = 5
PACKED_BYTE: Final = 1
COLUMN_BYTE: Final = 1
RAW_CELL_BYTES: Final = RAW_CELL_COLUMNS * COLUMN_BYTE


@unique
class CellMask(IntFlag):
    """The first byte of a packed cell: which columns follow it.

    A byte with the high bit set is a mask and the columns it names follow in this order. A byte with
    the high bit clear is itself the note, and all five columns follow uncompressed — which is why a
    cell that states every column costs one byte less than its mask would.
    """

    NOTE = 0x01
    INSTRUMENT = 0x02
    VOLUME = 0x04
    EFFECT = 0x08
    PARAMETER = 0x10
    PACKED = 0x80
