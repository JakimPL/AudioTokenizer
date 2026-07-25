from enum import IntEnum, IntFlag, unique
from typing import Final

CHANNEL_MARKER: Final = 0x80
END_OF_ROW: Final = 0x00
UNSET: Final = -1

CHANNEL_BYTE: Final = 1
MASK_BYTE: Final = 1
COLUMN_BYTE: Final = 1
EFFECT_BYTES: Final = 2
ROW_TERMINATOR_BYTE: Final = 1


@unique
class CellMask(IntFlag):
    """The mask byte of a packed cell: which columns follow, and which repeat the channel's last value.

    The reuse bits are what make a packed pattern small — a column that repeats costs a bit rather than
    a byte, and re-stating a note still re-triggers the sample.
    """

    NOTE = 0x01
    INSTRUMENT = 0x02
    VOLUME = 0x04
    EFFECT = 0x08
    LAST_NOTE = 0x10
    LAST_INSTRUMENT = 0x20
    LAST_VOLUME = 0x40
    LAST_EFFECT = 0x80


@unique
class NoteByte(IntEnum):
    """The note-column values that act on the playing voice instead of naming a key."""

    FADE = 253
    CUT = 254
    OFF = 255


INSTRUMENT_OFFSET: Final = 1
NO_INSTRUMENT: Final = 0
