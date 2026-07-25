from enum import IntEnum, IntFlag, unique
from typing import Final


@unique
class HeaderFlag(IntFlag):
    """The song-wide switches the file header carries."""

    LINEAR_FREQUENCY = 0x01


@unique
class SampleFlag(IntFlag):
    """The storage switch a sample header's type byte carries beside its loop mode."""

    SIXTEEN_BIT = 0x10


@unique
class LoopType(IntEnum):
    """The loop mode a sample header's type byte selects with its low two bits."""

    NONE = 0x00
    FORWARD = 0x01
    PING_PONG = 0x02


LOOP_TYPE_MASK: Final = 0x03


@unique
class EnvelopeFlag(IntFlag):
    """The switches at the head of each of an instrument's two envelopes."""

    ENABLED = 0x01
    SUSTAIN = 0x02
    LOOP = 0x04
