from enum import IntEnum, IntFlag, unique


@unique
class HeaderFlag(IntFlag):
    """The song-wide switches the file header carries."""

    STEREO = 0x01
    VOLUME_ZERO_OPTIMISATION = 0x02
    USE_INSTRUMENTS = 0x04
    LINEAR_SLIDES = 0x08
    OLD_EFFECTS = 0x10
    LINK_GXX_MEMORY = 0x20


@unique
class SampleFlag(IntFlag):
    """The sample header's storage and looping switches."""

    DATA = 0x01
    SIXTEEN_BIT = 0x02
    STEREO = 0x04
    COMPRESSED = 0x08
    LOOP = 0x10
    SUSTAIN_LOOP = 0x20
    PING_PONG_LOOP = 0x40
    PING_PONG_SUSTAIN = 0x80


@unique
class SampleConvert(IntFlag):
    """How the stored frames are to be read."""

    SIGNED = 0x01


@unique
class SamplePanning(IntEnum):
    """The sample header's panning byte reserves its high bit as an enable switch."""

    ENABLED = 0x80


@unique
class EnvelopeFlag(IntFlag):
    """The switches at the head of each of an instrument's three envelopes."""

    ENABLED = 0x01
    LOOP = 0x02
    SUSTAIN = 0x04
    CARRY = 0x08
    FILTER = 0x80
