from enum import IntEnum, unique
from typing import Final

DIGITS: Final = 10


@unique
class XMEffect(IntEnum):
    """The effect commands this format numbers, printed as ``0`` through ``9`` and then ``A`` onwards."""

    ARPEGGIO = 0x00
    PORTAMENTO_UP = 0x01
    PORTAMENTO_DOWN = 0x02
    TONE_PORTAMENTO = 0x03
    VIBRATO = 0x04
    TONE_PORTAMENTO_VOLUME_SLIDE = 0x05
    VIBRATO_VOLUME_SLIDE = 0x06
    TREMOLO = 0x07
    SET_PANNING = 0x08
    SAMPLE_OFFSET = 0x09
    VOLUME_SLIDE = 0x0A
    POSITION_JUMP = 0x0B
    SET_VOLUME = 0x0C
    PATTERN_BREAK = 0x0D
    EXTENDED = 0x0E
    SET_SPEED = 0x0F
    GLOBAL_VOLUME = 0x10
    GLOBAL_VOLUME_SLIDE = 0x11
    KEY_OFF = 0x14
    SET_ENVELOPE_POSITION = 0x15
    PANNING_SLIDE = 0x19
    MULTI_RETRIGGER = 0x1B
    TREMOR = 0x1D
    EXTRA_FINE_PORTAMENTO = 0x21

    @property
    def letter(self) -> str:
        """The character FastTracker 2 prints for this command."""
        if self < DIGITS:
            return chr(ord("0") + self)

        return chr(ord("A") + self - DIGITS)


@unique
class XMExtended(IntEnum):
    """The sub-commands the ``E`` effect selects with its high nibble."""

    FINE_PORTAMENTO_UP = 0x1
    FINE_PORTAMENTO_DOWN = 0x2
    GLISSANDO = 0x3
    VIBRATO_WAVEFORM = 0x4
    FINETUNE = 0x5
    PATTERN_LOOP = 0x6
    TREMOLO_WAVEFORM = 0x7
    PANNING = 0x8
    RETRIGGER = 0x9
    FINE_VOLUME_UP = 0xA
    FINE_VOLUME_DOWN = 0xB
    NOTE_CUT = 0xC
    NOTE_DELAY = 0xD
    PATTERN_DELAY = 0xE
