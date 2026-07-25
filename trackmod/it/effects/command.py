from enum import IntEnum, unique


@unique
class ITEffect(IntEnum):
    """The effect commands this format numbers ``A`` through ``Z``."""

    SET_SPEED = 1
    POSITION_JUMP = 2
    PATTERN_BREAK = 3
    VOLUME_SLIDE = 4
    PORTAMENTO_DOWN = 5
    PORTAMENTO_UP = 6
    TONE_PORTAMENTO = 7
    VIBRATO = 8
    TREMOR = 9
    ARPEGGIO = 10
    VIBRATO_VOLUME_SLIDE = 11
    PORTAMENTO_VOLUME_SLIDE = 12
    CHANNEL_VOLUME = 13
    CHANNEL_VOLUME_SLIDE = 14
    SAMPLE_OFFSET = 15
    PANNING_SLIDE = 16
    RETRIGGER = 17
    TREMOLO = 18
    EXTENDED = 19
    SET_TEMPO = 20
    FINE_VIBRATO = 21
    GLOBAL_VOLUME = 22
    GLOBAL_VOLUME_SLIDE = 23
    SET_PANNING = 24
    PANBRELLO = 25
    MIDI_MACRO = 26

    @property
    def letter(self) -> str:
        """The letter Impulse Tracker prints for this command."""
        return chr(ord("A") + self - 1)


@unique
class ITExtended(IntEnum):
    """The sub-commands the ``S`` effect selects with its high nibble."""

    GLISSANDO = 0x1
    FINETUNE = 0x2
    VIBRATO_WAVEFORM = 0x3
    TREMOLO_WAVEFORM = 0x4
    PANBRELLO_WAVEFORM = 0x5
    PATTERN_DELAY_TICKS = 0x6
    NOTE_CONTROL = 0x7
    PANNING = 0x8
    SOUND_CONTROL = 0x9
    HIGH_OFFSET = 0xA
    PATTERN_LOOP = 0xB
    NOTE_CUT = 0xC
    NOTE_DELAY = 0xD
    PATTERN_DELAY_ROWS = 0xE
    MIDI_MACRO_SELECT = 0xF
