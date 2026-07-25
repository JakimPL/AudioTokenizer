from enum import IntEnum, unique


@unique
class NewNoteAction(IntEnum):
    """What happens to a channel's playing voice when the same channel starts a new note."""

    CUT = 0
    CONTINUE = 1
    NOTE_OFF = 2
    FADE = 3


@unique
class DuplicateCheck(IntEnum):
    """What makes two voices of one instrument count as duplicates."""

    OFF = 0
    NOTE = 1
    SAMPLE = 2
    INSTRUMENT = 3


@unique
class DuplicateAction(IntEnum):
    """What happens to the voice a duplicate check finds."""

    CUT = 0
    NOTE_OFF = 1
    FADE = 2
