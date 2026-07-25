from enum import IntEnum, unique

from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import NOTE_COUNT


@unique
class NoteCommand(IntEnum):
    """A note-column entry that acts on the playing voice instead of starting a pitch.

    The values continue past the key range so a note column can hold either a :class:`Note` or a command
    in one integer.
    """

    OFF = NOTE_COUNT
    CUT = NOTE_COUNT + 1
    FADE = NOTE_COUNT + 2


NoteValue = Note | NoteCommand
