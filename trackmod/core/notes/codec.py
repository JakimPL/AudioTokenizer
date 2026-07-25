from trackmod.core.notes.command import NoteCommand, NoteValue
from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import NOTE_COUNT


def encode_note(note: NoteValue) -> int:
    """The integer a note column stores for ``note``."""
    match note:
        case NoteCommand():
            return int(note)
        case Note():
            return note.value


def decode_note(code: int) -> NoteValue:
    """The note or command a note column's integer stands for.

    Raises:
        ValueError: when ``code`` names neither a key nor a command.
    """
    if code < NOTE_COUNT:
        return Note(code)

    return NoteCommand(code)
