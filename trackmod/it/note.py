from typing import Final

from trackmod.core.notes.codec import decode_note as decode_shared_note
from trackmod.core.notes.command import NoteCommand, NoteValue
from trackmod.core.notes.pitch import Note
from trackmod.it.spec.cells import NoteByte
from trackmod.spec.pitch import NOTE_COUNT

COMMAND_BYTES: Final[dict[NoteCommand, NoteByte]] = {
    NoteCommand.OFF: NoteByte.OFF,
    NoteCommand.CUT: NoteByte.CUT,
    NoteCommand.FADE: NoteByte.FADE,
}
BYTE_COMMANDS: Final[dict[int, NoteCommand]] = {int(byte): command for command, byte in COMMAND_BYTES.items()}


def encode_note(note: NoteValue) -> int:
    """The note byte this format stores for a note-column entry.

    Impulse Tracker numbers its keyboard from C-0 exactly as the shared model does, so a key needs no
    offset; the commands occupy the top of the byte range.
    """
    match note:
        case NoteCommand():
            return int(COMMAND_BYTES[note])
        case Note():
            return note.value


def decode_note(byte: int) -> NoteValue:
    """The note-column entry a stored note byte stands for.

    Raises:
        ValueError: when the byte names neither a key nor a command this format defines.
    """
    command = BYTE_COMMANDS.get(byte)
    if command is not None:
        return command

    return Note(byte)


NOTE_BYTES: Final = tuple(encode_note(decode_shared_note(code)) for code in range(NOTE_COUNT + len(NoteCommand)))
