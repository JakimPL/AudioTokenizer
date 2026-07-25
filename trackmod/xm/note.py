from typing import Final

from trackmod.core.notes.codec import decode_note as decode_shared_note
from trackmod.core.notes.command import NoteCommand, NoteValue
from trackmod.core.notes.pitch import Note
from trackmod.spec.grid import EMPTY
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.xm.spec.cells import KEY_OFF, NOTE_OFFSET, UNSTORABLE
from trackmod.xm.spec.ranges import MAX_NOTE

COMMAND_BYTES: Final[dict[NoteCommand, int]] = {NoteCommand.OFF: KEY_OFF}
BYTE_COMMANDS: Final[dict[int, NoteCommand]] = {byte: command for command, byte in COMMAND_BYTES.items()}


def stored_byte(note: NoteValue) -> int | None:
    """The note byte this format stores, or ``None`` when its note column cannot state ``note``.

    FastTracker 2 numbers its keyboard from one and reaches eight octaves, so a key is stored one above
    the shared numbering and the two octaves above that have no byte. The column's only command is a key
    off, which releases the voice rather than cutting it outright.
    """
    match note:
        case NoteCommand():
            return COMMAND_BYTES.get(note)
        case Note():
            return note.value + NOTE_OFFSET if note.value <= MAX_NOTE else None


def refusal(note: NoteValue) -> str:
    """Why this format's note column cannot state ``note``."""
    match note:
        case NoteCommand():
            return f"the note column has no byte for {note.name}"
        case Note():
            return f"key {note} lies above {Note(MAX_NOTE)}, the highest key this format numbers"


def encode_note(note: NoteValue) -> int:
    """The note byte this format stores for a note-column entry.

    Raises:
        ValueError: when the note column cannot state ``note``.
    """
    byte = stored_byte(note)
    if byte is None:
        raise ValueError(refusal(note))

    return byte


def decode_note(byte: int) -> NoteValue:
    """The note-column entry a stored note byte stands for.

    Raises:
        ValueError: when the byte names neither a key nor a command this format defines.
    """
    command = BYTE_COMMANDS.get(byte)
    if command is not None:
        return command

    return Note(byte - NOTE_OFFSET)


NOTE_BYTES: Final = tuple(
    UNSTORABLE if byte is None else byte
    for byte in (stored_byte(decode_shared_note(code)) for code in range(NOTE_COUNT + len(NoteCommand)))
)


def stored_note(code: int) -> int:
    """The note byte a grid note code is written as, leaving an absent note absent.

    Raises:
        ValueError: when the code names a note this format's note column cannot state.
    """
    if code == EMPTY:
        return EMPTY

    byte = NOTE_BYTES[code]
    if byte == UNSTORABLE:
        raise ValueError(refusal(decode_shared_note(code)))

    return byte
