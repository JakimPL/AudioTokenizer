from trackmod.binary.records.values import ArrayValue
from trackmod.core.instruments.keymap import KeyAssignment, Keymap
from trackmod.core.notes.pitch import Note
from trackmod.it.spec.defaults import NO_SAMPLE


def note_map(keymap: Keymap) -> tuple[tuple[int, int], ...]:
    """The ``(played note, sample number)`` pair each key stores.

    Sample numbers are one-based here, so zero is what silences a key; an unmapped key still names its
    own pitch, which is the identity mapping a tracker writes for an instrument with nothing routed yet.
    """
    return tuple(
        (key, NO_SAMPLE) if assignment is None else (assignment.note.value, assignment.sample + 1)
        for key, assignment in enumerate(keymap)
    )


def parse_keymap(rows: ArrayValue) -> Keymap:
    """Rebuild a keymap from the stored ``(played note, sample number)`` pairs."""
    return tuple(
        None if sample == NO_SAMPLE else KeyAssignment(sample=sample - 1, note=Note(played)) for played, sample in rows
    )
