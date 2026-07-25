from trackmod.core.instruments.keymap import KeyAssignment, Keymap
from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.xm.spec.sizes import KEYMAP_NOTES


def stored_keymap(slots: tuple[int, ...]) -> bytes:
    """The keymap block a stored instrument carries: one sample position per key it numbers."""
    return bytes(slots)


def parse_keymap(raw: bytes, *, offset: int, length: int) -> Keymap:
    """Rebuild a keymap from the stored sample positions, resolved against the module's samples.

    A stored keymap has no way to leave a key silent — every byte names a position, and an instrument
    with no samples simply never plays — so a key is read as silent only where it falls outside the keys
    this format numbers or names a position the instrument does not own.
    """
    return tuple(
        KeyAssignment(sample=offset + raw[key], note=Note(key)) if key < KEYMAP_NOTES and raw[key] < length else None
        for key in range(NOTE_COUNT)
    )
