from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, Field

from trackmod.core.notes.pitch import Note
from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Index
from trackmod.spec.pitch import NOTE_COUNT


class KeyAssignment(BaseModel):
    """What one key of an instrument plays: a sample, sounded as if the given note had been pressed.

    Separating the pressed key from the sounded note is what lets an instrument route keys to samples
    without transposing them — every key can name the same note and still select a different sample.
    """

    model_config = FROZEN

    sample: Index
    note: Note


Keymap = Annotated[
    tuple[KeyAssignment | None, ...],
    Field(min_length=NOTE_COUNT, max_length=NOTE_COUNT),
]


def pitched_keymap(*, sample: int) -> Keymap:
    """A keymap where every key plays ``sample`` at that key's own pitch."""
    return tuple(KeyAssignment(sample=sample, note=Note(value)) for value in range(NOTE_COUNT))


def routed_keymap(assignments: Mapping[Note, KeyAssignment]) -> Keymap:
    """A keymap that answers only the given keys and leaves the rest silent."""
    return tuple(assignments.get(Note(value)) for value in range(NOTE_COUNT))
