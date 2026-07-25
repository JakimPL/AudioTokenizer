from __future__ import annotations

import math

import pydantic
from pydantic import BaseModel

from trackmod.core.notes.pitch import Note
from trackmod.schema.config import FROZEN
from trackmod.spec.pitch import NOTES_PER_OCTAVE, RATE_NOTE
from trackmod.xm.spec.tuning import (
    FINETUNE_UNITS,
    MAX_FINETUNE,
    MAX_RELATIVE_NOTE,
    MIN_FINETUNE,
    MIN_RELATIVE_NOTE,
    REFERENCE_NOTE,
    REFERENCE_RATE,
)


class Tuning(BaseModel):
    """How far a stored sample is transposed from the key that triggers it.

    FastTracker 2 stores no playback rate. A sample sounds at the pitch of the key it is played on,
    shifted by ``relative_note`` whole semitones and trimmed by ``finetune`` in units of one
    :data:`~trackmod.xm.spec.tuning.FINETUNE_UNITS`-th of a semitone, so a rate is expressed as a
    transposition rather than stored outright.
    """

    model_config = FROZEN

    relative_note: int = pydantic.Field(ge=MIN_RELATIVE_NOTE, le=MAX_RELATIVE_NOTE)
    finetune: int = pydantic.Field(ge=MIN_FINETUNE, le=MAX_FINETUNE)

    @property
    def units(self) -> int:
        """The whole transposition, counted in finetune units."""
        return FINETUNE_UNITS * self.relative_note + self.finetune


def pitch_units(rate: int) -> int:
    """The key, in finetune units, whose pitch a sample recorded at ``rate`` sounds at untransposed.

    The format's linear frequency table places a key ``n`` semitones above the reference at
    ``REFERENCE_RATE * 2 ** (n / 12)``, which inverts to a logarithm the finetune lattice then rounds.
    """
    semitones = REFERENCE_NOTE + NOTES_PER_OCTAVE * math.log2(rate / REFERENCE_RATE)
    return round(FINETUNE_UNITS * semitones)


def unit_rate(units: int) -> int:
    """The rate a sample plays at when its transposed key lands on ``units``."""
    semitones = units / FINETUNE_UNITS - REFERENCE_NOTE
    return round(REFERENCE_RATE * 2 ** (semitones / NOTES_PER_OCTAVE))


def tuning_for(rate: int, *, key: Note, sounded: Note) -> Tuning:
    """The transposition that sounds a sample recorded at ``rate`` as ``sounded`` when ``key`` is pressed.

    The remainder is taken towards negative infinity, so the finetune trim is always positive and a
    whole-semitone tuning spends nothing on it.

    Raises:
        ValueError: when the transposition needed leaves the range the relative-note byte holds.
    """
    offset = FINETUNE_UNITS * (sounded.value - RATE_NOTE - key.value)
    relative_note, finetune = divmod(pitch_units(rate) + offset, FINETUNE_UNITS)
    return Tuning(relative_note=relative_note, finetune=finetune)


def tuned_rate(tuning: Tuning, *, key: Note, sounded: Note) -> int:
    """The rate a sample stored with ``tuning`` was recorded at, given how its key is meant to sound."""
    offset = FINETUNE_UNITS * (sounded.value - RATE_NOTE - key.value)
    return unit_rate(tuning.units - offset)
