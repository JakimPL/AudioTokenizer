from __future__ import annotations

from enum import IntEnum, unique

import pydantic
from pydantic import RootModel

from trackmod.schema.config import FROZEN_ROOT
from trackmod.spec.pitch import MIDI_OFFSET, NOTE_COUNT, NOTES_PER_OCTAVE, PITCH_LABELS


@unique
class PitchClass(IntEnum):
    """A semitone within an octave, counted from C."""

    C = 0
    C_SHARP = 1
    D = 2
    D_SHARP = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    G_SHARP = 8
    A = 9
    A_SHARP = 10
    B = 11

    @property
    def label(self) -> str:
        """The two-character tracker spelling, e.g. ``"C-"`` or ``"F#"``."""
        return PITCH_LABELS[self]


class Note(RootModel[int]):
    """A playable key, counted in semitones above C-0.

    Trackers number their keyboards from C-0, and the octave a tracker prints is one above the octave the
    same pitch carries in MIDI: tracker C-5 is MIDI note 72. The stored number is the tracker numbering,
    which Impulse Tracker writes directly and FastTracker 2 writes offset by one.
    """

    model_config = FROZEN_ROOT

    root: int = pydantic.Field(ge=0, lt=NOTE_COUNT)

    @classmethod
    def of(cls, pitch_class: PitchClass, octave: int) -> Note:
        """The note named by a pitch class and a tracker octave."""
        return cls(octave * NOTES_PER_OCTAVE + pitch_class)

    @classmethod
    def from_midi(cls, midi: int) -> Note:
        """The note a MIDI key number names."""
        return cls(midi - MIDI_OFFSET)

    @property
    def value(self) -> int:
        """The key number, counted in semitones above C-0."""
        return self.root

    @property
    def pitch_class(self) -> PitchClass:
        """The semitone within the octave."""
        return PitchClass(self.root % NOTES_PER_OCTAVE)

    @property
    def octave(self) -> int:
        """The tracker octave, 0..9."""
        return self.root // NOTES_PER_OCTAVE

    @property
    def midi(self) -> int:
        """The MIDI key number for this pitch."""
        return self.root + MIDI_OFFSET

    def transposed(self, semitones: int) -> Note:
        """The note ``semitones`` above this one.

        Raises:
            ValueError: when the result leaves the key range.
        """
        return Note(self.root + semitones)

    def __lt__(self, other: Note) -> bool:
        return self.root < other.root

    def __str__(self) -> str:
        return f"{self.pitch_class.label}{self.octave}"
