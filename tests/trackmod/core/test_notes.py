from __future__ import annotations

from dataclasses import dataclass

import pytest

from trackmod.core.notes.codec import decode_note, encode_note
from trackmod.core.notes.command import NoteCommand, NoteValue
from trackmod.core.notes.pitch import Note, PitchClass
from trackmod.spec.pitch import MIDI_OFFSET, NOTE_COUNT


@dataclass(frozen=True)
class NamedNote:
    value: int
    pitch_class: PitchClass
    octave: int
    label: str


NAMED_NOTES = (
    NamedNote(value=0, pitch_class=PitchClass.C, octave=0, label="C-0"),
    NamedNote(value=60, pitch_class=PitchClass.C, octave=5, label="C-5"),
    NamedNote(value=61, pitch_class=PitchClass.C_SHARP, octave=5, label="C#5"),
    NamedNote(value=119, pitch_class=PitchClass.B, octave=9, label="B-9"),
)


@pytest.mark.parametrize("named", NAMED_NOTES, ids=lambda named: named.label)
def test_note_decomposes_into_pitch_class_and_octave(named: NamedNote) -> None:
    note = Note(named.value)
    assert note.value == named.value
    assert note.pitch_class == named.pitch_class
    assert note.octave == named.octave
    assert str(note) == named.label
    assert Note.of(named.pitch_class, named.octave) == note


def test_tracker_octave_sits_one_above_the_midi_octave() -> None:
    # Tracker C-5 is the key trackers print for MIDI 72; MIDI middle C (60) is tracker C-4.
    assert Note.from_midi(72) == Note.of(PitchClass.C, 5)
    assert Note.from_midi(60).octave == 4
    assert Note.of(PitchClass.C, 5).midi == 60 + MIDI_OFFSET


@pytest.mark.parametrize("value", [-1, NOTE_COUNT])
def test_note_rejects_keys_off_the_keyboard(value: int) -> None:
    with pytest.raises(ValueError):
        Note(value)


def test_transposition_stays_on_the_keyboard() -> None:
    assert Note(60).transposed(12) == Note(72)
    with pytest.raises(ValueError):
        Note(119).transposed(1)


@pytest.mark.parametrize("note", [Note(0), Note(60), Note(119), *NoteCommand])
def test_note_column_encoding_round_trips(note: NoteValue) -> None:
    assert decode_note(encode_note(note)) == note


def test_commands_encode_past_the_key_range() -> None:
    # A note column holds either a key or a command in one integer, so the commands must not collide.
    assert all(encode_note(command) >= NOTE_COUNT for command in NoteCommand)
    assert len({encode_note(command) for command in NoteCommand}) == len(NoteCommand)


def test_notes_order_by_pitch() -> None:
    assert sorted([Note(60), Note(0), Note(119)]) == [Note(0), Note(60), Note(119)]


def test_a_note_is_usable_as_a_key() -> None:
    assert {Note(60): "middle"}[Note(60)] == "middle"
