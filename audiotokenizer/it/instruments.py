"""The instrument record: its data shape, the keyboard note map, and the 554-byte IMPI header.

Every atom is its own stored sample played at unity rate, so an instrument here is a pure router: each
keyboard key maps to one sample and sounds it at C-5 (the sample's own C5Speed, 44100 Hz), which is why
:func:`fixed_c5_note_map` fixes every ``play_note`` to :data:`IT_C5_NOTE`. The pattern-cell note byte
selects the atom; pitch stays at unity for all of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from audiotokenizer.it.format import INSTRUMENT_HEADER, encode_name, require_it_note
from audiotokenizer.it.spec import (
    IT_C5_NOTE,
    KEYBOARD_NOTES,
    MAGIC_INSTRUMENT,
    MAX_GLOBAL_VOLUME,
    NAME_BYTES,
    NOTE_ACTION_CUT,
    PAN_CENTER,
)


@dataclass(frozen=True)
class ITInstrument:
    """One instrument: a 120-entry ``(play_note, sample_number)`` keyboard map plus playback defaults."""

    name: str
    note_map: tuple[tuple[int, int], ...]
    global_volume: int = MAX_GLOBAL_VOLUME
    default_pan: int = PAN_CENTER
    new_note_action: int = NOTE_ACTION_CUT


def fixed_c5_note_map(assignments: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Build a 120-entry note map where key ``note`` plays ``assignments[note]`` at C-5 (unity rate).

    Sample numbers are 1-based (``0`` = no sample). Every key's ``play_note`` is :data:`IT_C5_NOTE`, so
    each mapped sample sounds at its own C5Speed (44100 Hz) regardless of which key triggers it — the
    note byte carries which atom to play, not a pitch.
    """
    for note, sample_number in assignments.items():
        require_it_note(note)
        if sample_number < 0:
            raise ValueError(f"sample number {sample_number} must be non-negative")
    return tuple((IT_C5_NOTE, assignments.get(note, 0)) for note in range(KEYBOARD_NOTES))


def instrument_header(instrument: ITInstrument) -> bytes:
    """Serialize a 554-byte IMPI instrument header with disabled envelopes."""
    if len(instrument.note_map) != KEYBOARD_NOTES:
        raise ValueError(f"note map must have {KEYBOARD_NOTES} entries, got {len(instrument.note_map)}")
    # Envelopes (offsets 304..550) and the 4 trailing reserved bytes stay zero = disabled.
    return INSTRUMENT_HEADER.pack(
        {
            "magic": MAGIC_INSTRUMENT,
            "new_note_action": instrument.new_note_action & 0xFF,
            "pitch_pan_center": IT_C5_NOTE,
            "global_volume": min(instrument.global_volume, MAX_GLOBAL_VOLUME),
            "default_pan": instrument.default_pan & 0xFF,
            "name": encode_name(instrument.name, NAME_BYTES),
            "note_map": tuple(
                (play_note & 0xFF, sample_number & 0xFF) for play_note, sample_number in instrument.note_map
            ),
        }
    )
