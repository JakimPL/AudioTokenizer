"""Declarative layout of the Impulse Tracker binary records.

The IT file, sample and instrument headers are fixed-size records whose fields sit at hard-coded byte
offsets (see ITTECH.TXT). The generic layout machinery — :class:`Field`, :class:`ArrayField`,
:class:`Record` and :func:`encode_name` — is shared across formats and lives in
:mod:`audiotokenizer.tracker.records`; this module only supplies the IT-specific record *definitions*
and the IT note-range guard, so the record serializers call :meth:`Record.pack` against them.
"""

from __future__ import annotations

from audiotokenizer.it.spec import (
    CHANNELS_STORED,
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYBOARD_NOTES,
    MAX_IT_NOTE,
    SAMPLE_HEADER_BYTES,
)
from audiotokenizer.tracker.records import ArrayField, Field, Record, encode_name

__all__ = [
    "ArrayField",
    "Field",
    "Record",
    "encode_name",
    "require_it_note",
    "FILE_HEADER",
    "SAMPLE_HEADER",
    "INSTRUMENT_HEADER",
]


def require_it_note(note: int) -> int:
    """Return ``note`` when it is a playable IT key (0..119); raise ``ValueError`` when it is out of range."""
    if not 0 <= note <= MAX_IT_NOTE:
        raise ValueError(f"note {note} is outside the IT key range 0..{MAX_IT_NOTE}")
    return note


FILE_HEADER: Record = Record(
    size=FILE_HEADER_BYTES,
    fields=(
        Field("magic", 0, "4s"),  # "IMPM"
        Field("name", 4, "26s"),
        Field("highlight", 30, "<H"),  # pattern row highlight (unused -> 0)
        Field("order_count", 32, "<H"),
        Field("instrument_count", 34, "<H"),
        Field("sample_count", 36, "<H"),
        Field("pattern_count", 38, "<H"),
        Field("created_with", 40, "<H"),
        Field("compatible_with", 42, "<H"),
        Field("flags", 44, "<H"),
        Field("global_volume", 48, "B"),
        Field("mix_volume", 49, "B"),
        Field("speed", 50, "B"),
        Field("tempo", 51, "B"),
        Field("panning_separation", 52, "B"),
        Field("channel_pan", 64, f"{CHANNELS_STORED}s"),
        Field("channel_volume", 128, f"{CHANNELS_STORED}s"),
    ),
)

SAMPLE_HEADER: Record = Record(
    size=SAMPLE_HEADER_BYTES,
    fields=(
        Field("magic", 0, "4s"),  # "IMPS"
        Field("global_volume", 17, "B"),
        Field("flags", 18, "B"),
        Field("default_volume", 19, "B"),
        Field("name", 20, "26s"),
        Field("convert", 46, "B"),
        Field("length", 48, "<I"),  # frames
        Field("loop_begin", 52, "<I"),
        Field("loop_end", 56, "<I"),
        Field("c5speed", 60, "<I"),
        Field("sample_pointer", 72, "<I"),
    ),
)

INSTRUMENT_HEADER: Record = Record(
    size=INSTRUMENT_HEADER_BYTES,
    fields=(
        Field("magic", 0, "4s"),  # "IMPI"
        Field("new_note_action", 17, "B"),
        Field("pitch_pan_center", 23, "B"),
        Field("global_volume", 24, "B"),
        Field("default_pan", 25, "B"),
        Field("name", 32, "26s"),
    ),
    # 120 (play_note, sample_number) byte-pairs from offset 0x40; envelopes past it stay zero (disabled).
    arrays=(ArrayField("note_map", 64, KEYBOARD_NOTES, "BB"),),
)
