"""Declarative layout of the Impulse Tracker binary records.

The IT file, sample and instrument headers are fixed-size records whose fields sit at hard-coded byte
offsets (see ITTECH.TXT). Describing that layout as *data* — an ordered list of
:class:`Field`/:class:`ArrayField` specs per :class:`ITRecord` — keeps the on-disk structure explicit
and in one place, so the record serializers only supply field *values* and call :meth:`ITRecord.pack`,
never touching raw offsets. The two low-level field encoders shared across those serializers live here
too, next to the layout they serve.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from audiotokenizer.it.spec import (
    CHANNELS_STORED,
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYBOARD_NOTES,
    MAX_IT_NOTE,
    SAMPLE_HEADER_BYTES,
)

# A single field carries one struct value (an int, or a pre-padded byte block); an array field carries
# a row per element (the keyboard note map's (play_note, sample) pairs). ``pack`` accepts both in one map.
FieldValue = int | bytes
ArrayValue = Sequence[Sequence[int]]
RecordValues = Mapping[str, FieldValue | ArrayValue]


@dataclass(frozen=True)
class Field:
    """One fixed field of an IT record: a :mod:`struct` value written at a byte ``offset``.

    ``code`` is a ``struct`` format for a single value, e.g. ``"<I"`` (u32), ``"<H"`` (u16), ``"B"``
    (byte) or ``"26s"`` (a fixed-length byte block, already padded/truncated by the caller).
    """

    name: str
    offset: int
    code: str


@dataclass(frozen=True)
class ArrayField:
    """A contiguous run of ``count`` fixed-stride elements (e.g. the 120-entry keyboard note map).

    ``code`` is the ``struct`` format for one element; its packed size is the stride. Each supplied
    value is a tuple unpacked into that element (``("BB", (note, sample))`` -> two bytes per key).
    """

    name: str
    offset: int
    count: int
    code: str


@dataclass(frozen=True)
class ITRecord:
    """A fixed-size IT record: a byte ``size`` plus the fields and arrays laid out within it."""

    size: int
    fields: tuple[Field, ...]
    arrays: tuple[ArrayField, ...] = ()

    def pack(self, values: RecordValues) -> bytes:
        """Serialize ``values`` into ``size`` bytes; unwritten offsets (reserved regions) stay zero."""
        buffer = bytearray(self.size)
        for spec in self.fields:
            struct.pack_into(spec.code, buffer, spec.offset, values[spec.name])
        for array in self.arrays:
            stride = struct.calcsize(array.code)
            rows = values[array.name]
            assert not isinstance(rows, (int, bytes))  # array fields always carry a Sequence of element rows
            for index, row in enumerate(rows):
                struct.pack_into(array.code, buffer, array.offset + index * stride, *row)
        return bytes(buffer)


def encode_name(text: str, length: int) -> bytes:
    """Encode ``text`` to exactly ``length`` bytes, ASCII, null-padded (over-long names truncated)."""
    raw = text.encode("ascii", errors="replace")[:length]
    return raw + bytes(length - len(raw))


def require_it_note(note: int) -> int:
    """Return ``note`` when it is a playable IT key (0..119); raise ``ValueError`` when it is out of range."""
    if not 0 <= note <= MAX_IT_NOTE:
        raise ValueError(f"note {note} is outside the IT key range 0..{MAX_IT_NOTE}")
    return note


FILE_HEADER: ITRecord = ITRecord(
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

SAMPLE_HEADER: ITRecord = ITRecord(
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

INSTRUMENT_HEADER: ITRecord = ITRecord(
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
