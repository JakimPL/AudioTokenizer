"""Declarative layout of the FastTracker 2 binary records.

The XM file, pattern, instrument and sample headers are fixed-size records whose fields sit at hard-coded
byte offsets. The generic layout machinery — :class:`Field`, :class:`Record`, :func:`encode_name` — is
shared across formats and lives in :mod:`audiotokenizer.tracker.records`; this module only supplies the
XM-specific record *definitions* (mirroring :mod:`audiotokenizer.it.format`) and the XM note-range guard,
so the record serializers call :meth:`Record.pack` against them.

Signed fields use lower-case struct codes: a sample's ``finetune`` and ``relative_note`` are ``"b"``
(signed byte). Reserved regions carry no field, so :meth:`Record.pack` leaves them zero — the instrument
header's 22 trailing bytes and the sample header's one reserved byte included.
"""

from __future__ import annotations

from audiotokenizer.tracker.records import Field, Record, encode_name
from audiotokenizer.xm.spec import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYMAP_NOTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

__all__ = [
    "Field",
    "Record",
    "encode_name",
    "require_xm_note",
    "FILE_HEADER",
    "PATTERN_HEADER",
    "INSTRUMENT_HEADER",
    "SAMPLE_HEADER",
]

_MAX_XM_NOTE = KEYMAP_NOTES  # notes are 1-based; 1..96 are playable keys, 97 is key-off


def require_xm_note(note: int) -> int:
    """Return ``note`` when it is a playable XM key (1..96); raise ``ValueError`` when it is out of range."""
    if not 1 <= note <= _MAX_XM_NOTE:
        raise ValueError(f"note {note} is outside the XM key range 1..{_MAX_XM_NOTE}")
    return note


# The fixed 80-byte header prefix (magic .. bpm); module.py appends the 256-byte order table after it.
FILE_HEADER: Record = Record(
    size=FILE_HEADER_BYTES,
    fields=(
        Field("magic", 0, "17s"),  # "Extended Module: "
        Field("name", 17, "20s"),
        Field("stripped", 37, "B"),  # 0x1A
        Field("tracker", 38, "20s"),
        Field("version", 58, "<H"),  # 0x0104
        Field("header_size", 60, "<I"),  # 276 = 20 fixed bytes + 256 order bytes
        Field("song_length", 64, "<H"),  # order-table entries in use
        Field("restart_position", 66, "<H"),
        Field("channels", 68, "<H"),
        Field("pattern_count", 70, "<H"),
        Field("instrument_count", 72, "<H"),
        Field("flags", 74, "<H"),  # bit0: linear frequency table
        Field("speed", 76, "<H"),  # ticks per row
        Field("bpm", 78, "<H"),  # the 16-bit tempo word — the FastTracker 2 hack
    ),
)

PATTERN_HEADER: Record = Record(
    size=PATTERN_HEADER_BYTES,
    fields=(
        Field("header_length", 0, "<I"),  # 9
        Field("packing_type", 4, "B"),  # 0
        Field("rows", 5, "<H"),
        Field("packed_size", 7, "<H"),  # bytes of packed cell stream that follow
    ),
)

# The extended instrument header written when the instrument owns samples (always, here). The 96-byte
# keymap is passed pre-built as one block; the volume/panning envelopes and every type/point field stay
# zero (envelopes disabled), so a one-shot sample plays at its sample volume and stops when it ends.
INSTRUMENT_HEADER: Record = Record(
    size=INSTRUMENT_HEADER_BYTES,
    fields=(
        Field("header_size", 0, "<I"),  # 263
        Field("name", 4, "22s"),
        Field("type", 26, "B"),  # 0
        Field("sample_count", 27, "<H"),
        Field("sample_header_size", 29, "<I"),  # 40
        Field("keymap", 33, f"{KEYMAP_NOTES}s"),  # 96 bytes: key -> 1-based sample within the instrument
        # offsets 129..263 (envelopes, points, types, vibrato, fadeout, reserved) stay zero = disabled.
    ),
)

SAMPLE_HEADER: Record = Record(
    size=SAMPLE_HEADER_BYTES,
    fields=(
        Field("length", 0, "<I"),  # bytes of PCM (frames * bytes-per-frame)
        Field("loop_start", 4, "<I"),  # bytes
        Field("loop_length", 8, "<I"),  # bytes
        Field("volume", 12, "B"),  # 0..64, the atom's static gain
        Field("finetune", 13, "b"),  # signed
        Field("type", 14, "B"),  # bit4 = 16-bit, bits0-1 = loop mode
        Field("panning", 15, "B"),  # 0..255
        Field("relative_note", 16, "b"),  # signed; re-tunes the key to native rate
        Field("reserved", 17, "B"),  # 0
        Field("name", 18, "22s"),
    ),
)
