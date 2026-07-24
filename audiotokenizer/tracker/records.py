"""Declarative layout of fixed-size binary records, shared by every tracker format writer.

A module file (``.IT``, ``.XM``, …) is a run of fixed-size records whose fields sit at hard-coded byte
offsets. Describing that layout as *data* — an ordered list of :class:`Field`/:class:`ArrayField` specs
per :class:`Record` — keeps the on-disk structure explicit and in one place, so the record serializers
only supply field *values* and call :meth:`Record.pack`, never touching raw offsets. The low-level field
encoders shared across those serializers live here too, next to the layout machinery they serve. The
concrete record *definitions* stay in each format's ``format`` module; only this format-agnostic
machinery is shared.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# A single field carries one struct value (an int, or a pre-padded byte block); an array field carries
# a row per element (e.g. a keyboard note map's (play_note, sample) pairs). ``pack`` accepts both in one map.
FieldValue = int | bytes
ArrayValue = Sequence[Sequence[int]]
RecordValues = Mapping[str, FieldValue | ArrayValue]


@dataclass(frozen=True)
class Field:
    """One fixed field of a record: a :mod:`struct` value written at a byte ``offset``.

    ``code`` is a ``struct`` format for a single value, e.g. ``"<I"`` (u32), ``"<H"`` (u16), ``"B"``
    (byte) or ``"26s"`` (a fixed-length byte block, already padded/truncated by the caller).
    """

    name: str
    offset: int
    code: str


@dataclass(frozen=True)
class ArrayField:
    """A contiguous run of ``count`` fixed-stride elements (e.g. a keyboard note map).

    ``code`` is the ``struct`` format for one element; its packed size is the stride. Each supplied
    value is a tuple unpacked into that element (``("BB", (note, sample))`` -> two bytes per key).
    """

    name: str
    offset: int
    count: int
    code: str


@dataclass(frozen=True)
class Record:
    """A fixed-size record: a byte ``size`` plus the fields and arrays laid out within it."""

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


def require_note(note: int, *, low: int = 0, high: int) -> int:
    """Return ``note`` when it lies in ``low..high`` inclusive; raise ``ValueError`` when it is out of range."""
    if not low <= note <= high:
        raise ValueError(f"note {note} is outside the key range {low}..{high}")
    return note
