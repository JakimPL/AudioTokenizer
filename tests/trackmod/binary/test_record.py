from __future__ import annotations

import pytest

from trackmod.binary.records.field import ArrayField, Field
from trackmod.binary.records.record import Record

HEADER = Record(
    size=16,
    fields=(
        Field(name="magic", offset=0, code="4s"),
        Field(name="count", offset=4, code="<H"),
        Field(name="flags", offset=6, code="B"),
        Field(name="offset", offset=8, code="<I"),
        Field(name="trim", offset=12, code="b"),
    ),
)

KEYMAP = Record(
    size=8,
    fields=(Field(name="kind", offset=0, code="B"),),
    arrays=(ArrayField(name="keys", offset=2, count=3, code="BB"),),
)

VALUES = {"magic": b"IMPM", "count": 400, "flags": 0x0C, "offset": 0xDEADBEEF, "trim": -3}


def test_a_packed_record_reads_back_field_for_field() -> None:
    assert HEADER.unpack(HEADER.pack(VALUES)) == VALUES


def test_a_record_packs_to_exactly_its_declared_size() -> None:
    assert len(HEADER.pack(VALUES)) == HEADER.size


def test_undescribed_offsets_stay_zero() -> None:
    # Bytes 13..15 are reserved: no field describes them, so packing must leave them clear.
    assert HEADER.pack(VALUES)[13:] == bytes(3)


def test_array_elements_round_trip_at_their_stride() -> None:
    values = {"kind": 1, "keys": ((10, 1), (20, 2), (30, 3))}
    packed = KEYMAP.pack(values)
    assert packed[2:8] == bytes([10, 1, 20, 2, 30, 3])
    assert KEYMAP.unpack(packed) == values


def test_unpacking_a_short_buffer_raises() -> None:
    with pytest.raises(ValueError):
        HEADER.unpack(HEADER.pack(VALUES)[:-1])


def test_a_field_past_the_end_of_the_record_is_rejected() -> None:
    with pytest.raises(ValueError):
        Record(size=4, fields=(Field(name="word", offset=2, code="<I"),))


def test_an_array_past_the_end_of_the_record_is_rejected() -> None:
    with pytest.raises(ValueError):
        Record(size=4, fields=(), arrays=(ArrayField(name="keys", offset=0, count=3, code="BB"),))


def test_a_negative_offset_is_rejected() -> None:
    with pytest.raises(ValueError):
        Field(name="word", offset=-1, code="<I")


def test_a_missing_value_raises() -> None:
    with pytest.raises(KeyError):
        HEADER.pack({key: value for key, value in VALUES.items() if key != "flags"})


def test_an_array_given_a_scalar_raises() -> None:
    with pytest.raises(TypeError):
        KEYMAP.pack({"kind": 1, "keys": 3})


def test_field_and_array_sizes_follow_their_struct_codes() -> None:
    assert Field(name="word", offset=0, code="<I").size == 4
    assert ArrayField(name="keys", offset=0, count=3, code="BB").stride == 2
    assert ArrayField(name="keys", offset=0, count=3, code="BB").size == 6
