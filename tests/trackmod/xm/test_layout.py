from __future__ import annotations

from trackmod.binary.records.record import Record
from trackmod.xm.layout.envelope import ENVELOPE_KINDS, POINTS_OFFSETS, envelope_field
from trackmod.xm.layout.file import FILE_HEADER
from trackmod.xm.layout.instrument import EMPTY_INSTRUMENT_HEADER, INSTRUMENT_HEADER
from trackmod.xm.layout.pattern import PATTERN_HEADER
from trackmod.xm.layout.sample import SAMPLE_HEADER
from trackmod.xm.spec.identity import MAGIC, MAGIC_BYTES
from trackmod.xm.spec.sizes import (
    EMPTY_INSTRUMENT_HEADER_BYTES,
    FILE_HEADER_BYTES,
    HEADER_SIZE_FIELD,
    HEADER_SIZE_OFFSET,
    INSTRUMENT_HEADER_BYTES,
    ORDER_TABLE_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

RECORDS = (
    (FILE_HEADER, FILE_HEADER_BYTES),
    (PATTERN_HEADER, PATTERN_HEADER_BYTES),
    (SAMPLE_HEADER, SAMPLE_HEADER_BYTES),
    (INSTRUMENT_HEADER, INSTRUMENT_HEADER_BYTES),
    (EMPTY_INSTRUMENT_HEADER, EMPTY_INSTRUMENT_HEADER_BYTES),
)


def occupied_bytes(record: Record) -> set[int]:
    spans = [range(field.offset, field.offset + field.size) for field in record.fields]
    spans += [range(array.offset, array.offset + array.size) for array in record.arrays]
    occupied: set[int] = set()
    for span in spans:
        assert not set(span) & occupied
        occupied |= set(span)

    return occupied


def test_record_sizes_match_the_format() -> None:
    assert all(record.size == size for record, size in RECORDS)


def test_no_two_fields_of_a_record_overlap() -> None:
    for record, _ in RECORDS:
        occupied_bytes(record)


def test_the_declared_header_size_covers_the_order_table_and_what_precedes_it() -> None:
    # The field counts from where it sits, so the first pattern begins exactly this far into the file.
    assert HEADER_SIZE_OFFSET + HEADER_SIZE_FIELD == FILE_HEADER_BYTES + ORDER_TABLE_BYTES


def test_the_module_tag_fills_the_field_that_carries_it() -> None:
    assert len(MAGIC) == MAGIC_BYTES


def test_the_short_instrument_header_is_the_opening_of_the_long_one() -> None:
    short = {(field.name, field.offset, field.code) for field in EMPTY_INSTRUMENT_HEADER.fields}
    long = {(field.name, field.offset, field.code) for field in INSTRUMENT_HEADER.fields}
    assert short <= long


def test_each_envelope_has_its_own_point_table() -> None:
    assert len({POINTS_OFFSETS[kind] for kind in ENVELOPE_KINDS}) == len(ENVELOPE_KINDS)
    names = {envelope_field(kind, "flags") for kind in ENVELOPE_KINDS}
    assert names <= {field.name for field in INSTRUMENT_HEADER.fields}
