from __future__ import annotations

from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.it.layout.envelope import ENVELOPE_OFFSETS, envelope_field
from trackmod.it.layout.file import FILE_HEADER
from trackmod.it.layout.instrument import INSTRUMENT_HEADER
from trackmod.it.layout.pattern import PATTERN_HEADER
from trackmod.it.layout.sample import SAMPLE_HEADER
from trackmod.it.spec.sizes import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

RECORDS = (
    (FILE_HEADER, FILE_HEADER_BYTES),
    (PATTERN_HEADER, PATTERN_HEADER_BYTES),
    (SAMPLE_HEADER, SAMPLE_HEADER_BYTES),
    (INSTRUMENT_HEADER, INSTRUMENT_HEADER_BYTES),
)


def test_record_sizes_match_the_format() -> None:
    assert all(record.size == size for record, size in RECORDS)


def test_no_two_fields_of_a_record_overlap() -> None:
    for record, _ in RECORDS:
        occupied: set[int] = set()
        for field in record.fields:
            span = set(range(field.offset, field.offset + field.size))
            assert not span & occupied
            occupied |= span


def test_each_envelope_block_is_named_and_placed_apart() -> None:
    offsets = {kind: ENVELOPE_OFFSETS[kind] for kind in EnvelopeKind}
    assert len(set(offsets.values())) == len(EnvelopeKind)
    names = {envelope_field(kind, "flags") for kind in EnvelopeKind}
    assert names <= {field.name for field in INSTRUMENT_HEADER.fields}
