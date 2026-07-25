from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.xm.layout.envelope import ENVELOPE_KINDS, envelope_fields, envelope_points
from trackmod.xm.spec.sizes import (
    EMPTY_INSTRUMENT_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYMAP_NOTES,
    NAME_BYTES,
)

IDENTITY_FIELDS: Final = (
    Field(name="header_size", offset=0, code="<I"),
    Field(name="name", offset=4, code=f"{NAME_BYTES}s"),
    Field(name="type", offset=26, code="B"),
    Field(name="sample_count", offset=27, code="<H"),
)

EMPTY_INSTRUMENT_HEADER: Final = Record(size=EMPTY_INSTRUMENT_HEADER_BYTES, fields=IDENTITY_FIELDS)

INSTRUMENT_HEADER: Final = Record(
    size=INSTRUMENT_HEADER_BYTES,
    fields=(
        *IDENTITY_FIELDS,
        Field(name="sample_header_size", offset=29, code="<I"),
        Field(name="keymap", offset=33, code=f"{KEYMAP_NOTES}s"),
        *(field for kind in ENVELOPE_KINDS for field in envelope_fields(kind)),
        Field(name="vibrato_type", offset=235, code="B"),
        Field(name="vibrato_sweep", offset=236, code="B"),
        Field(name="vibrato_depth", offset=237, code="B"),
        Field(name="vibrato_rate", offset=238, code="B"),
        Field(name="fadeout", offset=239, code="<H"),
        Field(name="reserved", offset=241, code="<H"),
    ),
    arrays=tuple(envelope_points(kind) for kind in ENVELOPE_KINDS),
)
