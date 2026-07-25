from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.xm.spec.sizes import PATTERN_HEADER_BYTES

PATTERN_HEADER: Final = Record(
    size=PATTERN_HEADER_BYTES,
    fields=(
        Field(name="header_length", offset=0, code="<I"),
        Field(name="packing_type", offset=4, code="B"),
        Field(name="rows", offset=5, code="<H"),
        Field(name="packed_size", offset=7, code="<H"),
    ),
)
