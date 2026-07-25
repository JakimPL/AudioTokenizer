from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.it.spec.sizes import PATTERN_HEADER_BYTES

PATTERN_HEADER: Final = Record(
    size=PATTERN_HEADER_BYTES,
    fields=(
        Field(name="packed_size", offset=0, code="<H"),
        Field(name="rows", offset=2, code="<H"),
        Field(name="reserved", offset=4, code="<I"),
    ),
)
