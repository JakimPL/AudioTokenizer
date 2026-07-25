from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.xm.spec.sizes import NAME_BYTES, SAMPLE_HEADER_BYTES

SAMPLE_HEADER: Final = Record(
    size=SAMPLE_HEADER_BYTES,
    fields=(
        Field(name="length", offset=0, code="<I"),
        Field(name="loop_begin", offset=4, code="<I"),
        Field(name="loop_length", offset=8, code="<I"),
        Field(name="volume", offset=12, code="B"),
        Field(name="finetune", offset=13, code="b"),
        Field(name="type", offset=14, code="B"),
        Field(name="panning", offset=15, code="B"),
        Field(name="relative_note", offset=16, code="b"),
        Field(name="reserved", offset=17, code="B"),
        Field(name="name", offset=18, code=f"{NAME_BYTES}s"),
    ),
)
