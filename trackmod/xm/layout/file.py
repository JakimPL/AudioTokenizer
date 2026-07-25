from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.xm.spec.identity import MAGIC_BYTES
from trackmod.xm.spec.sizes import FILE_HEADER_BYTES, MODULE_NAME_BYTES, TRACKER_NAME_BYTES

FILE_HEADER: Final = Record(
    size=FILE_HEADER_BYTES,
    fields=(
        Field(name="magic", offset=0, code=f"{MAGIC_BYTES}s"),
        Field(name="name", offset=17, code=f"{MODULE_NAME_BYTES}s"),
        Field(name="stripped", offset=37, code="B"),
        Field(name="tracker", offset=38, code=f"{TRACKER_NAME_BYTES}s"),
        Field(name="version", offset=58, code="<H"),
        Field(name="header_size", offset=60, code="<I"),
        Field(name="order_count", offset=64, code="<H"),
        Field(name="restart_position", offset=66, code="<H"),
        Field(name="channels", offset=68, code="<H"),
        Field(name="pattern_count", offset=70, code="<H"),
        Field(name="instrument_count", offset=72, code="<H"),
        Field(name="flags", offset=74, code="<H"),
        Field(name="speed", offset=76, code="<H"),
        Field(name="tempo", offset=78, code="<H"),
    ),
)
