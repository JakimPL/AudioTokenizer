from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.it.spec.sizes import CHANNELS_STORED, FILE_HEADER_BYTES, NAME_BYTES

FILE_HEADER: Final = Record(
    size=FILE_HEADER_BYTES,
    fields=(
        Field(name="magic", offset=0, code="4s"),
        Field(name="name", offset=4, code=f"{NAME_BYTES}s"),
        Field(name="highlight", offset=30, code="<H"),
        Field(name="order_count", offset=32, code="<H"),
        Field(name="instrument_count", offset=34, code="<H"),
        Field(name="sample_count", offset=36, code="<H"),
        Field(name="pattern_count", offset=38, code="<H"),
        Field(name="created_with", offset=40, code="<H"),
        Field(name="compatible_with", offset=42, code="<H"),
        Field(name="flags", offset=44, code="<H"),
        Field(name="special", offset=46, code="<H"),
        Field(name="global_volume", offset=48, code="B"),
        Field(name="mix_volume", offset=49, code="B"),
        Field(name="speed", offset=50, code="B"),
        Field(name="tempo", offset=51, code="B"),
        Field(name="panning_separation", offset=52, code="B"),
        Field(name="pitch_wheel_depth", offset=53, code="B"),
        Field(name="message_length", offset=54, code="<H"),
        Field(name="message_offset", offset=56, code="<I"),
        Field(name="channel_pan", offset=64, code=f"{CHANNELS_STORED}s"),
        Field(name="channel_volume", offset=128, code=f"{CHANNELS_STORED}s"),
    ),
)
