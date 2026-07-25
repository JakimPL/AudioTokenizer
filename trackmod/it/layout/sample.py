from typing import Final

from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record
from trackmod.it.spec.sizes import FILENAME_BYTES, NAME_BYTES, SAMPLE_HEADER_BYTES

SAMPLE_HEADER: Final = Record(
    size=SAMPLE_HEADER_BYTES,
    fields=(
        Field(name="magic", offset=0, code="4s"),
        Field(name="filename", offset=4, code=f"{FILENAME_BYTES}s"),
        Field(name="global_volume", offset=17, code="B"),
        Field(name="flags", offset=18, code="B"),
        Field(name="default_volume", offset=19, code="B"),
        Field(name="name", offset=20, code=f"{NAME_BYTES}s"),
        Field(name="convert", offset=46, code="B"),
        Field(name="default_pan", offset=47, code="B"),
        Field(name="length", offset=48, code="<I"),
        Field(name="loop_begin", offset=52, code="<I"),
        Field(name="loop_end", offset=56, code="<I"),
        Field(name="c5speed", offset=60, code="<I"),
        Field(name="sustain_begin", offset=64, code="<I"),
        Field(name="sustain_end", offset=68, code="<I"),
        Field(name="sample_pointer", offset=72, code="<I"),
        Field(name="vibrato_speed", offset=76, code="B"),
        Field(name="vibrato_depth", offset=77, code="B"),
        Field(name="vibrato_rate", offset=78, code="B"),
        Field(name="vibrato_waveform", offset=79, code="B"),
    ),
)
