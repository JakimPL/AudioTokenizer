from typing import Final

from trackmod.binary.records.field import ArrayField, Field
from trackmod.binary.records.record import Record
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.it.layout.envelope import envelope_fields, envelope_nodes
from trackmod.it.spec.sizes import FILENAME_BYTES, INSTRUMENT_HEADER_BYTES, NAME_BYTES
from trackmod.spec.pitch import NOTE_COUNT

NOTE_MAP_CODE: Final = "BB"

INSTRUMENT_HEADER: Final = Record(
    size=INSTRUMENT_HEADER_BYTES,
    fields=(
        Field(name="magic", offset=0, code="4s"),
        Field(name="filename", offset=4, code=f"{FILENAME_BYTES}s"),
        Field(name="new_note_action", offset=17, code="B"),
        Field(name="duplicate_check", offset=18, code="B"),
        Field(name="duplicate_action", offset=19, code="B"),
        Field(name="fadeout", offset=20, code="<H"),
        Field(name="pitch_pan_separation", offset=22, code="b"),
        Field(name="pitch_pan_center", offset=23, code="B"),
        Field(name="global_volume", offset=24, code="B"),
        Field(name="default_pan", offset=25, code="B"),
        Field(name="random_volume", offset=26, code="B"),
        Field(name="random_panning", offset=27, code="B"),
        Field(name="tracker_version", offset=28, code="<H"),
        Field(name="sample_count", offset=30, code="B"),
        Field(name="name", offset=32, code=f"{NAME_BYTES}s"),
        Field(name="filter_cutoff", offset=58, code="B"),
        Field(name="filter_resonance", offset=59, code="B"),
        Field(name="midi_channel", offset=60, code="B"),
        Field(name="midi_program", offset=61, code="B"),
        Field(name="midi_bank", offset=62, code="<H"),
        *envelope_fields(EnvelopeKind.VOLUME),
        *envelope_fields(EnvelopeKind.PANNING),
        *envelope_fields(EnvelopeKind.PITCH),
    ),
    arrays=(
        ArrayField(name="note_map", offset=64, count=NOTE_COUNT, code=NOTE_MAP_CODE),
        envelope_nodes(EnvelopeKind.VOLUME),
        envelope_nodes(EnvelopeKind.PANNING),
        envelope_nodes(EnvelopeKind.PITCH),
    ),
)
