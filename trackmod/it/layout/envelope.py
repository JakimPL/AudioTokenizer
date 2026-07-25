from typing import Final

from trackmod.binary.records.field import ArrayField, Field
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.it.spec.sizes import (
    ENVELOPE_NODES,
    PANNING_ENVELOPE_OFFSET,
    PITCH_ENVELOPE_OFFSET,
    VOLUME_ENVELOPE_OFFSET,
)

ENVELOPE_OFFSETS: Final[dict[EnvelopeKind, int]] = {
    EnvelopeKind.VOLUME: VOLUME_ENVELOPE_OFFSET,
    EnvelopeKind.PANNING: PANNING_ENVELOPE_OFFSET,
    EnvelopeKind.PITCH: PITCH_ENVELOPE_OFFSET,
}

NODE_CODE: Final = "<bH"
NODES_OFFSET: Final = 6


def envelope_field(kind: EnvelopeKind, name: str) -> str:
    """The record field name one envelope's property is stored under."""
    return f"{kind}_{name}"


def envelope_fields(kind: EnvelopeKind) -> tuple[Field, ...]:
    """The six header bytes that open one envelope block."""
    base = ENVELOPE_OFFSETS[kind]
    return (
        Field(name=envelope_field(kind, "flags"), offset=base + 0, code="B"),
        Field(name=envelope_field(kind, "count"), offset=base + 1, code="B"),
        Field(name=envelope_field(kind, "loop_begin"), offset=base + 2, code="B"),
        Field(name=envelope_field(kind, "loop_end"), offset=base + 3, code="B"),
        Field(name=envelope_field(kind, "sustain_begin"), offset=base + 4, code="B"),
        Field(name=envelope_field(kind, "sustain_end"), offset=base + 5, code="B"),
    )


def envelope_nodes(kind: EnvelopeKind) -> ArrayField:
    """The node table of one envelope: a signed value byte then a little-endian tick word, unaligned."""
    return ArrayField(
        name=envelope_field(kind, "nodes"),
        offset=ENVELOPE_OFFSETS[kind] + NODES_OFFSET,
        count=ENVELOPE_NODES,
        code=NODE_CODE,
    )
