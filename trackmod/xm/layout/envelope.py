from typing import Final

from trackmod.binary.records.field import ArrayField, Field
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.xm.spec.sizes import (
    ENVELOPE_POINTS,
    PANNING_COUNT_OFFSET,
    PANNING_FLAGS_OFFSET,
    PANNING_POINTS_OFFSET,
    PANNING_SPAN_OFFSET,
    VOLUME_COUNT_OFFSET,
    VOLUME_FLAGS_OFFSET,
    VOLUME_POINTS_OFFSET,
    VOLUME_SPAN_OFFSET,
)

ENVELOPE_KINDS: Final = (EnvelopeKind.VOLUME, EnvelopeKind.PANNING)

POINTS_OFFSETS: Final[dict[EnvelopeKind, int]] = {
    EnvelopeKind.VOLUME: VOLUME_POINTS_OFFSET,
    EnvelopeKind.PANNING: PANNING_POINTS_OFFSET,
}
COUNT_OFFSETS: Final[dict[EnvelopeKind, int]] = {
    EnvelopeKind.VOLUME: VOLUME_COUNT_OFFSET,
    EnvelopeKind.PANNING: PANNING_COUNT_OFFSET,
}
SPAN_OFFSETS: Final[dict[EnvelopeKind, int]] = {
    EnvelopeKind.VOLUME: VOLUME_SPAN_OFFSET,
    EnvelopeKind.PANNING: PANNING_SPAN_OFFSET,
}
FLAGS_OFFSETS: Final[dict[EnvelopeKind, int]] = {
    EnvelopeKind.VOLUME: VOLUME_FLAGS_OFFSET,
    EnvelopeKind.PANNING: PANNING_FLAGS_OFFSET,
}

POINT_CODE: Final = "<HH"


def envelope_field(kind: EnvelopeKind, name: str) -> str:
    """The record field name one envelope's property is stored under."""
    return f"{kind}_{name}"


def envelope_fields(kind: EnvelopeKind) -> tuple[Field, ...]:
    """The five bytes that describe one envelope, which this format scatters across the header."""
    span = SPAN_OFFSETS[kind]
    return (
        Field(name=envelope_field(kind, "count"), offset=COUNT_OFFSETS[kind], code="B"),
        Field(name=envelope_field(kind, "sustain"), offset=span + 0, code="B"),
        Field(name=envelope_field(kind, "loop_begin"), offset=span + 1, code="B"),
        Field(name=envelope_field(kind, "loop_end"), offset=span + 2, code="B"),
        Field(name=envelope_field(kind, "flags"), offset=FLAGS_OFFSETS[kind], code="B"),
    )


def envelope_points(kind: EnvelopeKind) -> ArrayField:
    """The point table of one envelope: a little-endian tick word then a value word."""
    return ArrayField(
        name=envelope_field(kind, "points"),
        offset=POINTS_OFFSETS[kind],
        count=ENVELOPE_POINTS,
        code=POINT_CODE,
    )
