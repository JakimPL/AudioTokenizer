from enum import StrEnum, unique


@unique
class EnvelopeKind(StrEnum):
    """Which property of a voice an envelope shapes over time."""

    VOLUME = "volume"
    PANNING = "panning"
    PITCH = "pitch"
