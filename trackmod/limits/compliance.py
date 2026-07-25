from enum import StrEnum, unique


@unique
class Compliance(StrEnum):
    """How strictly a module holds to the tracker its format was designed for."""

    CANONICAL = "canonical"
    EXTENDED = "extended"
