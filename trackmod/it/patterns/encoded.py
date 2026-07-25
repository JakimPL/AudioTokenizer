from dataclasses import dataclass, field


@dataclass
class EncodedCell:
    """One cell's mask byte and the column bytes that follow it."""

    mask: int = 0
    payload: list[int] = field(default_factory=list)
