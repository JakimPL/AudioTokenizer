from dataclasses import dataclass

from trackmod.it.spec.cells import UNSET


@dataclass
class ChannelMemory:
    """What a channel remembers between its present rows, which the reuse mask bits refer to."""

    note: int = UNSET
    instrument: int = UNSET
    volume: int = UNSET
    command: int = UNSET
    parameter: int = UNSET
    mask: int = UNSET
