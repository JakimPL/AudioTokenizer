from pathlib import Path
from typing import Protocol

from trackmod.core.songs.song import Song
from trackmod.limits.table import Limits
from trackmod.limits.violation import Violation
from trackmod.module.size import SizeReport


class TrackerModule(Protocol):
    """The surface every format binding offers, so callers can hold a module without naming its format."""

    @property
    def song(self) -> Song:
        """The format-agnostic content this module writes."""

    @property
    def limits(self) -> Limits:
        """The bounds this module is held to, at its compliance level."""

    @property
    def extension(self) -> str:
        """The file extension this format is written with, including the leading dot."""

    def violations(self) -> tuple[Violation, ...]:
        """Every bound the song breaks, empty when the module is writable."""

    def size(self) -> SizeReport:
        """How many bytes the module occupies, without serialising it."""

    def to_bytes(self) -> bytes:
        """Serialise the whole module."""

    def save(self, path: Path) -> None:
        """Serialise the module and write it to ``path``."""
