from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from trackmod.core.songs.song import Song
from trackmod.it.checks import violations
from trackmod.it.limits import it_limits
from trackmod.it.parser import ModuleReader
from trackmod.it.settings import ITSettings
from trackmod.it.sizing import module_bytes
from trackmod.it.spec.identity import EXTENSION
from trackmod.it.writer import write_module
from trackmod.limits.compliance import Compliance
from trackmod.limits.error import require
from trackmod.limits.table import Limits
from trackmod.limits.violation import Violation
from trackmod.module.size import SizeReport
from trackmod.schema.config import FROZEN


class ITModule(BaseModel):
    """An Impulse Tracker module: a song, the settings this format adds, and how strictly it is held."""

    model_config = FROZEN

    song: Song
    compliance: Compliance
    settings: ITSettings = ITSettings()

    @classmethod
    def from_song(cls, song: Song, *, compliance: Compliance, settings: ITSettings | None = None) -> ITModule:
        """Bind a song to this format at one compliance level."""
        return cls(song=song, compliance=compliance, settings=settings or ITSettings())

    @classmethod
    def parse(cls, data: bytes, *, compliance: Compliance = Compliance.EXTENDED) -> ITModule:
        """Rebuild a module from the bytes of an Impulse Tracker file.

        Raises:
            ValueError: when the data does not open with this format's tag.
        """
        reader = ModuleReader(data)
        return cls(song=reader.song(), compliance=compliance, settings=reader.settings())

    @classmethod
    def load(cls, path: Path, *, compliance: Compliance = Compliance.EXTENDED) -> ITModule:
        """Read a module from an Impulse Tracker file."""
        return cls.parse(path.read_bytes(), compliance=compliance)

    @property
    def extension(self) -> str:
        """The file extension this format is written with."""
        return EXTENSION

    @property
    def limits(self) -> Limits:
        """The bounds this module is held to, at its compliance level."""
        return it_limits(self.compliance)

    def violations(self) -> tuple[Violation, ...]:
        """Every bound the song breaks, empty when the module is writable."""
        return violations(self.song, self.settings, limits=self.limits)

    def size(self) -> SizeReport:
        """How many bytes the module occupies, without serialising it."""
        return module_bytes(self.song)

    def to_bytes(self) -> bytes:
        """Serialise the whole module.

        Raises:
            LimitError: when the song carries values this format refuses at its compliance level.
        """
        require(self.violations())
        return write_module(self.song, self.settings)

    def save(self, path: Path) -> None:
        """Serialise the module and write it to ``path``."""
        path.write_bytes(self.to_bytes())
