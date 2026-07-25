from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from trackmod.core.songs.song import Song
from trackmod.limits.compliance import Compliance
from trackmod.limits.error import require
from trackmod.limits.table import Limits
from trackmod.limits.violation import Violation
from trackmod.module.size import SizeReport
from trackmod.schema.config import FROZEN
from trackmod.xm.checks import violations
from trackmod.xm.limits import xm_limits
from trackmod.xm.parser import ModuleReader
from trackmod.xm.settings import XMSettings
from trackmod.xm.sizing import module_bytes
from trackmod.xm.spec.identity import EXTENSION
from trackmod.xm.writer import write_module


class XMModule(BaseModel):
    """A FastTracker 2 module: a song, the settings this format adds, and how strictly it is held.

    The format stores no shared sample table — each instrument owns copies of the samples its keys
    reach — so a song whose instruments share samples grows when written, and reading one back gives an
    instrument per group rather than the arrangement it was built from.
    """

    model_config = FROZEN

    song: Song
    compliance: Compliance
    settings: XMSettings = XMSettings()

    @classmethod
    def from_song(cls, song: Song, *, compliance: Compliance, settings: XMSettings | None = None) -> XMModule:
        """Bind a song to this format at one compliance level."""
        return cls(song=song, compliance=compliance, settings=settings or XMSettings())

    @classmethod
    def parse(cls, data: bytes, *, compliance: Compliance = Compliance.EXTENDED) -> XMModule:
        """Rebuild a module from the bytes of a FastTracker 2 file.

        Raises:
            ValueError: when the data does not open with this format's tag.
        """
        reader = ModuleReader(data)
        return cls(song=reader.song(), compliance=compliance, settings=reader.settings())

    @classmethod
    def load(cls, path: Path, *, compliance: Compliance = Compliance.EXTENDED) -> XMModule:
        """Read a module from a FastTracker 2 file."""
        return cls.parse(path.read_bytes(), compliance=compliance)

    @property
    def extension(self) -> str:
        """The file extension this format is written with."""
        return EXTENSION

    @property
    def limits(self) -> Limits:
        """The bounds this module is held to, at its compliance level."""
        return xm_limits(self.compliance)

    def violations(self) -> tuple[Violation, ...]:
        """Every bound the song breaks, empty when the module is writable."""
        return violations(self.song, limits=self.limits)

    def size(self) -> SizeReport:
        """How many bytes the module occupies, without serialising it."""
        return module_bytes(self.song)

    def to_bytes(self) -> bytes:
        """Serialise the whole module.

        A bound this format leaves room for is reported rather than raised, so a caller sees every
        problem at once. Content it has no encoding for at all — a note cut, a sustain loop, a pitch
        envelope, or a keymap that transposes one key of a sample differently from another — is not a
        quantity to bound and raises where it is met.

        Raises:
            LimitError: when the song carries values this format refuses at its compliance level.
            ValueError: when the song carries content this format has no encoding for.
        """
        require(self.violations())
        return write_module(self.song, self.settings)

    def save(self, path: Path) -> None:
        """Serialise the module and write it to ``path``."""
        path.write_bytes(self.to_bytes())
