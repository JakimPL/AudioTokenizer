from pydantic import BaseModel

from trackmod.core.effects.effect import Effect
from trackmod.core.notes.command import NoteValue
from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Index, Volume


class Cell(BaseModel):
    """One position in a pattern grid: the four columns a tracker reads for a channel on a row.

    Every column is independently present or absent, which is what a tracker stores — a key-off carries a
    note and nothing else, a mid-pattern tempo change carries only an effect.
    """

    model_config = FROZEN

    note: NoteValue | None = None
    instrument: Index | None = None
    volume: Volume | None = None
    effect: Effect | None = None

    @property
    def is_empty(self) -> bool:
        """Whether every column is absent, so a tracker plays nothing new here."""
        return self.note is None and self.instrument is None and self.volume is None and self.effect is None
