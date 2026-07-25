from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Index


class OrderList(BaseModel):
    """The sequence of patterns a song plays, and where it resumes after the last one.

    Entries are indices into a song's patterns. Formats that mark separators or an end of song in the
    same table write those markers themselves, so this list holds playable positions only. The entries
    are read through :attr:`entries` rather than by iterating the list, because a model iterates its own
    fields.
    """

    model_config = FROZEN

    entries: tuple[Index, ...]
    restart: Index = 0

    @model_validator(mode="after")
    def _restart_lands(self) -> OrderList:
        if self.restart > max(len(self.entries) - 1, 0):
            raise ValueError(f"restart position {self.restart} falls outside {len(self.entries)} orders")

        return self

    @classmethod
    def sequential(cls, count: int) -> OrderList:
        """An order list that plays patterns ``0..count - 1`` once, in order."""
        return cls(entries=tuple(range(count)))

    @property
    def length(self) -> int:
        """How many positions the song plays through."""
        return len(self.entries)
