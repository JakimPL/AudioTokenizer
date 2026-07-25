from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.core.instruments.instrument import Instrument
from trackmod.core.patterns.grid import Pattern
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Channels


class Song(BaseModel):
    """A complete piece of tracker music, held in the terms every tracker format shares.

    Patterns, instruments and samples are flat lists that the columns and keymaps index into, so a song
    is self-contained: an instrument names a sample by its position here, and an order names a pattern
    the same way. Every pattern is the same width, which is the contract formats that store one channel
    count for the whole module need.
    """

    model_config = FROZEN

    name: str
    channels: Channels
    patterns: tuple[Pattern, ...]
    order: OrderList
    instruments: tuple[Instrument, ...]
    samples: tuple[Sample, ...]
    playback: Playback

    @model_validator(mode="after")
    def _references_resolve(self) -> Song:
        for index, pattern in enumerate(self.patterns):
            if pattern.channels != self.channels:
                raise ValueError(
                    f"pattern {index} is {pattern.channels} channels wide, but the song declares {self.channels}"
                )

        for position, entry in enumerate(self.order.entries):
            if entry >= len(self.patterns):
                raise ValueError(f"order position {position} names pattern {entry} of {len(self.patterns)}")

        for index, instrument in enumerate(self.instruments):
            for sample in instrument.samples:
                if sample >= len(self.samples):
                    raise ValueError(f"instrument {index} names sample {sample} of {len(self.samples)}")

        return self

    @property
    def rows(self) -> int:
        """How many rows the song plays in total, following its order list."""
        return sum(self.patterns[entry].rows for entry in self.order.entries)
