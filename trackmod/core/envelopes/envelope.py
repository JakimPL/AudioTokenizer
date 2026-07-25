from __future__ import annotations

import pydantic
from pydantic import BaseModel, model_validator

from trackmod.core.envelopes.point import EnvelopePoint
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.schema.config import FROZEN


class Envelope(BaseModel):
    """A breakpoint curve applied to every voice an instrument starts.

    An instrument that carries no envelope of a kind leaves that property alone, so absence rather than
    an enable flag is what switches an envelope off.
    """

    model_config = FROZEN

    points: tuple[EnvelopePoint, ...] = pydantic.Field(min_length=1)
    loop: EnvelopeSpan | None = None
    sustain: EnvelopeSpan | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Envelope:
        ticks = [point.tick for point in self.points]
        if any(later <= earlier for earlier, later in zip(ticks, ticks[1:])):
            raise ValueError(f"envelope ticks must increase, got {ticks}")

        for name, span in (("loop", self.loop), ("sustain", self.sustain)):
            if span is not None and span.end >= self.length:
                raise ValueError(f"envelope {name} span {span.begin}..{span.end} escapes {self.length} points")

        return self

    @property
    def length(self) -> int:
        """How many breakpoints the curve holds."""
        return len(self.points)
