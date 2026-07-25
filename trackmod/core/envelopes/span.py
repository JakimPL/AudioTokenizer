from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Index


class EnvelopeSpan(BaseModel):
    """An inclusive range of point indices, as trackers store envelope loop and sustain points."""

    model_config = FROZEN

    begin: Index
    end: Index

    @model_validator(mode="after")
    def _forwards(self) -> EnvelopeSpan:
        if self.end < self.begin:
            raise ValueError(f"envelope span {self.begin}..{self.end} runs backwards")

        return self
