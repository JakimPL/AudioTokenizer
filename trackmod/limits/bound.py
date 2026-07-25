from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.schema.config import FROZEN


class Bound(BaseModel):
    """An inclusive integer range a value must land in."""

    model_config = FROZEN

    minimum: int
    maximum: int

    @model_validator(mode="after")
    def _inhabited(self) -> Bound:
        if self.minimum > self.maximum:
            raise ValueError(f"bound {self.minimum}..{self.maximum} is empty")

        return self

    def contains(self, value: int) -> bool:
        """Whether ``value`` lies within the range."""
        return self.minimum <= value <= self.maximum

    def __str__(self) -> str:
        return f"{self.minimum}..{self.maximum}"
