from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.limits.bound import Bound
from trackmod.schema.config import FROZEN


class Capacity(BaseModel):
    """The two bounds a format declares for one capability.

    ``structural`` is what the file's fields can physically hold, derived from the record layout;
    ``canonical`` is what the tracker the format was designed for honours. A value between the two is
    the whole basis of a "hacked" module: the bytes carry it and a modern player reads it.
    """

    model_config = FROZEN

    canonical: Bound
    structural: Bound

    @model_validator(mode="after")
    def _nested(self) -> Capacity:
        if not self.structural.contains(self.canonical.minimum) or not self.structural.contains(self.canonical.maximum):
            raise ValueError(f"canonical bound {self.canonical} escapes the structural bound {self.structural}")

        return self

    @classmethod
    def fixed(cls, bound: Bound) -> Capacity:
        """A capacity whose canonical and structural bounds coincide, for fields with no headroom."""
        return cls(canonical=bound, structural=bound)
