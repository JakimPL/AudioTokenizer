from __future__ import annotations

from pydantic import BaseModel, model_validator

from trackmod.core.samples.sample import Sample
from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Index
from trackmod.xm.spec.sizes import KEYMAP_NOTES
from trackmod.xm.tuning import Tuning


class SampleGroup(BaseModel):
    """The samples one instrument owns, as this format stores them.

    A stored instrument carries its own copy of every sample its keys reach, rather than pointing into a
    table the whole module shares, so a sample two instruments both play is written twice. The keymap
    names positions within this group, and each sample carries the transposition that sounds the keys
    routed to it at the pitch the shared model asks for.
    """

    model_config = FROZEN

    samples: tuple[Sample, ...]
    tunings: tuple[Tuning, ...]
    keymap: tuple[Index, ...]

    @model_validator(mode="after")
    def _consistent(self) -> SampleGroup:
        if len(self.tunings) != len(self.samples):
            raise ValueError(f"{len(self.samples)} samples carry {len(self.tunings)} tunings")

        if len(self.keymap) != KEYMAP_NOTES:
            raise ValueError(f"a keymap covers {KEYMAP_NOTES} keys, got {len(self.keymap)}")

        for key, slot in enumerate(self.keymap):
            if self.samples and slot >= len(self.samples):
                raise ValueError(f"key {key} names sample {slot} of {len(self.samples)}")

        return self

    @property
    def length(self) -> int:
        """How many samples the instrument owns."""
        return len(self.samples)
