"""How the dictionary's stored samples are spread over a tracker's instruments and keys.

A tracker plays a sample by pressing a key of an instrument, so a dictionary wider than one instrument's
keymap runs over consecutive instruments. Every atom is stored twice — its positive PCM and its negation —
and each of those slots gets a key of its own, so a pattern cell selects an atom-and-polarity by naming an
instrument and a key rather than by carrying a pitch. Each key sounds its sample at the sample's own
recorded rate, which is what :data:`~trackmod.spec.pitch.RATE_NOTE` means: the routing transposes nothing.

The two formats differ only in how many keys one instrument routes and which key the first sample answers
to, which is why this is one model both bindings configure rather than two implementations.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import RATE_NOTE


class Routing(BaseModel):
    """Which instrument and key a stored sample is reached through."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples_per_instrument: int = Field(ge=1)
    first_key: int = Field(ge=0)

    def instrument(self, slot: int) -> int:
        """Which instrument owns the stored sample in ``slot``."""
        return slot // self.samples_per_instrument

    def key(self, slot: int) -> Note:
        """The key that plays the stored sample in ``slot``."""
        return Note(self.first_key + slot % self.samples_per_instrument)

    def instruments(self, slots: int) -> int:
        """How many instruments a dictionary of ``slots`` stored samples occupies."""
        return max(1, -(-slots // self.samples_per_instrument))

    def keymap(self, instrument: int, *, slots: int) -> Keymap:
        """The keymap of one instrument: each key it owns sounds its sample at that sample's own rate."""
        first = instrument * self.samples_per_instrument
        last = min(first + self.samples_per_instrument, slots)
        return routed_keymap(
            {self.key(slot): KeyAssignment(sample=slot, note=Note(RATE_NOTE)) for slot in range(first, last)}
        )
