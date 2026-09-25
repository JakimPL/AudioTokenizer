"""The stored samples a dictionary becomes: two per atom, its own PCM and its negation.

A tracker's volume column is unsigned, so a signed coefficient is played by pointing the cell at whichever
of the two stored polarities carries the sign. That doubling is the codec's one storage decision the
formats both pay for, and it is why the routing spreads twice as many samples as there are atoms.
"""

from __future__ import annotations

from typing import Final

from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample

from audiotokenizer.coding.quantisation import StoredAtoms
from audiotokenizer.module.binding import Binding

POLARITIES: Final = 2
_SIGNS: Final = ((1.0, "+"), (-1.0, "-"))


def stored_samples(stored: StoredAtoms, binding: Binding, *, rate: int, depth: BitDepth) -> tuple[Sample, ...]:
    """Every atom's two stored samples, in the slot order the routing reaches them by."""
    samples: list[Sample] = []
    for index, gain in enumerate(stored.global_volume):
        for sign, suffix in _SIGNS:
            samples.append(
                binding.sample(
                    f"atom{index}{suffix}",
                    sign * stored.pcm[index],
                    rate=rate,
                    gain=int(gain),
                    depth=depth,
                )
            )
    return tuple(samples)
