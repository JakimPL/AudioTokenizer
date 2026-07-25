from __future__ import annotations

import numpy as np
from pydantic import BaseModel, model_validator

from trackmod.core.samples.depth import DEFAULT_DEPTH, BitDepth
from trackmod.core.samples.loop import Loop
from trackmod.schema.array import Waveform
from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Panning, Rate, Volume
from trackmod.spec.levels import MAX_VOLUME


class Sample(BaseModel):
    """One stored waveform and how a tracker should sound it.

    ``pcm`` is float in ``[-1, 1]`` and ``rate`` is the frequency in hertz at which it plays back
    unaltered. Formats reach that rate differently — one stores the frequency outright, another tunes the
    triggering key towards it — so the intent is recorded here in hertz and each writer derives its own
    encoding. A sample with no frames is a placeholder slot for a waveform a tracker will supply later.

    ``volume`` is the level a cell without a volume column plays at, and ``gain`` is a fixed multiplier
    applied on top of whatever level plays. A format with no room for a per-sample multiplier bounds
    ``gain`` to full and reports anything quieter, so a caller learns that the scaling has to be baked
    into the waveform instead of being dropped in silence.
    """

    model_config = FROZEN

    name: str
    pcm: Waveform
    rate: Rate
    depth: BitDepth = DEFAULT_DEPTH
    volume: Volume = MAX_VOLUME
    gain: Volume = MAX_VOLUME
    panning: Panning | None = None
    loop: Loop | None = None
    sustain_loop: Loop | None = None

    @model_validator(mode="after")
    def _loops_fit(self) -> Sample:
        for loop in (self.loop, self.sustain_loop):
            if loop is not None and loop.end > self.frames:
                raise ValueError(f"sample {self.name!r} loop end {loop.end} exceeds {self.frames} frames")

        return self

    @property
    def frames(self) -> int:
        """How many frames the waveform holds."""
        return int(self.pcm.size)

    @property
    def stored_bytes(self) -> int:
        """How many bytes the waveform occupies at this sample's depth."""
        return self.frames * self.depth.bytes_per_frame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented

        return _settings(self) == _settings(other) and np.array_equal(self.pcm, other.pcm)

    def __hash__(self) -> int:
        return hash(_settings(self))


def _settings(sample: Sample) -> tuple[object, ...]:
    return (
        sample.name,
        sample.rate,
        sample.depth,
        sample.volume,
        sample.gain,
        sample.panning,
        sample.loop,
        sample.sustain_loop,
        sample.frames,
    )
