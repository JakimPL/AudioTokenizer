from typing import Final

from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.capacity import Capacity
from trackmod.spec.levels import MAX_INSTRUMENT_VOLUME, MAX_PANNING, MAX_VOLUME
from trackmod.spec.width import WORD_MAX
from trackmod.xm.spec.ranges import (
    CANONICAL_MAX_CHANNELS,
    CANONICAL_MAX_INSTRUMENTS,
    CANONICAL_MAX_SPEED,
    CANONICAL_MAX_TEMPO,
    CANONICAL_MIN_TEMPO,
    CANONICAL_SAMPLES_PER_INSTRUMENT,
    EXTENDED_MAX_CHANNELS,
    EXTENDED_MAX_INSTRUMENTS,
    EXTENDED_SAMPLES_PER_INSTRUMENT,
    MAX_NOTE,
    MAX_ORDERS,
    MAX_PATTERNS,
    MAX_ROWS,
    MAX_SAMPLE_FRAMES,
    MAX_SAMPLE_RATE,
    MAX_SAMPLES,
    MAX_SPEED,
    MAX_TEMPO,
    MIN_ROWS,
    MIN_SAMPLE_RATE,
    MIN_SPEED,
    MIN_TEMPO,
)
from trackmod.xm.spec.sizes import ENVELOPE_POINTS

CAPACITIES: Final = {
    Capability.CHANNELS: Capacity(
        canonical=Bound(minimum=1, maximum=CANONICAL_MAX_CHANNELS),
        structural=Bound(minimum=1, maximum=EXTENDED_MAX_CHANNELS),
    ),
    Capability.PATTERNS: Capacity.fixed(Bound(minimum=0, maximum=MAX_PATTERNS)),
    Capability.ORDERS: Capacity.fixed(Bound(minimum=0, maximum=MAX_ORDERS)),
    Capability.PATTERN_ROWS: Capacity.fixed(Bound(minimum=MIN_ROWS, maximum=MAX_ROWS)),
    Capability.PATTERN_BYTES: Capacity.fixed(Bound(minimum=0, maximum=WORD_MAX)),
    Capability.INSTRUMENTS: Capacity(
        canonical=Bound(minimum=0, maximum=CANONICAL_MAX_INSTRUMENTS),
        structural=Bound(minimum=0, maximum=EXTENDED_MAX_INSTRUMENTS),
    ),
    Capability.SAMPLES: Capacity.fixed(Bound(minimum=0, maximum=MAX_SAMPLES)),
    Capability.SAMPLES_PER_INSTRUMENT: Capacity(
        canonical=Bound(minimum=0, maximum=CANONICAL_SAMPLES_PER_INSTRUMENT),
        structural=Bound(minimum=0, maximum=EXTENDED_SAMPLES_PER_INSTRUMENT),
    ),
    Capability.SAMPLE_FRAMES: Capacity.fixed(Bound(minimum=0, maximum=MAX_SAMPLE_FRAMES)),
    Capability.SAMPLE_RATE: Capacity.fixed(Bound(minimum=MIN_SAMPLE_RATE, maximum=MAX_SAMPLE_RATE)),
    Capability.SAMPLE_VOLUME: Capacity.fixed(Bound(minimum=0, maximum=MAX_VOLUME)),
    Capability.SAMPLE_GAIN: Capacity.fixed(Bound(minimum=MAX_VOLUME, maximum=MAX_VOLUME)),
    Capability.INSTRUMENT_VOLUME: Capacity.fixed(Bound(minimum=MAX_INSTRUMENT_VOLUME, maximum=MAX_INSTRUMENT_VOLUME)),
    Capability.ENVELOPE_POINTS: Capacity.fixed(Bound(minimum=1, maximum=ENVELOPE_POINTS)),
    Capability.ENVELOPE_VALUE: Capacity.fixed(Bound(minimum=0, maximum=MAX_VOLUME)),
    Capability.ENVELOPE_TICK: Capacity.fixed(Bound(minimum=0, maximum=WORD_MAX)),
    Capability.FADEOUT: Capacity.fixed(Bound(minimum=0, maximum=WORD_MAX)),
    Capability.NOTE: Capacity.fixed(Bound(minimum=0, maximum=MAX_NOTE)),
    Capability.TEMPO: Capacity(
        canonical=Bound(minimum=CANONICAL_MIN_TEMPO, maximum=CANONICAL_MAX_TEMPO),
        structural=Bound(minimum=MIN_TEMPO, maximum=MAX_TEMPO),
    ),
    Capability.SPEED: Capacity(
        canonical=Bound(minimum=MIN_SPEED, maximum=CANONICAL_MAX_SPEED),
        structural=Bound(minimum=MIN_SPEED, maximum=MAX_SPEED),
    ),
    Capability.VOLUME: Capacity.fixed(Bound(minimum=0, maximum=MAX_VOLUME)),
    Capability.PANNING: Capacity.fixed(Bound(minimum=0, maximum=MAX_PANNING)),
}
