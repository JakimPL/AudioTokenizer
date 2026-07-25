from typing import Final

from trackmod.core.timing.lattice import exact_timings as shared_exact_timings
from trackmod.core.timing.lattice import nearest_timing as shared_nearest_timing
from trackmod.core.timing.lattice import row_frames as shared_row_frames
from trackmod.core.timing.timing import Timing
from trackmod.it.spec.capacities import CAPACITIES
from trackmod.limits.capability import Capability

SPEED_BOUND: Final = CAPACITIES[Capability.SPEED].structural
TEMPO_BOUND: Final = CAPACITIES[Capability.TEMPO].structural


def row_frames(speed: int, tempo: int, *, frame_rate: int) -> int:
    """The frames one Impulse Tracker row spans, bound to this format's speed and tempo ranges."""
    return shared_row_frames(
        speed,
        tempo,
        frame_rate=frame_rate,
        speed_bound=SPEED_BOUND,
        tempo_bound=TEMPO_BOUND,
    )


def exact_timings(*, frame_rate: int, speed: int) -> list[Timing]:
    """Every Impulse Tracker tempo whose row is a whole number of frames at one speed."""
    return shared_exact_timings(
        frame_rate=frame_rate,
        speed=speed,
        speed_bound=SPEED_BOUND,
        tempo_bound=TEMPO_BOUND,
    )


def nearest_timing(target_frames: int, *, frame_rate: int, speed: int) -> Timing:
    """The Impulse Tracker timing whose row length is closest to a target."""
    return shared_nearest_timing(
        target_frames,
        frame_rate=frame_rate,
        speed=speed,
        speed_bound=SPEED_BOUND,
        tempo_bound=TEMPO_BOUND,
    )
