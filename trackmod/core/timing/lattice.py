from trackmod.core.timing.timing import Timing
from trackmod.limits.bound import Bound
from trackmod.spec.clock import TICK_DENOMINATOR, TICK_NUMERATOR


def exact_row_frames(speed: int, tempo: int, frame_rate: int) -> int | None:
    """The whole frames one row spans, or ``None`` when the row does not land on a frame boundary."""
    numerator = speed * frame_rate * TICK_NUMERATOR
    denominator = tempo * TICK_DENOMINATOR
    if numerator % denominator != 0:
        return None

    return numerator // denominator


def row_frames(speed: int, tempo: int, *, frame_rate: int, speed_bound: Bound, tempo_bound: Bound) -> int:
    """The frames one row spans at a given speed and tempo.

    A tick lasts ``5 / (2 * tempo)`` seconds and a row lasts ``speed`` ticks, so a row spans
    ``speed * 5 * frame_rate / (2 * tempo)`` frames. Working in that exact rational keeps the whole-frame
    decision free of floating-point slack, which matters when a caller derives a block length from the
    row rather than rounding one to fit.

    Raises:
        ValueError: when speed or tempo leaves its bound, or when the pair gives a fractional row.
    """
    if not speed_bound.contains(speed):
        raise ValueError(f"speed {speed} is outside {speed_bound}")

    if not tempo_bound.contains(tempo):
        raise ValueError(f"tempo {tempo} is outside {tempo_bound}")

    frames = exact_row_frames(speed, tempo, frame_rate)
    if frames is None:
        raise ValueError(f"speed {speed}, tempo {tempo} give a fractional row at {frame_rate} Hz")

    return frames


def exact_timings(*, frame_rate: int, speed: int, speed_bound: Bound, tempo_bound: Bound) -> list[Timing]:
    """Every tempo whose row is a whole number of frames at one speed, ordered by row length.

    Sweeping tempo at a fixed speed walks the achievable row lengths: a shorter row buys time resolution,
    a longer row spends fewer pattern bytes.

    Raises:
        ValueError: when ``speed`` leaves its bound.
    """
    if not speed_bound.contains(speed):
        raise ValueError(f"speed {speed} is outside {speed_bound}")

    timings = []
    for tempo in range(tempo_bound.minimum, tempo_bound.maximum + 1):
        frames = exact_row_frames(speed, tempo, frame_rate)
        if frames is not None:
            timings.append(Timing(speed=speed, tempo=tempo, row_frames=frames))

    return sorted(timings, key=lambda timing: timing.row_frames)


def nearest_timing(
    target_frames: int,
    *,
    frame_rate: int,
    speed: int,
    speed_bound: Bound,
    tempo_bound: Bound,
) -> Timing:
    """The whole-frame timing closest to a target row length, resolving ties to the shorter row.

    Raises:
        ValueError: when no tempo in range gives a whole-frame row at this speed and frame rate.
    """
    candidates = exact_timings(
        frame_rate=frame_rate,
        speed=speed,
        speed_bound=speed_bound,
        tempo_bound=tempo_bound,
    )
    if not candidates:
        raise ValueError(f"no whole-frame tempo at speed {speed}, {frame_rate} Hz")

    return min(candidates, key=lambda timing: (abs(timing.row_frames - target_frames), timing.row_frames))
