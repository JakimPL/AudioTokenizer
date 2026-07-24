"""Map Impulse Tracker's ``speed``/``tempo`` onto an exact integer row length in frames.

A row lasts ``speed`` ticks and a tick lasts ``TICK_SECONDS_NUMERATOR / tempo`` seconds, so a row spans
``speed * TICK_SECONDS_NUMERATOR * frame_rate / tempo`` frames. The whole codec rests on one atom filling
exactly one row, so only ``(speed, tempo)`` pairs whose row length is a whole number of frames are usable:
the block length is *derived* from the timing, never rounded. Working in the exact rational
``speed * 5 * frame_rate / (2 * tempo)`` keeps that integrality decision free of floating-point slack.
"""

from __future__ import annotations

from dataclasses import dataclass

from audiotokenizer.it.spec import MAX_TEMPO, MIN_SPEED, MIN_TEMPO

_TICK_NUMERATOR = 5  # TICK_SECONDS_NUMERATOR (2.5) as the exact fraction 5/2
_TICK_DENOMINATOR = 2


@dataclass(frozen=True)
class Timing:
    """A row length realized exactly by IT ``speed`` and ``tempo``: ``row_frames`` frames per row."""

    speed: int
    tempo: int
    row_frames: int


def row_frames(speed: int, tempo: int, *, frame_rate: int, max_tempo: int = MAX_TEMPO) -> int:
    """Return the exact frames in one row, or raise when ``(speed, tempo)`` gives a fractional row.

    Raises:
        ValueError: when ``speed * 2.5 * frame_rate / tempo`` is not a whole number of frames, or when
            ``speed``/``tempo`` fall outside the IT ranges.
    """
    if not MIN_SPEED <= speed <= max_tempo:
        raise ValueError(f"speed {speed} out of range {MIN_SPEED}..{max_tempo}")
    if not MIN_TEMPO <= tempo <= max_tempo:
        raise ValueError(f"tempo {tempo} out of range {MIN_TEMPO}..{max_tempo}")

    numerator = speed * frame_rate * _TICK_NUMERATOR
    denominator = tempo * _TICK_DENOMINATOR
    if numerator % denominator != 0:
        raise ValueError(f"speed {speed}, tempo {tempo} give a fractional row at {frame_rate} Hz")

    return numerator // denominator


def exact_timings(*, frame_rate: int, speed: int = MIN_SPEED) -> list[Timing]:
    """Every tempo whose row length is a whole number of frames at ``speed``, sorted by row length.

    Sweeping tempo at a fixed ``speed`` walks the achievable block lengths: a shorter row (higher tempo)
    buys time resolution, a longer row spends fewer pattern bytes.
    """
    timings: list[Timing] = []
    for tempo in range(MIN_TEMPO, MAX_TEMPO + 1):
        numerator = speed * frame_rate * _TICK_NUMERATOR
        denominator = tempo * _TICK_DENOMINATOR
        if numerator % denominator == 0:
            timings.append(Timing(speed=speed, tempo=tempo, row_frames=numerator // denominator))
    return sorted(timings, key=lambda timing: timing.row_frames)


def nearest_timing(target_frames: int, *, frame_rate: int, speed: int = MIN_SPEED) -> Timing:
    """The exact-row timing whose length is closest to ``target_frames`` (ties resolve to the shorter row).

    Raises:
        ValueError: when no tempo in range yields a whole-frame row at ``speed`` and ``frame_rate``.
    """
    candidates = exact_timings(frame_rate=frame_rate, speed=speed)
    if not candidates:
        raise ValueError(f"no exact-row tempo at speed {speed}, {frame_rate} Hz")
    return min(candidates, key=lambda timing: (abs(timing.row_frames - target_frames), timing.row_frames))
