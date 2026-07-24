"""FastTracker 2's view of the shared tracker clock: ``speed``/``tempo`` -> exact integer row length.

XM shares Impulse Tracker's clock exactly; the only difference is the ceiling — XM stores tempo in a
16-bit word, so it reaches rows far shorter than IT's 255-tempo floor allows, which is the whole reason
this format is here. The integrality logic lives in :mod:`audiotokenizer.tracker.timing`; this module only
binds XM's ranges (from :mod:`audiotokenizer.xm.spec`) so callers work in XM terms. See the shared module
for the derivation of ``row_frames``. ``max_tempo`` defaults to the hacked ceiling because the point of
XM export is to exceed 255; the caller passes the profile's real ceiling to bound a sweep.
"""

from __future__ import annotations

from audiotokenizer.tracker.timing import Timing
from audiotokenizer.tracker.timing import exact_timings as _exact_timings
from audiotokenizer.tracker.timing import nearest_timing as _nearest_timing
from audiotokenizer.tracker.timing import row_frames as _row_frames
from audiotokenizer.xm.spec import HACKED_MAX_TEMPO, MIN_SPEED, MIN_TEMPO

__all__ = ["Timing", "row_frames", "exact_timings", "nearest_timing"]


def row_frames(speed: int, tempo: int, *, frame_rate: int, max_tempo: int = HACKED_MAX_TEMPO) -> int:
    """Return the exact frames in one XM row, or raise when ``(speed, tempo)`` gives a fractional row."""
    return _row_frames(
        speed, tempo, frame_rate=frame_rate, min_speed=MIN_SPEED, min_tempo=MIN_TEMPO, max_tempo=max_tempo
    )


def exact_timings(*, frame_rate: int, speed: int = MIN_SPEED, max_tempo: int = HACKED_MAX_TEMPO) -> list[Timing]:
    """Every XM tempo whose row length is a whole number of frames at ``speed``, sorted by row length."""
    return _exact_timings(frame_rate=frame_rate, min_tempo=MIN_TEMPO, max_tempo=max_tempo, speed=speed)


def nearest_timing(
    target_frames: int, *, frame_rate: int, speed: int = MIN_SPEED, max_tempo: int = HACKED_MAX_TEMPO
) -> Timing:
    """The exact XM-row timing whose length is closest to ``target_frames`` (ties -> the shorter row)."""
    return _nearest_timing(target_frames, frame_rate=frame_rate, min_tempo=MIN_TEMPO, max_tempo=max_tempo, speed=speed)
