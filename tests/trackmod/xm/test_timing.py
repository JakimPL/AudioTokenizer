from __future__ import annotations

import pytest

from trackmod.it.timing import exact_timings as it_timings
from trackmod.it.timing import row_frames as it_row_frames
from trackmod.xm.timing import SPEED_BOUND, TEMPO_BOUND, exact_timings, nearest_timing, row_frames

FRAME_RATE = 44100


def test_this_format_shares_the_tracker_clock_where_both_reach() -> None:
    assert row_frames(6, 125, frame_rate=FRAME_RATE) == it_row_frames(6, 125, frame_rate=FRAME_RATE)


def test_the_tempo_word_reaches_rows_a_one_byte_tempo_cannot() -> None:
    # This is the whole point of the format for a caller working to a frame budget: the shortest row
    # Impulse Tracker can reach is bounded by its one-byte tempo, and this lattice runs far past it.
    assert row_frames(1, 441, frame_rate=FRAME_RATE) == 250
    shortest = min(timing.row_frames for timing in exact_timings(frame_rate=FRAME_RATE, speed=1))
    assert shortest < min(timing.row_frames for timing in it_timings(frame_rate=FRAME_RATE, speed=1))


def test_the_bounds_come_from_the_capacity_table() -> None:
    assert TEMPO_BOUND.maximum == 0xFFFF
    assert SPEED_BOUND.maximum == 0xFFFF


def test_the_nearest_timing_lands_on_a_whole_frame_row() -> None:
    timing = nearest_timing(250, frame_rate=FRAME_RATE, speed=1)
    assert timing.row_frames == 250
    assert row_frames(timing.speed, timing.tempo, frame_rate=FRAME_RATE) == timing.row_frames


def test_a_fractional_row_is_refused() -> None:
    with pytest.raises(ValueError):
        row_frames(1, 187, frame_rate=FRAME_RATE)
