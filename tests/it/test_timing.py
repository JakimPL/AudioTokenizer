from __future__ import annotations

import pytest

from audiotokenizer.audio.io import SAMPLE_RATE
from audiotokenizer.it.timing import exact_timings, nearest_timing, row_frames


@pytest.mark.parametrize(("tempo", "frames"), [(150, 735), (125, 882), (250, 441), (75, 1470)])
def test_row_frames_hits_the_exact_lattice(tempo: int, frames: int) -> None:
    assert row_frames(1, tempo, frame_rate=SAMPLE_RATE) == frames


def test_row_frames_rejects_a_fractional_row() -> None:
    with pytest.raises(ValueError):
        row_frames(1, 187, frame_rate=SAMPLE_RATE)  # 110250 / 187 is not an integer


def test_row_frames_rejects_out_of_range_tempo() -> None:
    with pytest.raises(ValueError):
        row_frames(1, 300, frame_rate=SAMPLE_RATE)


def test_exact_timings_are_whole_frames_and_sorted() -> None:
    timings = exact_timings(frame_rate=SAMPLE_RATE)
    assert timings
    lengths = [timing.row_frames for timing in timings]
    assert lengths == sorted(lengths)
    for timing in timings:
        assert row_frames(timing.speed, timing.tempo, frame_rate=SAMPLE_RATE) == timing.row_frames


def test_nearest_timing_returns_an_exact_match_when_available() -> None:
    timing = nearest_timing(882, frame_rate=SAMPLE_RATE)
    assert timing.row_frames == 882
    assert timing.tempo == 125


def test_nearest_timing_picks_the_closest_feasible_row() -> None:
    # 588 frames is unreachable (needs tempo 187.5); the nearest lattice rows are 525 and 630.
    timing = nearest_timing(588, frame_rate=SAMPLE_RATE)
    assert timing.row_frames in {525, 630}
