from __future__ import annotations

import pytest

from audiotokenizer.audio.io import SAMPLE_RATE
from audiotokenizer.xm.spec import HACKED_MAX_TEMPO, STRICT_MAX_TEMPO
from audiotokenizer.xm.timing import exact_timings, nearest_timing, row_frames


@pytest.mark.parametrize(("tempo", "frames"), [(125, 882), (250, 441), (441, 250), (2205, 50), (11025, 10)])
def test_row_frames_hits_the_exact_lattice(tempo: int, frames: int) -> None:
    assert row_frames(1, tempo, frame_rate=SAMPLE_RATE) == frames


def test_the_16bit_hack_reaches_tempi_impossible_in_it() -> None:
    # tempo 441 > IT's 255 ceiling gives a 250-frame row — the whole reason XM export exists.
    assert 441 > STRICT_MAX_TEMPO
    assert row_frames(1, 441, frame_rate=SAMPLE_RATE) == 250
    # ... and it stays feasible only under the hacked ceiling, not the strict one.
    with pytest.raises(ValueError):
        row_frames(1, 441, frame_rate=SAMPLE_RATE, max_tempo=STRICT_MAX_TEMPO)


def test_row_frames_rejects_a_fractional_row() -> None:
    with pytest.raises(ValueError):
        row_frames(1, 443, frame_rate=SAMPLE_RATE)  # 110250 / 443 is not an integer


def test_row_frames_rejects_tempo_over_the_16bit_word() -> None:
    with pytest.raises(ValueError):
        row_frames(1, HACKED_MAX_TEMPO + 1, frame_rate=SAMPLE_RATE)


def test_exact_timings_are_whole_frames_and_sorted() -> None:
    timings = exact_timings(frame_rate=SAMPLE_RATE)
    assert timings
    lengths = [timing.row_frames for timing in timings]
    assert lengths == sorted(lengths)
    for timing in timings:
        assert row_frames(timing.speed, timing.tempo, frame_rate=SAMPLE_RATE) == timing.row_frames
    # The hacked ceiling admits far shorter rows than IT's 255 tempo (which floors at 441 frames).
    assert lengths[0] < 441


def test_nearest_timing_returns_an_exact_match_when_available() -> None:
    timing = nearest_timing(250, frame_rate=SAMPLE_RATE)
    assert timing.row_frames == 250
    assert timing.tempo == 441
