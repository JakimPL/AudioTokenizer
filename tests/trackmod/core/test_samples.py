from __future__ import annotations

import numpy as np
import pytest

from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.loop import Loop, LoopMode
from trackmod.core.samples.sample import Sample
from trackmod.spec.levels import MAX_VOLUME

RATE = 44100


def make_sample(*, frames: int = 8, depth: BitDepth = BitDepth.SIXTEEN) -> Sample:
    return Sample(name="atom", pcm=np.zeros(frames), rate=RATE, depth=depth)


def test_a_placeholder_sample_carries_no_frames() -> None:
    empty = make_sample(frames=0)
    assert empty.frames == 0
    assert empty.stored_bytes == 0


def test_stored_size_follows_the_bit_depth() -> None:
    assert make_sample(frames=10).stored_bytes == 20
    assert make_sample(frames=10, depth=BitDepth.EIGHT).stored_bytes == 10
    assert BitDepth.EIGHT.scale == 128.0
    assert BitDepth.SIXTEEN.scale == 32768.0


def test_a_loop_past_the_end_of_the_waveform_is_rejected() -> None:
    with pytest.raises(ValueError):
        Sample(name="s", pcm=np.zeros(8), rate=RATE, loop=Loop(begin=0, end=9))


def test_a_backwards_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        Loop(begin=4, end=4)


def test_a_loop_reports_the_frames_it_repeats() -> None:
    assert Loop(begin=8, end=40, mode=LoopMode.PING_PONG).frames == 32


def test_a_rate_of_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        Sample(name="s", pcm=np.zeros(8), rate=0)


def test_a_level_off_the_shared_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        Sample(name="s", pcm=np.zeros(8), rate=RATE, volume=MAX_VOLUME + 1)


def test_stereo_waveforms_are_rejected() -> None:
    with pytest.raises(ValueError):
        Sample(name="s", pcm=np.zeros((8, 2)), rate=RATE)


def test_samples_compare_by_their_waveform_and_settings() -> None:
    first = Sample(name="s", pcm=np.linspace(-1.0, 1.0, 8), rate=RATE)
    second = Sample(name="s", pcm=np.linspace(-1.0, 1.0, 8), rate=RATE)
    assert first == second
    assert hash(first) == hash(second)
    assert first != Sample(name="s", pcm=np.zeros(8), rate=RATE)
