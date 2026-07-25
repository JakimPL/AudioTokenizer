from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from trackmod.binary.pcm.codec import decode_pcm, encode_pcm
from trackmod.binary.pcm.encoding import PcmEncoding
from trackmod.binary.pcm.quantise import quantise
from trackmod.core.samples.depth import BitDepth

DEPTHS = list(BitDepth)
ENCODINGS = list(PcmEncoding)


@pytest.fixture
def waveform() -> NDArray[np.float64]:
    """A sine plus a full-scale ramp — the ramp is the worst case for delta storage."""
    time = np.linspace(0.0, 1.0, 512, endpoint=False)
    return np.concatenate([np.sin(2 * np.pi * 4 * time), np.linspace(-1.0, 1.0, 512, endpoint=False)])


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda depth: f"{depth}bit")
@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda encoding: str(encoding))
def test_stored_frames_read_back_within_one_quantisation_step(
    waveform: NDArray[np.float64], depth: BitDepth, encoding: PcmEncoding
) -> None:
    recovered = decode_pcm(encode_pcm(waveform, depth=depth, encoding=encoding), depth=depth, encoding=encoding)
    assert np.max(np.abs(recovered - waveform)) <= 1.5 / depth.scale


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda depth: f"{depth}bit")
@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda encoding: str(encoding))
def test_stored_size_is_one_frame_per_sample(
    waveform: NDArray[np.float64], depth: BitDepth, encoding: PcmEncoding
) -> None:
    stored = encode_pcm(waveform, depth=depth, encoding=encoding)
    assert len(stored) == waveform.size * depth.bytes_per_frame


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda depth: f"{depth}bit")
def test_the_first_delta_is_the_first_absolute_amplitude(depth: BitDepth) -> None:
    # Taking the first difference against zero is what keeps a delta-stored waveform free of a DC step.
    pcm = np.asarray([0.5, 0.5, 0.5])
    delta = np.frombuffer(encode_pcm(pcm, depth=depth, encoding=PcmEncoding.DELTA), dtype=f"<i{depth // 8}")
    assert delta[0] == quantise(pcm, depth)[0]
    assert np.all(delta[1:] == 0)


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda depth: f"{depth}bit")
def test_a_delta_that_overshoots_the_stored_width_wraps_back_on_the_running_sum(depth: BitDepth) -> None:
    swing = np.asarray([-1.0, 1.0, -1.0, 1.0])
    recovered = decode_pcm(
        encode_pcm(swing, depth=depth, encoding=PcmEncoding.DELTA), depth=depth, encoding=PcmEncoding.DELTA
    )
    assert np.max(np.abs(recovered - swing)) <= 1.5 / depth.scale


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda depth: f"{depth}bit")
def test_quantisation_saturates_at_the_signed_range(depth: BitDepth) -> None:
    quantised = quantise(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), depth)
    assert quantised.min() == -depth.scale
    assert quantised.max() == depth.scale - 1
