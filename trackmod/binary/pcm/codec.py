import numpy as np
from numpy.typing import NDArray

from trackmod.binary.pcm.encoding import PcmEncoding, dtype_for
from trackmod.binary.pcm.quantise import dequantise, quantise
from trackmod.core.samples.depth import BitDepth


def encode_pcm(pcm: NDArray[np.float64], *, depth: BitDepth, encoding: PcmEncoding) -> bytes:
    """Serialise float PCM in ``[-1, 1]`` to stored frames.

    Deltas are taken in a width wider than the store and cast back, so a difference that overshoots the
    signed range wraps exactly as the player's running sum unwraps it. The first stored delta is taken
    against zero, which makes it the waveform's first absolute amplitude.
    """
    dtype = dtype_for(depth)
    quantised = quantise(pcm, depth)
    match encoding:
        case PcmEncoding.ABSOLUTE:
            return quantised.astype(dtype).tobytes()
        case PcmEncoding.DELTA:
            return np.diff(quantised, prepend=0).astype(dtype).tobytes()


def decode_pcm(data: bytes, *, depth: BitDepth, encoding: PcmEncoding) -> NDArray[np.float64]:
    """Read stored frames back as float PCM in ``[-1, 1]``."""
    dtype = dtype_for(depth)
    stored = np.frombuffer(data, dtype=dtype)
    match encoding:
        case PcmEncoding.ABSOLUTE:
            frames = stored
        case PcmEncoding.DELTA:
            frames = np.cumsum(stored, dtype=dtype)

    return dequantise(frames, depth)
