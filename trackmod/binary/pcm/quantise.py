import numpy as np
from numpy.typing import NDArray

from trackmod.core.samples.depth import BitDepth


def quantise(pcm: NDArray[np.float64], depth: BitDepth) -> NDArray[np.int64]:
    """Map float PCM in ``[-1, 1]`` onto the signed integer range of ``depth``."""
    scale = depth.scale
    return np.clip(np.round(np.asarray(pcm, dtype=np.float64) * scale), -scale, scale - 1).astype(np.int64)


def dequantise(frames: NDArray[np.int64], depth: BitDepth) -> NDArray[np.float64]:
    """Map stored integer frames back onto float PCM in ``[-1, 1]``."""
    return np.asarray(frames, dtype=np.float64) / depth.scale
