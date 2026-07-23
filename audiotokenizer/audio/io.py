from __future__ import annotations

from pathlib import Path
from typing import Final, Tuple, Union

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

__all__ = ["SAMPLE_RATE", "load_audio", "save_audio", "to_mono", "normalise"]

SAMPLE_RATE: Final = 44100

PathLike = Union[str, Path]


def load_audio(path: PathLike, *, mono: bool = True) -> Tuple[NDArray[np.float64], int]:
    """Read ``path`` and return ``(samples, sample_rate)`` as float64 in [-1, 1]."""
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float64")
    samples = np.asarray(data, dtype=np.float64)
    return (to_mono(samples) if mono else samples), int(sample_rate)


def save_audio(
    path: PathLike,
    samples: NDArray[np.float64],
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
) -> None:
    """Write ``samples`` (float, in [-1, 1]) to ``path`` as a WAV."""
    sf.write(str(path), np.asarray(samples, dtype=np.float64), sample_rate, subtype=subtype)


def to_mono(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average any channel layout down to a 1-D signal."""
    array = np.asarray(data, dtype=np.float64)
    return array if array.ndim == 1 else np.asarray(array.mean(axis=1), dtype=np.float64)


def normalise(signal: NDArray[np.float64], *, peak: float = 0.99) -> NDArray[np.float64]:
    """Scale ``signal`` so its largest absolute value is ``peak``."""
    array = np.asarray(signal, dtype=np.float64)
    largest = float(np.max(np.abs(array))) if array.size else 0.0
    if largest == 0.0:
        return array
    return array * (peak / largest)
