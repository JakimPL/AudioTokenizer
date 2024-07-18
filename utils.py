import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

SAMPLING_RATE = 44100


def play_sample(sample: np.ndarray, rate: int = SAMPLING_RATE):
    from IPython.display import Audio, display
    display(Audio(sample, rate=rate))


def save_sample(sample: np.ndarray, path: os.PathLike, rate: int = SAMPLING_RATE):
    reconstruction_path = Path(path).with_suffix(".wav")
    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(reconstruction_path, rate, sample)


def float_to_int16(signal: np.ndarray) -> np.ndarray:
    signal = np.clip(signal, 0, 1)
    signal = 2 * signal - 1
    signal = signal * 32767
    int16_signal = signal.astype(np.int16)
    return int16_signal


def int16_to_float(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32) / 32767.0
    return signal


def pad(signal: np.ndarray, new_size: int, value: Any = 0.0) -> np.ndarray:
    length = len(signal)
    if length > new_size:
        return signal[:new_size]
    else:
        return np.pad(signal, (0, new_size - length), mode="constant", constant_values=value)


def convert_to_mono(signal: np.ndarray) -> np.ndarray:
    if signal.ndim >= 2:
        signal = signal.mean(axis=1)

    return signal


def infer_module_format(path: Path) -> str:
    module_format = path.suffix[1:].lower()
    if module_format not in ["xm", "it"]:
        raise ValueError(f"Unsupported module format: {module_format}. The supported formats are XM and IT.")

    return module_format
