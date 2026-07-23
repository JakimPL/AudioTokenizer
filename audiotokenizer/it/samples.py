"""The stored-sample record: its data shape, PCM quantization, and 80-byte IMPS header."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.it.format import SAMPLE_HEADER, encode_name
from audiotokenizer.it.spec import (
    CVT_SIGNED,
    MAGIC_SAMPLE,
    MAX_VOLUME,
    NAME_BYTES,
    SMP_FLAG_16BIT,
    SMP_FLAG_DATA,
    SMP_FLAG_LOOP,
)

DEFAULT_DEPTH_BITS: Final = 16
DEFAULT_C5SPEED_HZ: Final = 44_100
_INT16_SCALE: Final = 32768.0
_INT8_SCALE: Final = 128.0


@dataclass(frozen=True)
class ITSample:
    """One stored sample: its PCM (float in ``[-1, 1]``), storage depth and playback rate."""

    name: str
    pcm: NDArray[np.floating]  # (frames,) float in [-1, 1]
    depth_bits: int = DEFAULT_DEPTH_BITS
    c5speed: int = DEFAULT_C5SPEED_HZ
    global_volume: int = MAX_VOLUME
    default_volume: int = MAX_VOLUME
    loop: tuple[int, int] | None = None  # forward loop over half-open frame range [begin, end)

    @property
    def frames(self) -> int:
        return int(np.asarray(self.pcm).size)


def _depth_dtype_and_scale(depth_bits: int) -> tuple[str, float]:
    """Return the little-endian signed dtype and full-scale factor for an 8- or 16-bit depth.

    This is the single home of the supported-depth guard: both the PCM conversion and the sample-header
    flag derive their 8-vs-16-bit choice from it, so an unsupported depth raises in exactly one place.
    """
    if depth_bits == 16:
        return "<i2", _INT16_SCALE
    if depth_bits == 8:
        return "<i1", _INT8_SCALE
    raise ValueError(f"unsupported depth {depth_bits} (expected 8 or 16)")


def pcm_bytes(sample: ITSample) -> bytes:
    """Convert float PCM in ``[-1, 1]`` to signed little-endian bytes at the sample's depth."""
    pcm = np.asarray(sample.pcm, dtype=np.float64)
    dtype, scale = _depth_dtype_and_scale(sample.depth_bits)
    quantized = np.clip(np.round(pcm * scale), -scale, scale - 1).astype(dtype)
    return quantized.tobytes()


def sample_header(sample: ITSample, data_offset: int) -> bytes:
    """Serialize an 80-byte IMPS sample header pointing at ``data_offset``."""
    dtype, _ = _depth_dtype_and_scale(sample.depth_bits)  # validates the depth; raises if unsupported
    flags = SMP_FLAG_DATA | (SMP_FLAG_16BIT if dtype == "<i2" else 0)
    loop_begin, loop_end = 0, 0
    if sample.loop is not None:
        flags |= SMP_FLAG_LOOP
        loop_begin, loop_end = sample.loop  # loop end is the frame after the loop; playback wraps here
    return SAMPLE_HEADER.pack(
        {
            "magic": MAGIC_SAMPLE,
            "global_volume": min(sample.global_volume, MAX_VOLUME),
            "flags": flags,
            "default_volume": min(sample.default_volume, MAX_VOLUME),
            "name": encode_name(sample.name, NAME_BYTES),
            "convert": CVT_SIGNED,
            "length": sample.frames,
            "loop_begin": loop_begin,
            "loop_end": loop_end,
            "c5speed": int(sample.c5speed),
            "sample_pointer": data_offset,
        }
    )
