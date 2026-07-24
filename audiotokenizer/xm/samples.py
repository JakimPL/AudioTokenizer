"""The stored-sample record: its data shape, delta-encoded PCM, and 40-byte XM sample header.

XM differs from IT in two ways that live here. Its PCM is **delta-encoded** — the file stores successive
differences and the player integrates them — so :func:`pcm_bytes` writes ``diff`` of the quantised samples
(with a leading absolute value, ``prepend=0``; the old writer's ``prepend=x[0]`` dropped it and shifted the
whole sample by a DC step). And XM carries no explicit playback rate: a sample plays native only when its
``relative_note`` (and ``finetune``) put the triggering key on the tuning reference, so those ride in the
header instead of IT's C5Speed. The atom's static gain rides in the sample **volume** byte, the counterpart
of IT's sample global-volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.xm.format import SAMPLE_HEADER, encode_name
from audiotokenizer.xm.spec import (
    MAX_VOLUME,
    NAME_BYTES,
    PAN_CENTER,
    RELATIVE_NOTE_TARGET,
    SAMPLE_FINETUNE,
    SMP_LOOP_FORWARD,
    SMP_LOOP_NONE,
    SMP_TYPE_16BIT,
)

DEFAULT_DEPTH_BITS: Final = 16
_INT16_SCALE: Final = 32768.0
_INT8_SCALE: Final = 128.0


@dataclass(frozen=True)
class XMSample:
    """One stored sample: its PCM (float in ``[-1, 1]``), storage depth, gain, and note-relative tuning.

    ``relative_note``/``finetune`` re-tune the key that triggers this sample so it sounds at its recorded
    rate; the defaults put native 44100 Hz on the pattern key (see :mod:`audiotokenizer.xm.spec`).
    """

    name: str
    pcm: NDArray[np.floating]  # (frames,) float in [-1, 1]
    depth_bits: int = DEFAULT_DEPTH_BITS
    volume: int = MAX_VOLUME  # 0..64, the atom's static gain (IT's sample global-volume counterpart)
    relative_note: int = 0
    finetune: int = SAMPLE_FINETUNE
    loop: tuple[int, int] | None = None  # forward loop over half-open frame range [begin, end)

    @property
    def frames(self) -> int:
        return int(np.asarray(self.pcm).size)


def relative_note_for(play_note: int) -> int:
    """The relative-note that re-tunes key ``play_note`` to the native-rate reference (a signed byte)."""
    return RELATIVE_NOTE_TARGET - play_note


def _depth_dtype_and_scale(depth_bits: int) -> tuple[str, float]:
    """Return the little-endian signed dtype and full-scale factor for an 8- or 16-bit depth.

    The single home of the supported-depth guard: both the PCM conversion and the 16-bit header flag derive
    their choice from it, so an unsupported depth raises in exactly one place.
    """
    if depth_bits == 16:
        return "<i2", _INT16_SCALE
    if depth_bits == 8:
        return "<i1", _INT8_SCALE
    raise ValueError(f"unsupported depth {depth_bits} (expected 8 or 16)")


def pcm_bytes(sample: XMSample) -> bytes:
    """Delta-encode float PCM in ``[-1, 1]`` to signed little-endian bytes at the sample's depth.

    The differences are taken in a width wider than the store and cast back, so a delta that overshoots the
    signed range wraps exactly as the player's running-sum integrator will unwrap it. ``prepend=0`` makes
    the first stored value the sample's first absolute amplitude.
    """
    pcm = np.asarray(sample.pcm, dtype=np.float64)
    dtype, scale = _depth_dtype_and_scale(sample.depth_bits)
    quantized = np.clip(np.round(pcm * scale), -scale, scale - 1).astype(np.int64)
    delta = np.diff(quantized, prepend=0).astype(dtype)  # cast wraps the difference to the stored width
    return delta.tobytes()


def sample_header(sample: XMSample) -> bytes:
    """Serialize a 40-byte XM sample header (its PCM follows all headers of its instrument, not a pointer)."""
    dtype, _ = _depth_dtype_and_scale(sample.depth_bits)  # validates the depth; raises if unsupported
    bytes_per_frame = 2 if dtype == "<i2" else 1
    type_flags = SMP_TYPE_16BIT if bytes_per_frame == 2 else 0
    loop_start, loop_length = 0, 0
    if sample.loop is not None:
        type_flags |= SMP_LOOP_FORWARD
        begin, end = sample.loop
        loop_start, loop_length = begin * bytes_per_frame, (end - begin) * bytes_per_frame
    else:
        type_flags |= SMP_LOOP_NONE
    return SAMPLE_HEADER.pack(
        {
            "length": sample.frames * bytes_per_frame,
            "loop_start": loop_start,
            "loop_length": loop_length,
            "volume": min(sample.volume, MAX_VOLUME),
            "finetune": int(sample.finetune),
            "type": type_flags,
            "panning": PAN_CENTER,
            "relative_note": int(sample.relative_note),
            "reserved": 0,
            "name": encode_name(sample.name, NAME_BYTES),
        }
    )
