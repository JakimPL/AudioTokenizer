"""Pattern records and their packed byte stream, plus the global playback settings.

A pattern is a dense grid of cells, present where the volume is positive; :func:`pack_pattern` serializes
it into IT's channel-marker byte stream with the mask-reuse compression the byte budget rests on. IT keeps
per-channel memory while unpacking a pattern: the ``0x10``/``0x20`` mask bits re-use a channel's last note
and instrument (spending no byte and still re-triggering the sample), and the mask byte itself is emitted
only when it changes. A channel that keeps its atom therefore settles to two bytes a row — a marker and a
volume. This packing is the exact counterpart of :func:`audiotokenizer.coding.cost.pattern_bytes`; the two
are pinned together by the ``pack_pattern == pattern_bytes`` keystone test.

``sample_no`` is the 0-based atom-and-polarity slot: its keyboard note is ``sample_no % KEYBOARD_NOTES``
and its instrument is ``sample_no // KEYBOARD_NOTES`` (both 0-based here; the emitted instrument byte is
1-based). Playback settings ride alongside because ``speed``/``tempo`` set each row's duration.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.it.spec import (
    CHANNEL_MARKER,
    END_OF_ROW,
    KEYBOARD_NOTES,
    MASK_INSTRUMENT,
    MASK_LAST_INSTRUMENT,
    MASK_LAST_NOTE,
    MASK_NOTE,
    MASK_VOLUME,
    MAX_ROWS,
)

_PACKED_LENGTH_MAX = 0xFFFF  # the pattern header stores the packed byte length as a u16
_UNSET = -1  # per-channel "no previous value yet" sentinel, reset at each pattern's first row


@dataclass(frozen=True)
class ITPattern:
    """A pattern of ``rows`` rows over ``n_channels`` channels, given as aligned integer grids.

    A cell is present where ``volume > 0``; ``sample_no`` is read only there. Both arrays are
    ``(rows, n_channels)``.
    """

    rows: int
    sample_no: NDArray[np.int64]
    volume: NDArray[np.int64]

    @property
    def n_channels(self) -> int:
        return int(self.sample_no.shape[1])


@dataclass(frozen=True)
class ITPlayback:
    """Global playback settings; ``speed``/``tempo`` set the row duration, so they affect timing."""

    speed: int
    tempo: int
    global_volume: int
    mix_volume: int


def pack_pattern(pattern: ITPattern) -> bytes:
    """Serialize a pattern: an 8-byte header (packed length, rows, reserved) then the packed row stream.

    Emission per active cell follows IT's order — an optional mask byte, then the note, instrument and
    volume bytes the mask calls for — with a channel's note, instrument and mask each re-used from its
    previous present row whenever they are unchanged.
    """
    if not 1 <= pattern.rows <= MAX_ROWS:
        raise ValueError(f"pattern rows {pattern.rows} out of range 1..{MAX_ROWS}")
    sample_no = np.asarray(pattern.sample_no, dtype=np.int64)
    volume = np.asarray(pattern.volume, dtype=np.int64)
    n_channels = sample_no.shape[1]

    last_note = [_UNSET] * n_channels
    last_instrument = [_UNSET] * n_channels
    last_mask = [_UNSET] * n_channels

    stream = bytearray()
    for row in range(pattern.rows):
        for channel in range(n_channels):
            level = int(volume[row, channel])
            if level <= 0:
                continue
            slot = int(sample_no[row, channel])
            note = slot % KEYBOARD_NOTES
            instrument = slot // KEYBOARD_NOTES + 1  # 1-based in the cell; 0 would mean "no instrument"

            keep_note = note == last_note[channel]
            keep_instrument = instrument == last_instrument[channel]
            mask = MASK_VOLUME
            mask |= MASK_LAST_NOTE if keep_note else MASK_NOTE
            mask |= MASK_LAST_INSTRUMENT if keep_instrument else MASK_INSTRUMENT

            if mask == last_mask[channel]:
                stream.append((channel + 1) & 0xFF)  # high bit clear -> re-use the channel's last mask
            else:
                stream.append(((channel + 1) | CHANNEL_MARKER) & 0xFF)
                stream.append(mask)
                last_mask[channel] = mask
            if not keep_note:
                stream.append(note & 0xFF)
                last_note[channel] = note
            if not keep_instrument:
                stream.append(instrument & 0xFF)
                last_instrument[channel] = instrument
            stream.append(level & 0xFF)
        stream.append(END_OF_ROW)

    if len(stream) > _PACKED_LENGTH_MAX:
        raise ValueError(
            f"packed pattern is {len(stream)} bytes, over the {_PACKED_LENGTH_MAX}-byte u16 limit; "
            "use fewer rows per pattern or a longer row (larger T)"
        )
    return struct.pack("<HHI", len(stream), pattern.rows, 0) + bytes(stream)
