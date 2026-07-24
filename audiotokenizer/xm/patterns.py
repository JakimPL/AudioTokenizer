"""Pattern records and their packed byte stream, plus the global playback settings.

A pattern is a dense grid of cells, present where the volume is positive; :func:`pack_pattern` serializes
it into XM's ``0x80`` mask stream. XM packs very differently from IT, and the byte model rests on these
rules:

* **Every cell is stored.** There is no silent-channel skip and no row terminator — the player reads exactly
  ``rows * n_channels`` cells. A silent channel costs one byte (:data:`EMPTY_CELL`).
* **No "reuse last" memory.** A present cell always emits a mask byte and always re-states its note, because
  the note is what re-triggers the one-shot sample each row. Only the instrument byte is elided, and only
  while the channel's instrument is unchanged (the note alone re-triggers with the channel's last
  instrument). So a kept cell settles to three bytes — mask, note, volume — and pays a fourth on an
  instrument change.

This packing is the exact counterpart of :func:`audiotokenizer.coding.xm_cost.pattern_bytes`; the two are
pinned together by the ``pack_pattern == pattern_bytes`` keystone test.

``sample_no`` is the 0-based atom-and-polarity slot: its instrument is ``sample_no //
SAMPLES_PER_INSTRUMENT`` (emitted 1-based) and its key is ``NOTE_BASE + sample_no % SAMPLES_PER_INSTRUMENT``.
A polarity flip re-points the cell at the ``-s`` sample, changing the note (and re-triggering) but not
necessarily the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.xm.format import PATTERN_HEADER
from audiotokenizer.xm.spec import (
    EMPTY_CELL,
    MASK_INSTRUMENT,
    MASK_NOTE,
    MASK_PACKED,
    MASK_VOLUME,
    MAX_PATTERN_BYTES,
    MAX_ROWS,
    NOTE_BASE,
    PATTERN_HEADER_BYTES,
    PATTERN_PACKING_TYPE,
    SAMPLES_PER_INSTRUMENT,
    VOLUME_COLUMN_BASE,
)

_UNSET = -1  # per-channel "no instrument emitted yet" sentinel, reset at each pattern's first row


@dataclass(frozen=True)
class XMPattern:
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
class XMPlayback:
    """Global playback settings; ``speed``/``tempo`` set the row duration, so they affect timing."""

    speed: int
    tempo: int


def pack_pattern(pattern: XMPattern) -> bytes:
    """Serialize a pattern: a 9-byte header (length, packing, rows, packed size) then the packed cell stream.

    Every cell of every row is emitted in row-major order with no terminator: a silent cell is one
    :data:`EMPTY_CELL` byte, and a present cell is a mask byte, its note, an instrument byte only when the
    channel's instrument changed, and its volume-column byte.
    """
    if not 1 <= pattern.rows <= MAX_ROWS:
        raise ValueError(f"pattern rows {pattern.rows} out of range 1..{MAX_ROWS}")
    sample_no = np.asarray(pattern.sample_no, dtype=np.int64)
    volume = np.asarray(pattern.volume, dtype=np.int64)
    n_channels = sample_no.shape[1]

    last_instrument = [_UNSET] * n_channels

    stream = bytearray()
    for row in range(pattern.rows):
        for channel in range(n_channels):
            level = int(volume[row, channel])
            if level <= 0:
                stream.append(EMPTY_CELL)  # silent channel: one byte, no fields
                continue
            slot = int(sample_no[row, channel])
            instrument = slot // SAMPLES_PER_INSTRUMENT + 1  # 1-based; 0 would mean "no instrument"
            note = NOTE_BASE + slot % SAMPLES_PER_INSTRUMENT

            keep_instrument = instrument == last_instrument[channel]
            mask = MASK_PACKED | MASK_NOTE | MASK_VOLUME
            if not keep_instrument:
                mask |= MASK_INSTRUMENT

            stream.append(mask)
            stream.append(note & 0xFF)  # always: the note re-triggers the one-shot sample
            if not keep_instrument:
                stream.append(instrument & 0xFF)
                last_instrument[channel] = instrument
            stream.append((VOLUME_COLUMN_BASE + level) & 0xFF)

    if len(stream) > MAX_PATTERN_BYTES:
        raise ValueError(
            f"packed pattern is {len(stream)} bytes, over the {MAX_PATTERN_BYTES}-byte u16 limit; "
            "use fewer rows per pattern or a longer row (larger T)"
        )
    header = PATTERN_HEADER.pack(
        {
            "header_length": PATTERN_HEADER_BYTES,
            "packing_type": PATTERN_PACKING_TYPE,
            "rows": pattern.rows,
            "packed_size": len(stream),
        }
    )
    return header + bytes(stream)
