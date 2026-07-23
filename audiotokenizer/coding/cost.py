from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..it.spec import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYBOARD_NOTES,
    MASK_INSTRUMENT,
    MASK_LAST_INSTRUMENT,
    MASK_LAST_NOTE,
    MASK_NOTE,
    MASK_VOLUME,
    MAX_ROWS,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

__all__ = ["Cost", "pattern_bytes", "module_bytes"]


@dataclass(frozen=True)
class Cost:
    pattern: int
    pcm: int
    headers: int
    n_rows: int
    n_cells: int

    @property
    def total(self) -> int:
        return self.pattern + self.pcm + self.headers

    @property
    def kilobytes(self) -> float:
        return self.total / 1024.0

    def bitrate_kbps(self, duration_seconds: float) -> float:
        return self.total * 8.0 / duration_seconds / 1000.0 if duration_seconds > 0 else 0.0


def pattern_bytes(sample_no: NDArray[np.int64], volume: NDArray[np.int64]) -> int:
    """Bytes the packed patterns occupy for an assignment.

    ``sample_no`` and ``volume`` are ``(n_rows, n_channels)``; a cell is present
    where ``volume > 0``. A present cell costs a channel marker and a volume
    byte; a note or instrument byte only when either differs from that channel's
    previous present row (which is where a polarity flip pays); and a mask byte
    only when the cell's mask differs from the channel's last. The mask keeps a
    channel's note alive with :data:`MASK_LAST_NOTE`, which re-triggers the
    one-row sample for free, so a sustained channel settles to two bytes a row.
    Counting runs per channel, so it stays cheap on very long patterns.
    """
    samples = np.asarray(sample_no)
    volumes = np.asarray(volume)
    active = volumes > 0
    n_rows, n_channels = samples.shape
    note = samples % KEYBOARD_NOTES
    instrument = samples // KEYBOARD_NOTES

    total = n_rows + 2 * int(np.count_nonzero(active))  # terminators + marker + volume
    for channel in range(n_channels):
        rows = np.flatnonzero(active[:, channel])
        if rows.size == 0:
            continue
        channel_note = note[rows, channel]
        channel_instrument = instrument[rows, channel]
        note_changed = np.empty(rows.size, dtype=bool)
        note_changed[0] = True
        note_changed[1:] = channel_note[1:] != channel_note[:-1]
        instrument_changed = np.empty(rows.size, dtype=bool)
        instrument_changed[0] = True
        instrument_changed[1:] = channel_instrument[1:] != channel_instrument[:-1]
        total += int(note_changed.sum()) + int(instrument_changed.sum())

        mask = (
            MASK_VOLUME
            | np.where(note_changed, MASK_NOTE, MASK_LAST_NOTE)
            | np.where(instrument_changed, MASK_INSTRUMENT, MASK_LAST_INSTRUMENT)
        )
        mask_changed = np.empty(rows.size, dtype=bool)
        mask_changed[0] = True
        mask_changed[1:] = mask[1:] != mask[:-1]
        total += int(mask_changed.sum())
    return int(total)


def module_bytes(
    sample_no: NDArray[np.int64],
    volume: NDArray[np.int64],
    *,
    n_stored_samples: int,
    pcm_frames: int,
    bits_per_frame: int = 8,
    rows_per_pattern: int = MAX_ROWS,
) -> Cost:
    """The whole ``.IT`` file's size for an assignment and its stored samples."""
    n_rows = int(sample_no.shape[0])
    pattern = pattern_bytes(sample_no, volume)
    pcm = n_stored_samples * pcm_frames * bits_per_frame // 8

    instruments = max(1, -(-n_stored_samples // KEYBOARD_NOTES))
    patterns = max(1, -(-n_rows // rows_per_pattern))
    headers = (
        FILE_HEADER_BYTES
        + 5 * patterns  # order-list entry + pattern offset
        + 4 * instruments  # instrument offset table
        + 4 * n_stored_samples  # sample offset table
        + INSTRUMENT_HEADER_BYTES * instruments
        + SAMPLE_HEADER_BYTES * n_stored_samples
        + PATTERN_HEADER_BYTES * patterns
    )
    cells = int(np.count_nonzero(np.asarray(volume) > 0))
    return Cost(pattern=pattern, pcm=pcm, headers=headers, n_rows=n_rows, n_cells=cells)
