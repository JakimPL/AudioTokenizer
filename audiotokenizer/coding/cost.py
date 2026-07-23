"""The byte-exact size model of an uncompressed ``.IT`` file for an atom assignment.

This is the whole basis of the byte budget, and its ``pattern_bytes`` is the exact counterpart of the
writer's :func:`audiotokenizer.it.patterns.pack_pattern`: they must agree byte-for-byte, which the
keystone test asserts. A packed row is a list of present channels ending in a zero byte; a present cell
costs a marker and a volume byte, a note byte only when the atom differs from that channel's previous
present row, and a mask byte only when the cell's mask changes. Impulse Tracker resets its per-channel
mask/note memory at each pattern boundary, so the model sums over the same 200-row pattern slices the
writer emits. ``sample_no`` is the 0-based atom-and-polarity slot (note ``sample_no % KEYBOARD_NOTES``,
instrument ``sample_no // KEYBOARD_NOTES``); a polarity flip re-points a cell at the ``−s`` sample, so it
changes the note and pays a note byte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.it.spec import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    KEYBOARD_NOTES,
    MASK_INSTRUMENT,
    MASK_LAST_INSTRUMENT,
    MASK_LAST_NOTE,
    MASK_NOTE,
    MASK_VOLUME,
    MAX_ROWS,
    OFFSET_TABLE_ENTRY_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

_ORDER_TERMINATOR_BYTES = 1


@dataclass(frozen=True)
class Cost:
    """The byte breakdown of a written module: packed patterns, stored PCM, and fixed record overhead."""

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


def pattern_slices(n_rows: int, *, rows_per_pattern: int = MAX_ROWS) -> list[tuple[int, int]]:
    """Split ``n_rows`` rows into the half-open ``[start, stop)`` spans of successive patterns."""
    if n_rows <= 0:
        return []
    return [(start, min(start + rows_per_pattern, n_rows)) for start in range(0, n_rows, rows_per_pattern)]


def pattern_bytes(sample_no: NDArray[np.int64], volume: NDArray[np.int64]) -> int:
    """Bytes one packed pattern's row stream occupies (excluding its 8-byte header).

    ``sample_no`` and ``volume`` are ``(n_rows, n_channels)`` for a single pattern; a cell is present
    where ``volume > 0``. Counting runs per channel with fresh state at the pattern's first row, so it
    stays cheap on very long patterns and matches how the writer resets memory each pattern.
    """
    samples = np.asarray(sample_no)
    volumes = np.asarray(volume)
    active = volumes > 0
    n_rows, n_channels = samples.shape
    note = samples % KEYBOARD_NOTES
    instrument = samples // KEYBOARD_NOTES

    total = n_rows + 2 * int(np.count_nonzero(active))  # row terminators + marker + volume per present cell
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
    """The whole ``.IT`` file's size for an assignment and its stored samples, byte-exact to the writer."""
    samples = np.asarray(sample_no)
    volumes = np.asarray(volume)
    n_rows = int(samples.shape[0])
    slices = pattern_slices(n_rows, rows_per_pattern=rows_per_pattern)
    pattern = sum(pattern_bytes(samples[start:stop], volumes[start:stop]) for start, stop in slices)
    pcm = n_stored_samples * pcm_frames * bits_per_frame // 8

    instruments = max(1, -(-n_stored_samples // KEYBOARD_NOTES))
    patterns = max(1, len(slices))
    order_bytes = patterns + _ORDER_TERMINATOR_BYTES
    offset_tables = OFFSET_TABLE_ENTRY_BYTES * (instruments + n_stored_samples + patterns)
    record_headers = (
        INSTRUMENT_HEADER_BYTES * instruments + SAMPLE_HEADER_BYTES * n_stored_samples + PATTERN_HEADER_BYTES * patterns
    )
    headers = FILE_HEADER_BYTES + order_bytes + offset_tables + record_headers
    cells = int(np.count_nonzero(volumes > 0))
    return Cost(pattern=pattern, pcm=pcm, headers=headers, n_rows=n_rows, n_cells=cells)
