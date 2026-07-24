"""The byte-exact size model of an ``.XM`` file for an atom assignment — XM's twin of :mod:`.cost`.

This is the XM counterpart of :mod:`audiotokenizer.coding.cost`, and its :func:`xm_pattern_bytes` is the
exact counterpart of :func:`audiotokenizer.xm.patterns.pack_pattern`: they must agree byte-for-byte, which
the keystone test asserts. XM packs far more simply than IT — no row terminator, no silent-channel skip,
no "reuse last" memory — so the whole per-pattern stream reduces to a closed form:

    ``bytes = rows * channels + 2 * present_cells + instrument_changes``

Every cell costs one byte just to exist (an empty channel is a single ``0x80``); a present cell adds two
more (its note and volume, always spent — the note re-triggers the one-shot sample each row); and a present
cell pays one extra byte whenever its instrument differs from the last one emitted on that channel. The
instrument persists across empty rows, so the change count is taken per channel over its present cells with
the first always counting. ``sample_no`` is the 0-based atom-and-polarity slot; its instrument is
``sample_no // SAMPLES_PER_INSTRUMENT`` and a polarity flip that stays inside one instrument costs no
instrument byte.

XM also has a hard **feasibility wall** IT lacks: the order table holds at most :data:`MAX_PATTERNS`
patterns and each packed pattern is a u16. :func:`xm_writable` reports whether an assignment clears both,
so the compiler can reject an infeasible song with a clear error instead of writing a corrupt file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.coding.cost import Cost, pattern_slices
from audiotokenizer.xm.spec import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    MAX_PATTERNS,
    MAX_ROWS,
    ORDER_TABLE_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
    SAMPLES_PER_INSTRUMENT,
)


def xm_pattern_bytes(sample_no: NDArray[np.int64], volume: NDArray[np.int64]) -> int:
    """Bytes one packed pattern's cell stream occupies (excluding its 9-byte header).

    ``sample_no`` and ``volume`` are ``(n_rows, n_channels)`` for a single pattern; a cell is present where
    ``volume > 0``. Counting instrument changes per channel with fresh state at the pattern's first row
    matches how the writer resets memory each pattern and stays cheap on very long patterns.
    """
    samples = np.asarray(sample_no)
    volumes = np.asarray(volume)
    active = volumes > 0
    n_rows, n_channels = samples.shape
    instrument = samples // SAMPLES_PER_INSTRUMENT

    present = int(np.count_nonzero(active))
    total = n_rows * n_channels + 2 * present  # one byte per cell + note & volume for every present cell
    for channel in range(n_channels):
        rows = np.flatnonzero(active[:, channel])
        if rows.size == 0:
            continue
        channel_instrument = instrument[rows, channel]
        instrument_changed = np.empty(rows.size, dtype=bool)
        instrument_changed[0] = True  # the first present cell always states its instrument
        instrument_changed[1:] = channel_instrument[1:] != channel_instrument[:-1]
        total += int(instrument_changed.sum())
    return int(total)


def _pcm_bytes(n_stored_samples: int, pcm_frames: int, bits_per_frame: int) -> int:
    """The stored PCM's size: one sample body per slot, ``pcm_frames`` frames at ``bits_per_frame``."""
    return n_stored_samples * pcm_frames * bits_per_frame // 8


def _headers_bytes(n_stored_samples: int, n_patterns: int) -> int:
    """The fixed record overhead: file header, order table, and per-record headers (XM keeps no offset tables).

    One instrument owns every :data:`SAMPLES_PER_INSTRUMENT` sample slots, so the instrument count follows
    from ``n_stored_samples``; the rest scale with the pattern and sample counts.
    """
    instruments = max(1, -(-n_stored_samples // SAMPLES_PER_INSTRUMENT))
    return (
        FILE_HEADER_BYTES
        + ORDER_TABLE_BYTES
        + PATTERN_HEADER_BYTES * n_patterns
        + INSTRUMENT_HEADER_BYTES * instruments
        + SAMPLE_HEADER_BYTES * n_stored_samples
    )


def xm_module_bytes(
    sample_no: NDArray[np.int64],
    volume: NDArray[np.int64],
    *,
    n_stored_samples: int,
    pcm_frames: int,
    bits_per_frame: int = 8,
    rows_per_pattern: int = MAX_ROWS,
) -> Cost:
    """The whole ``.XM`` file's size for an assignment and its stored samples, byte-exact to the writer."""
    samples = np.asarray(sample_no)
    volumes = np.asarray(volume)
    n_rows = int(samples.shape[0])
    slices = pattern_slices(n_rows, rows_per_pattern=rows_per_pattern)
    per_pattern = [xm_pattern_bytes(samples[start:stop], volumes[start:stop]) for start, stop in slices]
    pcm = _pcm_bytes(n_stored_samples, pcm_frames, bits_per_frame)
    headers = _headers_bytes(n_stored_samples, max(1, len(slices)))
    cells = int(np.count_nonzero(volumes > 0))
    return Cost(
        pattern=sum(per_pattern),
        pcm=pcm,
        headers=headers,
        n_rows=n_rows,
        n_cells=cells,
        max_pattern=max(per_pattern, default=0),
    )


def xm_writable(cost: Cost, *, rows_per_pattern: int = MAX_ROWS) -> bool:
    """Whether an XM cost clears the feasibility wall: every pattern a u16, and at most :data:`MAX_PATTERNS`.

    XM's order table and pattern count are one byte wide, so a song long enough to need more than
    :data:`MAX_PATTERNS` patterns cannot be written at all — unlike IT, whose pattern count is a u16. A
    caller with many channels and short rows can hit this before it hits the byte budget.
    """
    n_patterns = max(1, len(pattern_slices(cost.n_rows, rows_per_pattern=rows_per_pattern)))
    return cost.writable and n_patterns <= MAX_PATTERNS
