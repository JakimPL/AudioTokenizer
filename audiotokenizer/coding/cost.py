"""Budget bookkeeping: the byte figures a compilation is reported and planned against.

The exact size of a written module is ``trackmod``'s own model of it — :meth:`TrackerModule.size`, pinned
to the writers by the keystone tests — so nothing here re-derives it. What remains is the arithmetic the
budget is spoken in (kilobytes, kilobits per second) and one thing the module model cannot supply: a bound
on a candidate's size *before* it is compiled.

:func:`cost_lower_bound` is that bound, and it exists for the planner. Assembling a candidate costs a
per-row assignment and a full metric pass, so the sweep needs to reject a hopeless config from cell counts
alone. A packed Impulse Tracker pattern spends at least a terminator per row and a marker plus a volume
byte per present cell — note, instrument and mask bytes only ever add to that — while the PCM and the
record overhead are known exactly from the sample and pattern counts. The result therefore never exceeds
the true size, so a candidate it rejects could not have fit, and the sweep loses nothing by trusting it.
"""

from __future__ import annotations

from typing import Final

from trackmod.trackers.it.spec.orders import ORDER_TERMINATOR_BYTES
from trackmod.trackers.it.spec.sizes import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    OFFSET_TABLE_ENTRY_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)

from audiotokenizer.module.it import IT_BINDING

BITS_PER_BYTE: Final = 8
BYTES_PER_KILOBYTE: Final = 1024
BITS_PER_KILOBIT: Final = 1000

DEFAULT_BITS_PER_FRAME: Final = 8


def kilobytes(size_bytes: int) -> float:
    """A byte count in kilobytes, the unit a byte budget is quoted in."""
    return size_bytes / BYTES_PER_KILOBYTE


def bitrate_kbps(size_bytes: int, duration_seconds: float) -> float:
    """The rate a module of this size spends on a signal of this length, zero for an empty signal."""
    if duration_seconds <= 0:
        return 0.0
    return size_bytes * BITS_PER_BYTE / duration_seconds / BITS_PER_KILOBIT


def _pcm_bytes(n_stored_samples: int, pcm_frames: int, bits_per_frame: int) -> int:
    """The stored PCM's size: one sample body per slot, ``pcm_frames`` frames at ``bits_per_frame``."""
    return n_stored_samples * pcm_frames * bits_per_frame // BITS_PER_BYTE


def _record_bytes(n_stored_samples: int, n_patterns: int) -> int:
    """The fixed overhead: the file header, the order list, the offset tables and every record header.

    The instrument count follows from the sample count, because one instrument routes as many stored
    samples as this format's keymap reaches.
    """
    n_instruments = IT_BINDING.routing.instruments(n_stored_samples)
    offsets = OFFSET_TABLE_ENTRY_BYTES * (n_instruments + n_stored_samples + n_patterns)
    return (
        FILE_HEADER_BYTES
        + n_patterns
        + ORDER_TERMINATOR_BYTES
        + offsets
        + INSTRUMENT_HEADER_BYTES * n_instruments
        + SAMPLE_HEADER_BYTES * n_stored_samples
        + PATTERN_HEADER_BYTES * n_patterns
    )


def cost_lower_bound(
    n_rows: int,
    n_cells: int,
    *,
    n_stored_samples: int,
    pcm_frames: int,
    bits_per_frame: int = DEFAULT_BITS_PER_FRAME,
) -> int:
    """A floor on the bytes an Impulse Tracker module of this shape occupies, from counts alone.

    Takes the row and present-cell totals of a whole song rather than an assignment, so it costs nothing
    to evaluate and can never reject a candidate that would have fit.
    """
    n_patterns = IT_BINDING.pattern_count(n_rows)
    pattern_floor = n_rows + 2 * n_cells
    return (
        pattern_floor
        + _pcm_bytes(n_stored_samples, pcm_frames, bits_per_frame)
        + _record_bytes(n_stored_samples, n_patterns)
    )
