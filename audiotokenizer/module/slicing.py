"""Cutting a song's rows into the patterns a tracker plays them from.

The rows are spread as evenly as they divide rather than filling each pattern to the ceiling and leaving
the remainder in a stub. A stub is what a format's row floor refuses — Impulse Tracker's canonical form
asks for at least 32 rows a pattern — so a song of 405 rows becomes three patterns of 135 instead of two
full ones and a five-row tail.
"""

from __future__ import annotations


def pattern_slices(rows: int, *, count: int) -> list[tuple[int, int]]:
    """The half-open ``[start, stop)`` row spans of ``count`` patterns covering ``rows`` rows.

    Every span is within one row of every other, and there are never more of them than there are rows.
    """
    if rows <= 0:
        return []

    spans = max(1, min(count, rows))
    height, remainder = divmod(rows, spans)
    starts = [0]
    for index in range(spans):
        starts.append(starts[index] + height + (1 if index < remainder else 0))
    return [(starts[index], starts[index + 1]) for index in range(spans)]
