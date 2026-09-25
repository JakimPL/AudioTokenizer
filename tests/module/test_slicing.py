from __future__ import annotations

import pytest
from trackmod.trackers.it.spec.ranges import MAX_ROWS as IT_MAX_ROWS
from trackmod.trackers.xm.spec.ranges import CANONICAL_MAX_PATTERNS as XM_MAX_PATTERNS
from trackmod.trackers.xm.spec.ranges import MAX_ROWS as XM_MAX_ROWS

from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.slicing import pattern_slices
from audiotokenizer.module.xm import XM_BINDING


@pytest.mark.parametrize("rows", [1, 5, 31, 32, 200, 405, 530, 4096])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 256])
def test_slices_tile_the_rows_exactly_once(rows: int, count: int) -> None:
    slices = pattern_slices(rows, count=count)
    assert slices[0][0] == 0
    assert slices[-1][1] == rows
    assert all(stop == nxt for (_, stop), (nxt, _) in zip(slices, slices[1:]))
    assert sum(stop - start for start, stop in slices) == rows


@pytest.mark.parametrize("rows", [1, 5, 405, 530, 4096])
@pytest.mark.parametrize("count", [1, 3, 7, 256])
def test_no_slice_is_a_stub(rows: int, count: int) -> None:
    # Spreading the rows evenly rather than filling each pattern to the ceiling is what keeps the last
    # pattern off a format's row floor: 405 rows over 3 patterns is 135 each, not 200/200/5.
    heights = [stop - start for start, stop in pattern_slices(rows, count=count)]
    assert max(heights) - min(heights) <= 1
    assert min(heights) >= 1


def test_more_patterns_than_rows_is_clamped_rather_than_producing_empty_ones() -> None:
    assert pattern_slices(3, count=10) == [(0, 1), (1, 2), (2, 3)]


def test_no_rows_is_no_patterns() -> None:
    assert pattern_slices(0, count=4) == []


@pytest.mark.parametrize("rows", [1, 200, 201, 530, 20000])
def test_it_slices_stay_within_the_row_ceiling(rows: int) -> None:
    # IT keeps per-channel memory that resets at a pattern boundary, so it uses as few patterns as its
    # 200-row ceiling allows — but never one taller than that ceiling.
    slices = pattern_slices(rows, count=IT_BINDING.pattern_count(rows))
    assert max(stop - start for start, stop in slices) <= IT_MAX_ROWS


@pytest.mark.parametrize("rows", [1, 256, 257, 530, 20000, XM_MAX_PATTERNS * XM_MAX_ROWS])
def test_xm_slices_stay_within_both_of_its_ceilings(rows: int) -> None:
    # XM keeps no memory, so it splits as far as the order table allows: shorter patterns, further from
    # the 16-bit length field a wide song would otherwise hit.
    slices = pattern_slices(rows, count=XM_BINDING.pattern_count(rows))
    assert len(slices) <= XM_MAX_PATTERNS
    assert max(stop - start for start, stop in slices) <= XM_MAX_ROWS
