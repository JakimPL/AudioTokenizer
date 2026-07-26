from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.core.patterns.column import Column
from trackmod.spec.grid import EMPTY

from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.patterns import dictionary_patterns, pattern_columns, restated
from audiotokenizer.module.routing import Routing
from audiotokenizer.module.xm import XM_BINDING

ROUTING = Routing(samples_per_instrument=4, first_key=24)


def grids(volumes: list[list[int]], slots: list[list[int]]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return np.asarray(slots, dtype=np.int64), np.asarray(volumes, dtype=np.int64)


def test_a_cell_is_present_exactly_where_its_volume_is_positive() -> None:
    sample_no, volume = grids([[0, 64], [32, 0]], [[1, 2], [3, 0]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=True)
    present = np.asarray([[False, True], [True, False]])
    assert np.array_equal(columns[Column.NOTE] != EMPTY, present)
    assert np.array_equal(columns[Column.VOLUME] != EMPTY, present)


def test_the_note_selects_the_slot_and_the_instrument_owns_it() -> None:
    # Slot 6 of a 4-samples-per-instrument routing is instrument 1, key first_key + 2.
    sample_no, volume = grids([[64]], [[6]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=True)
    assert columns[Column.NOTE][0, 0] == ROUTING.first_key + 2
    assert columns[Column.INSTRUMENT][0, 0] == 1


def test_the_volume_column_carries_the_coefficient_unchanged() -> None:
    sample_no, volume = grids([[17, 64]], [[0, 5]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=True)
    assert columns[Column.VOLUME][0, 0] == 17
    assert columns[Column.VOLUME][0, 1] == 64


def test_the_effect_columns_are_left_empty() -> None:
    sample_no, volume = grids([[64]], [[0]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=True)
    assert np.all(columns[Column.EFFECT] == EMPTY)
    assert np.all(columns[Column.PARAMETER] == EMPTY)


def test_restated_marks_a_channels_first_cell_and_every_change_after_it() -> None:
    instrument = np.asarray([[0], [0], [1], [1], [0]], dtype=np.int64)
    present = np.asarray([[True], [True], [True], [True], [True]])
    assert np.array_equal(restated(instrument, present), np.asarray([[True], [False], [True], [False], [True]]))


def test_restated_compares_across_the_rows_a_channel_sits_out() -> None:
    # A channel remembers what it plays through its silent rows, so an unchanged instrument two rows later
    # is still unchanged — comparing row-adjacent cells instead would restate it needlessly.
    instrument = np.asarray([[0], [0], [0]], dtype=np.int64)
    present = np.asarray([[True], [False], [True]])
    assert np.array_equal(restated(instrument, present), np.asarray([[True], [False], [False]]))


def test_restated_leaves_a_silent_channel_alone() -> None:
    instrument = np.zeros((3, 2), dtype=np.int64)
    present = np.asarray([[True, False], [True, False], [True, False]])
    marks = restated(instrument, present)
    assert not marks[:, 1].any()


def test_restating_names_the_instrument_on_every_played_cell() -> None:
    # This is what Impulse Tracker needs: it re-applies a sample's volume only at a note that states an
    # instrument, and this codec keeps each atom's gain there — so a cell that stayed silent about it
    # would sound the new atom's waveform at the previous atom's gain.
    sample_no, volume = grids([[64], [64], [64]], [[0], [0], [1]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=True)
    assert np.array_equal(columns[Column.INSTRUMENT] != EMPTY, np.asarray([[True], [True], [True]]))


def test_not_restating_names_it_only_where_the_channel_is_not_carrying_it() -> None:
    # FastTracker 2 bakes the gain into the waveform, so there is nothing for a note to re-apply and the
    # column is left to mean "keep playing the one you have".
    sample_no, volume = grids([[64], [64], [64]], [[0], [0], [4]])
    columns = pattern_columns(sample_no, volume, ROUTING, restate=False)
    assert np.array_equal(columns[Column.INSTRUMENT] != EMPTY, np.asarray([[True], [False], [True]]))


def test_the_two_policies_agree_on_every_column_but_the_instrument() -> None:
    rng = np.random.default_rng(3)
    sample_no = rng.integers(0, 12, size=(20, 5))
    volume = rng.integers(0, 65, size=(20, 5))
    restating = pattern_columns(sample_no, volume, ROUTING, restate=True)
    lazy = pattern_columns(sample_no, volume, ROUTING, restate=False)
    for column in (Column.NOTE, Column.VOLUME, Column.EFFECT, Column.PARAMETER):
        assert np.array_equal(restating[column], lazy[column])
    # And the lazy one only ever states a subset of what the restating one does.
    assert np.all((lazy[Column.INSTRUMENT] != EMPTY) <= (restating[Column.INSTRUMENT] != EMPTY))


def test_each_format_asks_for_the_policy_its_gain_lives_under() -> None:
    assert IT_BINDING.restates_instrument is True
    assert XM_BINDING.restates_instrument is False


def test_each_pattern_is_built_from_its_own_rows_alone() -> None:
    # The reuse memory a channel keeps runs to the pattern boundary, so a pattern that inherited the
    # previous one's state would drop an instrument the tracker has already forgotten.
    sample_no, volume = grids([[64], [64], [64], [64]], [[0], [0], [0], [0]])
    patterns = dictionary_patterns(sample_no, volume, ROUTING, slices=[(0, 2), (2, 4)], restate=False)
    assert len(patterns) == 2
    assert all(pattern.instrument[0, 0] != EMPTY for pattern in patterns)  # each states on its own first row
    assert patterns[1].instrument[1, 0] == EMPTY
