from __future__ import annotations

import pytest

from tests.trackmod.conftest import GridShape, random_pattern
from trackmod.core.effects.effect import Effect
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.it.patterns.packer import pack_cells, stored_instrument
from trackmod.it.patterns.parser import unpack_cells
from trackmod.it.patterns.sizing import packed_bytes
from trackmod.it.spec.cells import NO_INSTRUMENT, CellMask
from trackmod.it.spec.ranges import MAX_ROWS

GRIDS = (
    GridShape(rows=16, channels=4, instruments=2, seed=1),
    GridShape(rows=64, channels=12, instruments=3, seed=2),
    GridShape(rows=MAX_ROWS, channels=32, instruments=4, seed=3),
    GridShape(rows=1, channels=1, instruments=1, seed=4),
)


@pytest.mark.parametrize("shape", GRIDS, ids=lambda shape: f"{shape.rows}x{shape.channels}")
def test_the_size_model_agrees_with_the_packer_byte_for_byte(shape: GridShape) -> None:
    pattern = random_pattern(shape)
    assert packed_bytes(pattern) == len(pack_cells(pattern))


def test_a_channel_holding_steady_settles_to_one_byte_a_row() -> None:
    # Row 0 states all three columns, row 1 switches to the all-reuse mask, and from row 2 on the
    # channel byte alone carries the cell: mask, note, instrument and volume all come from memory.
    builder = PatternBuilder(rows=4, channels=1)
    for row in range(4):
        builder.place(row, 0, Cell(note=Note(60), instrument=0, volume=32))

    stream = pack_cells(builder.build())
    stated = 1 + 1 + 3
    switched = 1 + 1
    settled = 1
    assert len(stream) == stated + switched + 2 * settled + 4


def test_a_silent_channel_costs_nothing() -> None:
    empty = Pattern.empty(rows=8, channels=16)
    assert len(pack_cells(empty)) == empty.rows


def test_a_changed_volume_spends_a_byte_and_keeps_the_mask() -> None:
    builder = PatternBuilder(rows=2, channels=1)
    builder.place(0, 0, Cell(note=Note(60), instrument=0, volume=32))
    builder.place(1, 0, Cell(note=Note(60), instrument=0, volume=48))
    stream = pack_cells(builder.build())
    # The second row restates only its volume, so its mask differs from the first and is spent again.
    assert stream[-3] == CellMask.LAST_NOTE | CellMask.LAST_INSTRUMENT | CellMask.VOLUME


@pytest.mark.parametrize("shape", GRIDS, ids=lambda shape: f"{shape.rows}x{shape.channels}")
def test_a_packed_pattern_unpacks_to_the_same_grid(shape: GridShape) -> None:
    pattern = random_pattern(shape)
    recovered = unpack_cells(pack_cells(pattern), rows=pattern.rows).widened(pattern.channels)
    assert recovered == pattern


def test_a_note_command_survives_a_round_trip() -> None:
    builder = PatternBuilder(rows=2, channels=1)
    builder.place(0, 0, Cell(note=Note(60), instrument=0, volume=64))
    builder.place(1, 0, Cell(note=NoteCommand.OFF))
    recovered = unpack_cells(pack_cells(builder.build()), rows=2)
    assert recovered.cell(1, 0) == Cell(note=NoteCommand.OFF)


def test_an_effect_survives_a_round_trip() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(effect=Effect(command=20, parameter=200)))
    recovered = unpack_cells(pack_cells(builder.build()), rows=1)
    assert recovered.cell(0, 0) == Cell(effect=Effect(command=20, parameter=200))


def test_a_column_left_out_of_a_cell_does_not_erase_what_the_channel_remembers() -> None:
    # Row 1 states nothing but a note, so row 2 must still be able to reuse row 0's volume.
    builder = PatternBuilder(rows=3, channels=1)
    builder.place(0, 0, Cell(note=Note(60), volume=32))
    builder.place(1, 0, Cell(note=Note(62)))
    builder.place(2, 0, Cell(note=Note(62), volume=32))
    pattern = builder.build()
    recovered = unpack_cells(pack_cells(pattern), rows=3)
    assert recovered.cell(1, 0) == Cell(note=Note(62))
    assert recovered.cell(2, 0) == Cell(note=Note(62), volume=32)


def test_a_stream_reusing_a_mask_that_was_never_stated_is_rejected() -> None:
    with pytest.raises(ValueError):
        unpack_cells(bytes([0x01, 0x00]), rows=1)


def test_the_first_instrument_is_stored_above_the_byte_that_means_none() -> None:
    # Zero is what a cell writes to stay on the instrument the channel already carries, so a stored
    # instrument number starts at one — writing the shared index straight through silences the cell.
    assert stored_instrument(0) == NO_INSTRUMENT + 1
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=Note(60), instrument=0, volume=64))
    assert NO_INSTRUMENT not in pack_cells(builder.build())[:-1]


def test_an_instrument_survives_a_round_trip() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=Note(60), instrument=0, volume=64))
    recovered = unpack_cells(pack_cells(builder.build()), rows=1)
    assert recovered.cell(0, 0).instrument == 0
