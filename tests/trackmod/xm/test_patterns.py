from __future__ import annotations

import pytest

from tests.trackmod.xm.conftest import xm_pattern
from trackmod.core.effects.effect import Effect
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.xm.note import stored_note
from trackmod.xm.patterns.packer import pack_cells
from trackmod.xm.patterns.parser import unpack_cells
from trackmod.xm.patterns.sizing import packed_bytes
from trackmod.xm.spec.cells import PACKED_BYTE, RAW_CELL_BYTES, VOLUME_COLUMN_BASE, CellMask
from trackmod.xm.spec.ranges import MAX_ROWS

GRIDS = (
    (16, 4, 2, 1),
    (64, 12, 3, 2),
    (MAX_ROWS, 16, 4, 3),
    (1, 1, 1, 4),
)


@pytest.mark.parametrize("rows,channels,instruments,seed", GRIDS, ids=lambda value: str(value))
def test_the_size_model_agrees_with_the_packer_byte_for_byte(
    rows: int, channels: int, instruments: int, seed: int
) -> None:
    pattern = xm_pattern(rows=rows, channels=channels, instruments=instruments, seed=seed)
    assert packed_bytes(pattern) == len(pack_cells(pattern))


@pytest.mark.parametrize("rows,channels,instruments,seed", GRIDS, ids=lambda value: str(value))
def test_a_packed_pattern_unpacks_to_the_same_grid(rows: int, channels: int, instruments: int, seed: int) -> None:
    pattern = xm_pattern(rows=rows, channels=channels, instruments=instruments, seed=seed)
    assert unpack_cells(pack_cells(pattern), rows=rows, channels=channels) == pattern


def test_a_silent_channel_still_costs_the_byte_that_holds_its_place() -> None:
    # There is no row terminator and no way to skip a channel, so a player reads exactly one cell per
    # grid position and an empty one is the bare mask.
    empty = Pattern.empty(rows=8, channels=16)
    stream = pack_cells(empty)
    assert len(stream) == PACKED_BYTE * empty.rows * empty.channels
    assert set(stream) == {int(CellMask.PACKED)}


def test_a_cell_stating_every_column_drops_its_mask() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=Note(60), instrument=0, volume=32, effect=Effect(command=15, parameter=6)))
    stream = pack_cells(builder.build())
    assert len(stream) == RAW_CELL_BYTES
    assert stream[0] == stored_note(60)
    assert not stream[0] & CellMask.PACKED


def test_a_partial_cell_pays_a_mask_and_a_byte_for_each_column_it_states() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=Note(60), volume=32))
    stream = pack_cells(builder.build())
    assert stream[0] == CellMask.PACKED | CellMask.NOTE | CellMask.VOLUME
    assert stream[1] == stored_note(60)
    assert stream[2] == VOLUME_COLUMN_BASE + 32
    assert len(stream) == 3


def test_the_format_keeps_no_memory_so_a_repeated_cell_costs_the_same_every_row() -> None:
    builder = PatternBuilder(rows=4, channels=1)
    for row in range(4):
        builder.place(row, 0, Cell(note=Note(60), instrument=0, volume=32))

    assert len(pack_cells(builder.build())) == 4 * (1 + 3)


def test_a_key_off_survives_a_round_trip() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=NoteCommand.OFF))
    recovered = unpack_cells(pack_cells(builder.build()), rows=1, channels=1)
    assert recovered.cell(0, 0) == Cell(note=NoteCommand.OFF)


@pytest.mark.parametrize("command", [NoteCommand.CUT, NoteCommand.FADE])
def test_a_note_command_the_column_cannot_state_is_refused(command: NoteCommand) -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=command))
    with pytest.raises(ValueError):
        pack_cells(builder.build())


def test_a_key_above_the_octaves_this_format_numbers_is_refused() -> None:
    builder = PatternBuilder(rows=1, channels=1)
    builder.place(0, 0, Cell(note=Note(NOTE_COUNT - 1)))
    with pytest.raises(ValueError):
        pack_cells(builder.build())


def test_a_volume_column_effect_is_dropped_rather_than_read_as_a_level() -> None:
    # 0x60 opens the column's own slide range, which the shared model has no cell to hold.
    recovered = unpack_cells(bytes([CellMask.PACKED | CellMask.VOLUME, 0x60]), rows=1, channels=1)
    assert recovered.cell(0, 0) == Cell()
