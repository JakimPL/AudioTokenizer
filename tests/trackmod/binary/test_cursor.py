from __future__ import annotations

import pytest

from trackmod.binary.cursor import Cursor
from trackmod.binary.records.field import Field
from trackmod.binary.records.record import Record

DATA = bytes(range(16))
WORD = Record(size=4, fields=(Field(name="value", offset=0, code="<I"),))


def test_taking_advances_the_cursor() -> None:
    cursor = Cursor(DATA)
    assert cursor.take(4) == DATA[:4]
    assert cursor.position == 4
    assert cursor.remaining == 12


def test_peeking_leaves_the_cursor_where_it_was() -> None:
    cursor = Cursor(DATA)
    cursor.skip(2)
    assert cursor.peek(3) == DATA[2:5]
    assert cursor.position == 2


def test_reading_a_record_consumes_exactly_its_size() -> None:
    cursor = Cursor(DATA)
    assert cursor.read(WORD) == {"value": 0x03020100}
    assert cursor.position == WORD.size


def test_seeking_moves_to_an_absolute_offset() -> None:
    cursor = Cursor(DATA)
    cursor.seek(len(DATA))
    assert cursor.at_end
    cursor.seek(0)
    assert cursor.remaining == len(DATA)


def test_reading_past_the_end_raises() -> None:
    cursor = Cursor(DATA)
    cursor.skip(14)
    with pytest.raises(ValueError):
        cursor.take(4)


def test_seeking_outside_the_data_raises() -> None:
    with pytest.raises(ValueError):
        Cursor(DATA).seek(len(DATA) + 1)


def test_a_negative_read_raises() -> None:
    with pytest.raises(ValueError):
        Cursor(DATA).take(-1)
