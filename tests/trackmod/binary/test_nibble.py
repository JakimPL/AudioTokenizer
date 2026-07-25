from __future__ import annotations

import pytest

from trackmod.binary.nibble import join_nibbles, split_nibbles
from trackmod.spec.width import BYTE_MAX, NIBBLE_MAX


@pytest.mark.parametrize("value", [0x00, 0x0D, 0xD3, BYTE_MAX])
def test_a_byte_round_trips_through_its_nibbles(value: int) -> None:
    assert join_nibbles(*split_nibbles(value)) == value


def test_the_high_nibble_leads() -> None:
    assert split_nibbles(0xD3) == (0xD, 0x3)
    assert join_nibbles(0xD, 0x3) == 0xD3


@pytest.mark.parametrize("value", [-1, BYTE_MAX + 1])
def test_splitting_something_wider_than_a_byte_raises(value: int) -> None:
    with pytest.raises(ValueError):
        split_nibbles(value)


@pytest.mark.parametrize("nibbles", [(NIBBLE_MAX + 1, 0), (0, -1)])
def test_joining_something_wider_than_a_nibble_raises(nibbles: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        join_nibbles(*nibbles)
