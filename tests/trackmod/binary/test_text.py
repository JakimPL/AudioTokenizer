from __future__ import annotations

import pytest

from trackmod.binary.text import decode_name, encode_name

LENGTH = 8


@pytest.mark.parametrize("name", ["", "atom", "12345678"])
def test_a_name_round_trips_through_its_fixed_width_field(name: str) -> None:
    assert decode_name(encode_name(name, LENGTH)) == name


def test_a_name_fills_its_field_and_pads_the_rest() -> None:
    assert encode_name("atom", LENGTH) == b"atom" + bytes(4)


def test_an_overlong_name_is_truncated_to_the_field() -> None:
    assert encode_name("far too long a name", LENGTH) == b"far too "


def test_characters_outside_the_tracker_set_become_placeholders() -> None:
    assert decode_name(encode_name("pianoé", LENGTH)) == "piano?"
