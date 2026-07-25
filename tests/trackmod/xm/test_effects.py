from __future__ import annotations

import pytest

from trackmod.binary.nibble import split_nibbles
from trackmod.core.effects.catalog import EffectCatalog
from trackmod.spec.width import NIBBLE_MAX
from trackmod.xm.effects.catalog import XM_EFFECTS
from trackmod.xm.effects.command import XMEffect, XMExtended
from trackmod.xm.spec.ranges import CANONICAL_MAX_SPEED, CANONICAL_MIN_TEMPO, MAX_BREAK_ROW

CATALOG: EffectCatalog = XM_EFFECTS


def test_the_catalog_answers_every_effect_the_protocol_declares() -> None:
    declared = {name for name in vars(EffectCatalog) if not name.startswith("_")}
    assert declared
    assert declared <= {name for name in dir(XM_EFFECTS) if not name.startswith("_")}


def test_commands_are_numbered_before_they_are_lettered() -> None:
    assert XMEffect.ARPEGGIO.letter == "0"
    assert XMEffect.SET_SPEED.letter == "F"
    assert XMEffect.GLOBAL_VOLUME.letter == "G"
    assert XMEffect.KEY_OFF.letter == "K"
    assert XMEffect.EXTRA_FINE_PORTAMENTO.letter == "X"


def test_speed_and_tempo_share_one_command_split_by_where_the_parameter_falls() -> None:
    speed, tempo = CATALOG.set_speed(6), CATALOG.set_tempo(125)
    assert speed.command == tempo.command == XMEffect.SET_SPEED
    assert speed.parameter < CANONICAL_MIN_TEMPO <= tempo.parameter


def test_a_speed_that_would_read_back_as_a_tempo_is_refused() -> None:
    with pytest.raises(ValueError):
        CATALOG.set_speed(CANONICAL_MAX_SPEED + 1)


def test_a_tempo_the_one_byte_parameter_cannot_carry_is_refused() -> None:
    # The header word reaches far past this, which is exactly the headroom the effect column lacks.
    with pytest.raises(ValueError):
        CATALOG.set_tempo(441)


def test_a_pattern_break_is_read_a_decimal_digit_to_a_nibble() -> None:
    effect = CATALOG.pattern_break(16)
    assert effect.command == XMEffect.PATTERN_BREAK
    assert effect.parameter == 0x16


def test_a_pattern_break_past_two_decimal_digits_is_refused() -> None:
    with pytest.raises(ValueError):
        CATALOG.pattern_break(MAX_BREAK_ROW + 1)


def test_an_extended_effect_packs_its_sub_command_into_the_high_nibble() -> None:
    effect = CATALOG.note_delay(3)
    assert effect.command == XMEffect.EXTENDED
    assert split_nibbles(effect.parameter) == (XMExtended.NOTE_DELAY, 3)


def test_a_delay_wider_than_a_nibble_is_refused() -> None:
    with pytest.raises(ValueError):
        CATALOG.note_delay(NIBBLE_MAX + 1)


def test_a_volume_slide_runs_one_way_only() -> None:
    assert split_nibbles(CATALOG.volume_slide(up=4, down=0).parameter) == (4, 0)
    assert split_nibbles(CATALOG.volume_slide(up=0, down=4).parameter) == (0, 4)
    with pytest.raises(ValueError):
        CATALOG.volume_slide(up=4, down=4)
