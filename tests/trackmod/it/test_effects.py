from __future__ import annotations

import pytest

from trackmod.binary.nibble import split_nibbles
from trackmod.core.effects.catalog import EffectCatalog
from trackmod.it.effects.catalog import IT_EFFECTS
from trackmod.it.effects.command import ITEffect, ITExtended
from trackmod.spec.width import NIBBLE_MAX

CATALOG: EffectCatalog = IT_EFFECTS


def test_the_catalog_answers_every_effect_the_protocol_declares() -> None:
    declared = {name for name in vars(EffectCatalog) if not name.startswith("_")}
    assert declared
    assert declared <= {name for name in dir(IT_EFFECTS) if not name.startswith("_")}


def test_commands_are_lettered_from_a() -> None:
    assert ITEffect.SET_SPEED.letter == "A"
    assert ITEffect.SET_TEMPO.letter == "T"


def test_a_tempo_change_carries_the_tempo_in_its_parameter() -> None:
    effect = CATALOG.set_tempo(125)
    assert effect.command == ITEffect.SET_TEMPO
    assert effect.parameter == 125


def test_an_extended_effect_packs_its_sub_command_into_the_high_nibble() -> None:
    effect = CATALOG.note_delay(3)
    assert effect.command == ITEffect.EXTENDED
    assert split_nibbles(effect.parameter) == (ITExtended.NOTE_DELAY, 3)


def test_a_delay_wider_than_a_nibble_is_refused() -> None:
    with pytest.raises(ValueError):
        CATALOG.note_delay(NIBBLE_MAX + 1)


def test_a_volume_slide_runs_one_way_only() -> None:
    assert split_nibbles(CATALOG.volume_slide(up=4, down=0).parameter) == (4, 0)
    assert split_nibbles(CATALOG.volume_slide(up=0, down=4).parameter) == (0, 4)
    with pytest.raises(ValueError):
        CATALOG.volume_slide(up=4, down=4)


@pytest.mark.parametrize("tempo", [31, 256])
def test_a_tempo_this_format_cannot_reach_is_refused(tempo: int) -> None:
    with pytest.raises(ValueError):
        CATALOG.set_tempo(tempo)


def test_a_pattern_break_past_the_tallest_pattern_is_refused() -> None:
    with pytest.raises(ValueError):
        CATALOG.pattern_break(200)
