from __future__ import annotations

import pytest

from tests.trackmod.conftest import GridShape, random_pattern, rescaled
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.song import Song
from trackmod.it.limits import it_limits
from trackmod.it.module import ITModule
from trackmod.it.spec.ranges import CANONICAL_MAX_CHANNELS, EXTENDED_MAX_CHANNELS, MAX_ROWS
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.limits.error import LimitError
from trackmod.limits.severity import Severity
from trackmod.spec.width import BYTE_MAX


@pytest.mark.parametrize("compliance", list(Compliance))
def test_this_format_has_no_headroom_above_its_one_byte_tempo(compliance: Compliance) -> None:
    # The header stores the tempo in a single byte, so no compliance level reaches past it.
    assert it_limits(compliance).bound(Capability.TEMPO).maximum == BYTE_MAX


def test_the_channel_count_is_the_one_capability_with_headroom() -> None:
    canonical = it_limits(Compliance.CANONICAL).bound(Capability.CHANNELS)
    extended = it_limits(Compliance.EXTENDED).bound(Capability.CHANNELS)
    assert canonical.maximum == CANONICAL_MAX_CHANNELS
    assert extended.maximum == EXTENDED_MAX_CHANNELS


def test_extra_channels_are_a_compliance_violation_the_extended_level_allows(song: Song) -> None:
    wide = rescaled(song, 96)
    canonical = ITModule.from_song(wide, compliance=Compliance.CANONICAL).violations()
    assert [violation.capability for violation in canonical] == [Capability.CHANNELS]
    assert canonical[0].severity is Severity.COMPLIANCE
    assert ITModule.from_song(wide, compliance=Compliance.EXTENDED).violations() == ()


def test_writing_a_module_the_format_refuses_raises(song: Song) -> None:
    over = rescaled(song, 200)
    with pytest.raises(LimitError) as error:
        ITModule.from_song(over, compliance=Compliance.EXTENDED).to_bytes()

    assert error.value.violations[0].severity is Severity.STRUCTURAL


def test_a_short_pattern_is_a_compliance_violation_the_extended_level_allows(song: Song) -> None:
    short = Song(
        name=song.name,
        channels=song.channels,
        patterns=(random_pattern(GridShape(rows=8, channels=song.channels, instruments=2, seed=5)),),
        order=OrderList(entries=(0,)),
        instruments=song.instruments,
        samples=song.samples,
        playback=song.playback,
    )
    canonical = ITModule.from_song(short, compliance=Compliance.CANONICAL).violations()
    assert [violation.capability for violation in canonical] == [Capability.PATTERN_ROWS]
    assert ITModule.from_song(short, compliance=Compliance.EXTENDED).violations() == ()


def test_a_pattern_over_the_size_field_is_reported(song: Song) -> None:
    crowded = random_pattern(GridShape(rows=MAX_ROWS, channels=127, instruments=8, seed=99))
    wide = Song(
        name=song.name,
        channels=127,
        patterns=(crowded,),
        order=OrderList(entries=(0,)),
        instruments=song.instruments,
        samples=song.samples,
        playback=song.playback,
    )
    reported = [
        violation.capability for violation in ITModule.from_song(wide, compliance=Compliance.EXTENDED).violations()
    ]
    assert Capability.PATTERN_BYTES in reported
