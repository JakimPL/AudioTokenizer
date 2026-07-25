from __future__ import annotations

import pytest

from tests.trackmod.conftest import rescaled
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.limits.error import LimitError
from trackmod.limits.severity import Severity
from trackmod.spec.levels import MAX_VOLUME
from trackmod.spec.width import BYTE_MAX, WORD_MAX
from trackmod.xm.limits import xm_limits
from trackmod.xm.module import XMModule
from trackmod.xm.spec.ranges import (
    CANONICAL_MAX_CHANNELS,
    CANONICAL_MAX_TEMPO,
    EXTENDED_MAX_CHANNELS,
    MAX_NOTE,
)


def test_the_tempo_word_is_where_this_format_has_its_headroom() -> None:
    # The header stores the tempo in sixteen bits while the tracker honours one byte of it, which is
    # the whole reason a caller reaching for a shorter row chooses this format.
    assert xm_limits(Compliance.CANONICAL).bound(Capability.TEMPO).maximum == CANONICAL_MAX_TEMPO
    assert xm_limits(Compliance.EXTENDED).bound(Capability.TEMPO).maximum == WORD_MAX


def test_the_channel_count_has_headroom_too() -> None:
    assert xm_limits(Compliance.CANONICAL).bound(Capability.CHANNELS).maximum == CANONICAL_MAX_CHANNELS
    assert xm_limits(Compliance.EXTENDED).bound(Capability.CHANNELS).maximum == EXTENDED_MAX_CHANNELS


def test_this_format_declares_no_song_wide_volume_at_all() -> None:
    limits = xm_limits(Compliance.EXTENDED)
    for capability in (Capability.SONG_VOLUME, Capability.MIX_VOLUME):
        with pytest.raises(KeyError):
            limits.bound(capability)


def test_a_hacked_tempo_is_a_compliance_violation_the_extended_level_allows(xm_song: Song) -> None:
    fast = Song(
        name=xm_song.name,
        channels=xm_song.channels,
        patterns=xm_song.patterns,
        order=xm_song.order,
        instruments=xm_song.instruments,
        samples=xm_song.samples,
        playback=Playback(speed=1, tempo=441),
    )
    canonical = XMModule.from_song(fast, compliance=Compliance.CANONICAL).violations()
    assert [violation.capability for violation in canonical] == [Capability.TEMPO]
    assert canonical[0].severity is Severity.COMPLIANCE
    assert XMModule.from_song(fast, compliance=Compliance.EXTENDED).violations() == ()


def test_extra_channels_are_a_compliance_violation_the_extended_level_allows(xm_song: Song) -> None:
    wide = rescaled(xm_song, 64)
    canonical = XMModule.from_song(wide, compliance=Compliance.CANONICAL).violations()
    assert [violation.capability for violation in canonical] == [Capability.CHANNELS]
    assert XMModule.from_song(wide, compliance=Compliance.EXTENDED).violations() == ()


def test_writing_a_module_the_format_refuses_raises(xm_song: Song) -> None:
    over = rescaled(xm_song, BYTE_MAX + 1)
    with pytest.raises(LimitError) as error:
        XMModule.from_song(over, compliance=Compliance.EXTENDED).to_bytes()

    assert error.value.violations[0].severity is Severity.STRUCTURAL


def test_a_key_above_the_eight_octaves_this_format_numbers_is_reported(xm_song: Song) -> None:
    builder = PatternBuilder(rows=8, channels=1)
    builder.place(0, 0, Cell(note=Note(MAX_NOTE), instrument=0))
    builder.place(1, 0, Cell(note=Note(MAX_NOTE + 1), instrument=0))
    tall = Song(
        name=xm_song.name,
        channels=1,
        patterns=(builder.build(),),
        order=OrderList(entries=(0,)),
        instruments=xm_song.instruments[:1],
        samples=xm_song.samples,
        playback=xm_song.playback,
    )
    reported = [
        violation.capability for violation in XMModule.from_song(tall, compliance=Compliance.EXTENDED).violations()
    ]
    assert reported == [Capability.NOTE]


def test_a_sample_asking_for_gain_this_format_cannot_apply_is_reported(xm_song: Song) -> None:
    # There is no per-sample multiplier here, so anything below full gain has to be baked into the
    # waveform instead — which the report is what tells a caller.
    quiet = xm_song.samples[0].model_copy(update={"gain": MAX_VOLUME // 2})
    song = Song(
        name=xm_song.name,
        channels=xm_song.channels,
        patterns=xm_song.patterns,
        order=xm_song.order,
        instruments=xm_song.instruments,
        samples=(quiet, *xm_song.samples[1:]),
        playback=xm_song.playback,
    )
    reported = [
        violation.capability for violation in XMModule.from_song(song, compliance=Compliance.EXTENDED).violations()
    ]
    assert reported == [Capability.SAMPLE_GAIN]
