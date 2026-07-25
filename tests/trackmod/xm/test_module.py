from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.trackmod.xm.conftest import keyed
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.point import EnvelopePoint
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.loop import Loop
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.song import Song
from trackmod.limits.compliance import Compliance
from trackmod.spec.pitch import RATE_NOTE
from trackmod.xm.instruments.grouping import song_groups
from trackmod.xm.module import XMModule
from trackmod.xm.patterns.sizing import packed_bytes
from trackmod.xm.settings import XMSettings
from trackmod.xm.spec.identity import MAGIC
from trackmod.xm.spec.sizes import EMPTY_INSTRUMENT_HEADER_BYTES, INSTRUMENT_HEADER_BYTES
from trackmod.xm.tuning import tuning_for


def module(song: Song, *, compliance: Compliance = Compliance.EXTENDED) -> XMModule:
    return XMModule.from_song(song, compliance=compliance)


def test_the_file_opens_with_the_format_tag(xm_song: Song) -> None:
    assert module(xm_song).to_bytes().startswith(MAGIC)


def test_the_size_model_agrees_with_the_written_file(xm_song: Song) -> None:
    assert module(xm_song).size().total == len(module(xm_song).to_bytes())


def test_the_size_report_names_the_largest_pattern(xm_song: Song) -> None:
    report = module(xm_song).size()
    assert report.largest_pattern == max(packed_bytes(pattern) for pattern in xm_song.patterns)
    assert report.total == report.patterns + report.pcm + report.headers


def test_a_written_module_parses_back_to_the_same_song(xm_song: Song) -> None:
    recovered = XMModule.parse(module(xm_song).to_bytes()).song
    assert recovered.name == xm_song.name
    assert recovered.channels == xm_song.channels
    assert recovered.playback == xm_song.playback
    assert recovered.order == xm_song.order
    assert recovered.patterns == xm_song.patterns
    assert recovered.instruments == xm_song.instruments
    assert recovered.samples == xm_song.samples


def test_writing_a_parsed_module_reproduces_its_bytes(xm_song: Song) -> None:
    data = module(xm_song).to_bytes()
    assert XMModule.parse(data).to_bytes() == data


def test_sample_waveforms_survive_the_delta_encoding(xm_song: Song) -> None:
    recovered = XMModule.parse(module(xm_song).to_bytes()).song
    for original, restored in zip(xm_song.samples, recovered.samples):
        assert np.allclose(restored.pcm, original.pcm, atol=1.5 / original.depth.scale)


def test_settings_survive_a_round_trip(xm_song: Song) -> None:
    settings = XMSettings(tracker="probe")
    written = XMModule(song=xm_song, compliance=Compliance.EXTENDED, settings=settings).to_bytes()
    assert XMModule.parse(written).settings == settings


def test_a_written_module_can_be_saved_and_loaded(tmp_path: Path, xm_song: Song) -> None:
    path = tmp_path / f"song{module(xm_song).extension}"
    module(xm_song).save(path)
    assert XMModule.load(path).song.name == xm_song.name


def test_parsing_something_that_is_not_a_module_raises() -> None:
    with pytest.raises(ValueError):
        XMModule.parse(b"not a module" + bytes(512))


def test_an_instrument_owning_nothing_is_written_in_the_short_form(xm_song: Song) -> None:
    placeholder = Instrument(name="silent", keymap=(None,) * len(xm_song.instruments[0].keymap))
    song = Song(
        name=xm_song.name,
        channels=xm_song.channels,
        patterns=xm_song.patterns,
        order=xm_song.order,
        instruments=(placeholder, *xm_song.instruments),
        samples=xm_song.samples,
        playback=xm_song.playback,
    )
    written = module(song).to_bytes()
    assert len(written) == len(module(xm_song).to_bytes()) + EMPTY_INSTRUMENT_HEADER_BYTES
    assert EMPTY_INSTRUMENT_HEADER_BYTES < INSTRUMENT_HEADER_BYTES
    assert XMModule.parse(written).song.instruments[0].samples == ()


def test_an_instrument_shares_no_sample_table_so_a_shared_sample_is_written_twice(xm_song: Song) -> None:
    sample = xm_song.samples[0]
    song = Song(
        name="shared",
        channels=xm_song.channels,
        patterns=xm_song.patterns,
        order=xm_song.order,
        instruments=(Instrument(name="a", keymap=keyed(0)), Instrument(name="b", keymap=keyed(0))),
        samples=(sample,),
        playback=xm_song.playback,
    )
    assert [group.length for group in song_groups(song)] == [1, 1]
    assert module(song).size().pcm == 2 * sample.stored_bytes


def test_a_sustain_loop_this_format_cannot_store_is_refused(xm_song: Song) -> None:
    looped = xm_song.samples[0].model_copy(update={"sustain_loop": Loop(begin=0, end=8)})
    song = replace_samples(xm_song, (looped, *xm_song.samples[1:]))
    with pytest.raises(ValueError, match="sustain loop"):
        module(song).to_bytes()


def test_a_pitch_envelope_this_format_cannot_store_is_refused(xm_song: Song) -> None:
    shaped = xm_song.instruments[0].model_copy(
        update={"pitch_envelope": Envelope(points=(EnvelopePoint(tick=0, value=0),))}
    )
    song = replace_instruments(xm_song, (shaped, *xm_song.instruments[1:]))
    with pytest.raises(ValueError, match="pitch envelope"):
        module(song).to_bytes()


def test_an_envelope_sustaining_over_a_span_is_refused(xm_song: Song) -> None:
    over = Envelope(
        points=(EnvelopePoint(tick=0, value=64), EnvelopePoint(tick=8, value=32), EnvelopePoint(tick=16, value=0)),
        sustain=EnvelopeSpan(begin=0, end=1),
    )
    shaped = xm_song.instruments[0].model_copy(update={"volume_envelope": over})
    song = replace_instruments(xm_song, (shaped, *xm_song.instruments[1:]))
    with pytest.raises(ValueError, match="one point"):
        module(song).to_bytes()


def test_a_keymap_transposing_one_key_differently_from_another_is_refused(xm_song: Song) -> None:
    uneven = Instrument(
        name="uneven",
        keymap=routed_keymap(
            {
                Note(60): KeyAssignment(sample=0, note=Note(60)),
                Note(61): KeyAssignment(sample=0, note=Note(72)),
            }
        ),
    )
    song = replace_instruments(xm_song, (uneven,))
    with pytest.raises(ValueError, match="transposes"):
        module(song).to_bytes()


def test_a_keymap_transposing_every_key_the_same_way_is_stored_once(xm_song: Song) -> None:
    shifted = Instrument(
        name="shifted",
        keymap=routed_keymap(
            {
                Note(60): KeyAssignment(sample=0, note=Note(72)),
                Note(61): KeyAssignment(sample=0, note=Note(73)),
            }
        ),
    )
    song = replace_instruments(xm_song, (shifted,))
    group = song_groups(song)[0]
    assert group.length == 1
    assert group.tunings[0].relative_note == 12 + tuning_of(xm_song, sample=0)


def tuning_of(song: Song, *, sample: int) -> int:
    reference = Note(RATE_NOTE)
    return tuning_for(song.samples[sample].rate, key=reference, sounded=reference).relative_note


def replace_samples(song: Song, samples: tuple[Sample, ...]) -> Song:
    return Song(
        name=song.name,
        channels=song.channels,
        patterns=song.patterns,
        order=song.order,
        instruments=song.instruments,
        samples=samples,
        playback=song.playback,
    )


def replace_instruments(song: Song, instruments: tuple[Instrument, ...]) -> Song:
    return Song(
        name=song.name,
        channels=song.channels,
        patterns=song.patterns,
        order=song.order,
        instruments=instruments,
        samples=song.samples,
        playback=song.playback,
    )
