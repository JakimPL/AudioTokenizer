from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackmod.core.songs.song import Song
from trackmod.it.module import ITModule
from trackmod.it.patterns.sizing import packed_bytes
from trackmod.it.settings import ITSettings
from trackmod.it.spec.identity import MAGIC_INSTRUMENT, MAGIC_MODULE, MAGIC_SAMPLE
from trackmod.limits.compliance import Compliance


def module(song: Song, *, compliance: Compliance = Compliance.EXTENDED) -> ITModule:
    return ITModule.from_song(song, compliance=compliance)


def test_every_section_opens_with_its_own_tag(song: Song) -> None:
    data = module(song).to_bytes()
    assert data.startswith(MAGIC_MODULE)
    assert MAGIC_INSTRUMENT in data
    assert MAGIC_SAMPLE in data


def test_the_size_model_agrees_with_the_written_file(song: Song) -> None:
    assert module(song).size().total == len(module(song).to_bytes())


def test_the_size_report_names_the_largest_pattern(song: Song) -> None:
    report = module(song).size()
    assert report.largest_pattern == max(packed_bytes(pattern) for pattern in song.patterns)
    assert report.total == report.patterns + report.pcm + report.headers


def test_a_written_module_parses_back_to_the_same_song(song: Song) -> None:
    recovered = ITModule.parse(module(song).to_bytes()).song
    assert recovered.name == song.name
    assert recovered.channels == song.channels
    assert recovered.playback == song.playback
    assert recovered.order.entries == song.order.entries
    assert recovered.instruments == song.instruments
    assert recovered.patterns == song.patterns


def test_sample_headers_and_waveforms_survive_a_round_trip(song: Song) -> None:
    recovered = ITModule.parse(module(song).to_bytes()).song
    for original, restored in zip(song.samples, recovered.samples):
        assert restored.name == original.name
        assert restored.rate == original.rate
        assert restored.depth == original.depth
        assert restored.loop == original.loop
        assert np.allclose(restored.pcm, original.pcm, atol=1.5 / original.depth.scale)


def test_settings_survive_a_round_trip(song: Song) -> None:
    settings = ITSettings(global_volume=96, mix_volume=64)
    written = ITModule(song=song, compliance=Compliance.EXTENDED, settings=settings).to_bytes()
    assert ITModule.parse(written).settings == settings


def test_a_written_module_can_be_saved_and_loaded(tmp_path: Path, song: Song) -> None:
    path = tmp_path / f"song{module(song).extension}"
    module(song).save(path)
    assert ITModule.load(path).song.name == song.name


def test_parsing_something_that_is_not_a_module_raises() -> None:
    with pytest.raises(ValueError):
        ITModule.parse(b"not a module" + bytes(256))
