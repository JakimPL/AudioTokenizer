from __future__ import annotations

import numpy as np
import pytest
from trackmod.core.samples.depth import BitDepth
from trackmod.core.songs.playback import Playback
from trackmod.limits.compliance import Compliance
from trackmod.spec.grid import EMPTY

from audiotokenizer.coding.assignment import Assignment
from audiotokenizer.coding.quantisation import StoredAtoms
from audiotokenizer.module.binding import Binding
from audiotokenizer.module.catalog import BINDINGS, binding_for
from audiotokenizer.module.format import Format
from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.samples import POLARITIES
from audiotokenizer.module.song import build_song
from audiotokenizer.module.xm import XM_BINDING

RATE = 44100
ROWS = 300
CHANNELS = 6
N_ATOMS = 5


@pytest.fixture
def stored() -> StoredAtoms:
    rng = np.random.default_rng(0)
    pcm = rng.uniform(-1.0, 1.0, (N_ATOMS, 64))
    return StoredAtoms(pcm=pcm, global_volume=np.full(N_ATOMS, 48), scale=1.0)


@pytest.fixture
def assignment() -> Assignment:
    rng = np.random.default_rng(1)
    volume = rng.integers(0, 65, size=(ROWS, CHANNELS)).astype(np.int64)
    sample_no = rng.integers(0, POLARITIES * N_ATOMS, size=(ROWS, CHANNELS)).astype(np.int64)
    return Assignment(sample_no=sample_no, volume=volume)


def song_for(binding: Binding, stored: StoredAtoms, assignment: Assignment):
    return build_song(
        binding,
        stored,
        assignment,
        name="probe",
        playback=Playback(speed=1, tempo=125),
        rate=RATE,
        depth=BitDepth.EIGHT,
    )


def test_the_catalog_covers_every_format_exactly_once() -> None:
    assert set(BINDINGS) == set(Format)
    assert binding_for(Format.IT) is IT_BINDING
    assert binding_for(Format.XM) is XM_BINDING
    assert all(binding_for(module_format).format is module_format for module_format in Format)


@pytest.mark.parametrize("binding", [IT_BINDING, XM_BINDING])
def test_the_song_holds_every_row_and_the_grid_width(
    binding: Binding, stored: StoredAtoms, assignment: Assignment
) -> None:
    song = song_for(binding, stored, assignment)
    assert song.channels == CHANNELS
    assert sum(pattern.rows for pattern in song.patterns) == ROWS
    assert song.order.entries == tuple(range(len(song.patterns)))
    assert len(song.samples) == POLARITIES * N_ATOMS


@pytest.mark.parametrize("binding", [IT_BINDING, XM_BINDING])
def test_every_note_the_song_plays_resolves_to_a_stored_sample(
    binding: Binding, stored: StoredAtoms, assignment: Assignment
) -> None:
    # The routing, the instruments and the note column have to agree, or a cell plays silence — or worse,
    # the wrong atom. Walking the song the way a tracker does is the only check that covers all three.
    song = song_for(binding, stored, assignment)
    for pattern in song.patterns:
        rows, channels = np.nonzero(pattern.note != EMPTY)
        for row, channel in zip(rows.tolist(), channels.tolist()):
            cell = pattern.cell(row, channel)
            assert cell.instrument is not None or not binding.restates_instrument
            instrument = song.instruments[cell.instrument if cell.instrument is not None else 0]
            assert instrument.assignment(cell.note) is not None


@pytest.mark.parametrize("binding", [IT_BINDING, XM_BINDING])
def test_the_song_each_binding_produces_writes_as_a_module(
    binding: Binding, stored: StoredAtoms, assignment: Assignment
) -> None:
    module = binding.module(song_for(binding, stored, assignment), compliance=Compliance.CANONICAL)
    assert module.violations() == ()
    assert module.size().total == len(module.to_bytes())


def test_the_two_bindings_cut_the_same_song_into_different_pattern_counts(
    stored: StoredAtoms, assignment: Assignment
) -> None:
    # IT resets its per-channel memory at each boundary, so it fills patterns to the ceiling; XM keeps no
    # memory and splits as far as its order table allows.
    it_song = song_for(IT_BINDING, stored, assignment)
    xm_song = song_for(XM_BINDING, stored, assignment)
    assert len(it_song.patterns) < len(xm_song.patterns)
    assert sum(p.rows for p in it_song.patterns) == sum(p.rows for p in xm_song.patterns) == ROWS
