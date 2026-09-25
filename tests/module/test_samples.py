from __future__ import annotations

import numpy as np
import pytest
from trackmod.core.samples.depth import BitDepth
from trackmod.spec.levels import MAX_VOLUME

from audiotokenizer.coding.quantisation import StoredAtoms
from audiotokenizer.module.binding import Binding, gained
from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.samples import POLARITIES, stored_samples
from audiotokenizer.module.xm import XM_BINDING

RATE = 44100
BINDINGS = (IT_BINDING, XM_BINDING)


@pytest.fixture
def stored() -> StoredAtoms:
    """Three atoms at three different gains, so where the gain lands is visible per atom."""
    pcm = np.asarray([[1.0, 0.5, -0.5, -1.0], [0.25, -0.25, 0.25, -0.25], [0.0, 1.0, 0.0, -1.0]])
    return StoredAtoms(pcm=pcm, global_volume=np.asarray([64, 32, 16]), scale=1.0)


@pytest.mark.parametrize("binding", BINDINGS)
def test_every_atom_is_stored_twice_in_slot_order(binding: Binding, stored: StoredAtoms) -> None:
    samples = stored_samples(stored, binding, rate=RATE, depth=BitDepth.EIGHT)
    assert len(samples) == POLARITIES * len(stored.global_volume)
    assert [sample.name for sample in samples[:4]] == ["atom0+", "atom0-", "atom1+", "atom1-"]


@pytest.mark.parametrize("binding", BINDINGS)
def test_the_odd_slot_is_the_negation_of_the_even_one(binding: Binding, stored: StoredAtoms) -> None:
    # A signed coefficient is played by pointing the cell at the polarity that carries the sign, which is
    # only true if the two slots really are negatives of each other.
    samples = stored_samples(stored, binding, rate=RATE, depth=BitDepth.EIGHT)
    for index in range(0, len(samples), POLARITIES):
        assert np.allclose(samples[index].pcm, -samples[index + 1].pcm)


@pytest.mark.parametrize("binding", BINDINGS)
def test_every_sample_records_the_rate_it_was_captured_at(binding: Binding, stored: StoredAtoms) -> None:
    samples = stored_samples(stored, binding, rate=RATE, depth=BitDepth.EIGHT)
    assert all(sample.rate == RATE for sample in samples)
    assert all(sample.depth is BitDepth.EIGHT for sample in samples)


def test_it_keeps_the_gain_in_the_sample_and_leaves_the_waveform_alone(stored: StoredAtoms) -> None:
    # This format multiplies the volume column by the sample's own lattice, so the waveform stays at full
    # resolution and the gain rides beside it.
    samples = stored_samples(stored, IT_BINDING, rate=RATE, depth=BitDepth.EIGHT)
    assert [sample.gain for sample in samples] == [64, 64, 32, 32, 16, 16]
    assert np.allclose(samples[0].pcm, stored.pcm[0])
    assert np.allclose(samples[4].pcm, stored.pcm[2])  # the quietest atom is stored just as loud


def test_xm_bakes_the_gain_into_the_waveform_because_the_column_would_erase_it(stored: StoredAtoms) -> None:
    # This format's volume column *overrides* the sample volume byte, so a gain left there is gone the
    # moment a row states its coefficient — the waveform is the only place left to put it.
    samples = stored_samples(stored, XM_BINDING, rate=RATE, depth=BitDepth.EIGHT)
    assert all(sample.gain == MAX_VOLUME for sample in samples)
    assert np.allclose(samples[0].pcm, stored.pcm[0])  # gain 64 of 64 scales by one
    assert np.allclose(samples[4].pcm, stored.pcm[2] * (16 / MAX_VOLUME))  # the quietest atom pays in resolution


def test_gained_scales_off_the_lattice_it_is_told_about() -> None:
    pcm = np.asarray([1.0, -1.0, 0.5])
    assert np.allclose(gained(pcm, MAX_VOLUME, levels=MAX_VOLUME), pcm)
    assert np.allclose(gained(pcm, 32, levels=MAX_VOLUME), pcm * 0.5)
    assert np.allclose(gained(pcm, 0, levels=MAX_VOLUME), np.zeros_like(pcm))


@pytest.mark.parametrize("binding", BINDINGS)
def test_an_empty_dictionary_stores_nothing(binding: Binding) -> None:
    empty = StoredAtoms(pcm=np.zeros((0, 4)), global_volume=np.zeros(0, dtype=np.int64), scale=1.0)
    assert stored_samples(empty, binding, rate=RATE, depth=BitDepth.EIGHT) == ()
