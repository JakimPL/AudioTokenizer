from __future__ import annotations

import pytest

from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import RATE_NOTE
from trackmod.xm.spec.tuning import FINETUNE_UNITS, REFERENCE_NOTE, REFERENCE_RATE
from trackmod.xm.tuning import Tuning, pitch_units, tuned_rate, tuning_for, unit_rate

RATES = (8363, 11025, 16000, 22050, 32000, 44100, 48000)


def test_the_reference_key_needs_no_transposition_at_all() -> None:
    assert pitch_units(REFERENCE_RATE) == FINETUNE_UNITS * REFERENCE_NOTE
    assert unit_rate(FINETUNE_UNITS * REFERENCE_NOTE) == REFERENCE_RATE


def test_an_octave_up_doubles_the_rate() -> None:
    octave = pitch_units(REFERENCE_RATE) + FINETUNE_UNITS * 12
    assert unit_rate(octave) == 2 * REFERENCE_RATE


def test_a_sample_played_on_the_key_it_is_tuned_to_needs_no_relative_note() -> None:
    reference = Note(RATE_NOTE)
    tuning = tuning_for(REFERENCE_RATE, key=reference, sounded=reference)
    assert tuning == Tuning(relative_note=REFERENCE_NOTE - RATE_NOTE, finetune=0)


@pytest.mark.parametrize("slot", range(16))
def test_the_derivation_reproduces_the_tuning_this_library_replaced(slot: int) -> None:
    # The exporter this package grew out of put native 44100 Hz on a fixed key per stored slot, with
    # `relative_note = 77 - play_note` against a one-based note and a finetune of 100 throughout.
    play_note = 25 + slot
    tuning = tuning_for(44100, key=Note(play_note - 1), sounded=Note(RATE_NOTE))
    assert tuning == Tuning(relative_note=77 - play_note, finetune=100)


@pytest.mark.parametrize("rate", RATES)
def test_a_derived_tuning_reads_back_as_the_rate_it_was_derived_from(rate: int) -> None:
    reference = Note(RATE_NOTE)
    recovered = tuned_rate(tuning_for(rate, key=reference, sounded=reference), key=reference, sounded=reference)
    assert recovered == pytest.approx(rate, rel=1 / FINETUNE_UNITS / 12)


@pytest.mark.parametrize("rate", RATES)
def test_a_rate_read_back_off_the_lattice_is_a_fixed_point(rate: int) -> None:
    reference = Note(RATE_NOTE)
    once = tuned_rate(tuning_for(rate, key=reference, sounded=reference), key=reference, sounded=reference)
    twice = tuned_rate(tuning_for(once, key=reference, sounded=reference), key=reference, sounded=reference)
    assert twice == once


def test_the_lattice_cannot_store_the_usual_recording_rate_exactly() -> None:
    # There is no explicit rate field: 44100 Hz falls between two finetune steps, which is a real loss
    # against Impulse Tracker and the reason a round trip moves the rate by a few hertz.
    reference = Note(RATE_NOTE)
    stored = tuned_rate(tuning_for(44100, key=reference, sounded=reference), key=reference, sounded=reference)
    assert stored != 44100
    assert abs(stored - 44100) / 44100 < 1e-3


def test_the_finetune_trim_is_never_spent_on_a_whole_semitone() -> None:
    assert tuning_for(REFERENCE_RATE, key=Note(0), sounded=Note(0)).finetune == 0


@pytest.mark.parametrize("key", [0, 24, 60, 95])
def test_a_key_sounding_its_own_pitch_tunes_the_same_wherever_it_sits(key: int) -> None:
    tuning = tuning_for(44100, key=Note(key), sounded=Note(key))
    assert tuned_rate(tuning, key=Note(key), sounded=Note(key)) == tuned_rate(
        tuning_for(44100, key=Note(RATE_NOTE), sounded=Note(RATE_NOTE)),
        key=Note(RATE_NOTE),
        sounded=Note(RATE_NOTE),
    )


def test_a_rate_the_relative_note_byte_cannot_reach_is_refused() -> None:
    with pytest.raises(ValueError):
        tuning_for(1, key=Note(95), sounded=Note(0))
