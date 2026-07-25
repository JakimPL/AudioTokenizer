from __future__ import annotations

import pytest

from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.core.envelopes.point import EnvelopePoint
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, pitched_keymap, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.spec.levels import MAX_INSTRUMENT_VOLUME


def test_a_pitched_keymap_answers_every_key_at_its_own_pitch() -> None:
    keymap = pitched_keymap(sample=2)
    assert all(assignment is not None for assignment in keymap)
    assert keymap[60] == KeyAssignment(sample=2, note=Note(60))


def test_a_routed_keymap_leaves_unnamed_keys_silent() -> None:
    reference = Note(60)
    keymap = routed_keymap({Note(10): KeyAssignment(sample=7, note=reference)})
    instrument = Instrument(name="router", keymap=keymap)
    assert instrument.assignment(Note(10)) == KeyAssignment(sample=7, note=reference)
    assert instrument.assignment(Note(11)) is None
    assert instrument.samples == (7,)


def test_reachable_samples_are_listed_once_in_key_order() -> None:
    keymap = routed_keymap(
        {
            Note(5): KeyAssignment(sample=3, note=Note(60)),
            Note(1): KeyAssignment(sample=1, note=Note(60)),
            Note(9): KeyAssignment(sample=3, note=Note(60)),
        }
    )
    assert Instrument(name="router", keymap=keymap).samples == (1, 3)


def test_a_keymap_that_does_not_cover_the_keyboard_is_rejected() -> None:
    with pytest.raises(ValueError):
        Instrument(name="short", keymap=(None, None))


def test_a_level_past_the_instrument_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        Instrument(name="loud", keymap=pitched_keymap(sample=0), global_volume=MAX_INSTRUMENT_VOLUME + 1)


def test_an_instrument_with_no_envelopes_reports_none_for_every_kind() -> None:
    instrument = Instrument(name="lead", keymap=pitched_keymap(sample=0))
    assert all(instrument.envelope(kind) is None for kind in EnvelopeKind)


def test_an_envelope_is_reached_by_its_kind() -> None:
    curve = Envelope(points=(EnvelopePoint(tick=0, value=64),))
    instrument = Instrument(name="lead", keymap=pitched_keymap(sample=0), panning_envelope=curve)
    assert instrument.envelope(EnvelopeKind.PANNING) == curve
    assert instrument.envelope(EnvelopeKind.VOLUME) is None
