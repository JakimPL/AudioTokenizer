from __future__ import annotations

import pytest

from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.point import EnvelopePoint
from trackmod.core.envelopes.span import EnvelopeSpan

POINTS = (EnvelopePoint(tick=0, value=64), EnvelopePoint(tick=10, value=0))


def test_envelope_ticks_must_advance() -> None:
    with pytest.raises(ValueError):
        Envelope(points=(EnvelopePoint(tick=0, value=64), EnvelopePoint(tick=0, value=0)))


def test_an_envelope_span_past_its_points_is_rejected() -> None:
    with pytest.raises(ValueError):
        Envelope(points=POINTS, loop=EnvelopeSpan(begin=0, end=2))


def test_an_envelope_reports_its_breakpoint_count() -> None:
    assert Envelope(points=POINTS, sustain=EnvelopeSpan(begin=1, end=1)).length == 2


def test_an_envelope_needs_at_least_one_point() -> None:
    with pytest.raises(ValueError):
        Envelope(points=())


def test_a_backwards_span_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnvelopeSpan(begin=2, end=1)


def test_a_negative_tick_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnvelopePoint(tick=-1, value=0)
