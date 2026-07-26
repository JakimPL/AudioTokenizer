from __future__ import annotations

import pytest
from trackmod.spec.pitch import RATE_NOTE

from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.routing import Routing
from audiotokenizer.module.xm import XM_BINDING

BINDINGS = (IT_BINDING, XM_BINDING)


def test_a_slot_is_reached_through_exactly_one_instrument_and_key() -> None:
    routing = Routing(samples_per_instrument=16, first_key=24)
    assert (routing.instrument(0), routing.key(0).value) == (0, 24)
    assert (routing.instrument(15), routing.key(15).value) == (0, 39)
    assert (routing.instrument(16), routing.key(16).value) == (1, 24)  # the next block restarts at first_key


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
@pytest.mark.parametrize("slots", [1, 2, 15, 16, 17, 120, 121, 255])
def test_every_slot_is_routed_once_and_only_once(routing: Routing, slots: int) -> None:
    # A dictionary is only playable if each stored sample answers to a distinct (instrument, key) pair —
    # a collision would silently make one atom unreachable and play another in its place.
    reached = {}
    for instrument in range(routing.instruments(slots)):
        for key, assignment in enumerate(routing.keymap(instrument, slots=slots)):
            if assignment is not None:
                assert (instrument, key) not in reached
                reached[(instrument, key)] = assignment.sample
    assert sorted(reached.values()) == list(range(slots))


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
def test_a_keymap_transposes_nothing(routing: Routing) -> None:
    # Every key sounds its sample at the sample's own recorded rate, which is what makes a note column a
    # selector rather than a pitch.
    keymap = routing.keymap(0, slots=routing.samples_per_instrument)
    assert all(assignment is None or assignment.note.value == RATE_NOTE for assignment in keymap)


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
def test_instrument_count_covers_the_slots_without_an_empty_tail(routing: Routing) -> None:
    per = routing.samples_per_instrument
    assert routing.instruments(0) == 1  # a dictionary always has somewhere to live
    assert routing.instruments(per) == 1
    assert routing.instruments(per + 1) == 2


def test_the_two_formats_route_differently_and_that_is_the_point() -> None:
    # One instrument covers the whole 120-key keyboard in IT and 16 keys in XM, which is why the routing
    # is a configured model rather than a constant.
    assert IT_BINDING.routing.samples_per_instrument == 120
    assert IT_BINDING.routing.first_key == 0
    assert XM_BINDING.routing.samples_per_instrument == 16
    assert XM_BINDING.routing.first_key == 24


def test_an_xm_instrument_block_stays_inside_the_keys_the_format_numbers() -> None:
    # The block of 16 has to sit above the transposition headroom and below the 96 keys XM writes.
    routing = XM_BINDING.routing
    assert routing.key(routing.samples_per_instrument - 1).value < 96
