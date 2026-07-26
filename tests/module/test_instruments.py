from __future__ import annotations

import pytest

from audiotokenizer.module.instruments import NAME_PREFIX, dictionary_instruments
from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.routing import Routing
from audiotokenizer.module.xm import XM_BINDING

BINDINGS = (IT_BINDING, XM_BINDING)


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
@pytest.mark.parametrize("slots", [1, 16, 17, 120, 121, 255])
def test_one_instrument_per_block_the_routing_reaches(routing: Routing, slots: int) -> None:
    instruments = dictionary_instruments(routing, slots=slots)
    assert len(instruments) == routing.instruments(slots)
    assert [instrument.name for instrument in instruments[:2]] == [f"{NAME_PREFIX}0", f"{NAME_PREFIX}1"][
        : len(instruments)
    ]


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
@pytest.mark.parametrize("slots", [1, 16, 17, 120, 121, 255])
def test_the_instruments_between_them_reach_every_stored_sample(routing: Routing, slots: int) -> None:
    reached = sorted(
        assignment.sample
        for instrument in dictionary_instruments(routing, slots=slots)
        for assignment in instrument.keymap
        if assignment is not None
    )
    assert reached == list(range(slots))


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
def test_the_last_instrument_stops_at_the_dictionary_rather_than_naming_a_missing_sample(routing: Routing) -> None:
    # A keymap that ran to the end of its block would point past the sample list, which a song refuses.
    slots = routing.samples_per_instrument + 1
    last = dictionary_instruments(routing, slots=slots)[-1]
    assert [assignment.sample for assignment in last.keymap if assignment is not None] == [slots - 1]


@pytest.mark.parametrize("routing", [binding.routing for binding in BINDINGS])
def test_an_instrument_carries_no_envelope_because_it_only_routes(routing: Routing) -> None:
    # Every stored sample plays at its own rate for its own length; a shaped voice is not what a
    # dictionary entry is.
    instrument = dictionary_instruments(routing, slots=8)[0]
    assert instrument.volume_envelope is None
    assert instrument.panning_envelope is None
    assert instrument.pitch_envelope is None
