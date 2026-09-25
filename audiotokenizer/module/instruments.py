"""The instruments a dictionary becomes: pure routers from keys onto stored samples.

Every stored sample plays at its own recorded rate whichever key triggers it, so an instrument here
transposes nothing and carries no envelope — a pattern cell's note selects which atom to play, not a
pitch. One instrument covers as many samples as its format's keymap routes, and the dictionary runs over
as many instruments as that takes.
"""

from __future__ import annotations

from typing import Final

from trackmod.core.instruments.instrument import Instrument

from audiotokenizer.module.routing import Routing

NAME_PREFIX: Final = "dict"


def dictionary_instruments(routing: Routing, *, slots: int) -> tuple[Instrument, ...]:
    """One instrument per block of stored samples the routing reaches through a single keymap."""
    return tuple(
        Instrument(name=f"{NAME_PREFIX}{index}", keymap=routing.keymap(index, slots=slots))
        for index in range(routing.instruments(slots))
    )
