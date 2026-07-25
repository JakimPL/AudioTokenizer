from typing import Annotated

from pydantic import Field

from trackmod.spec.clock import MIN_SPEED, MIN_TEMPO
from trackmod.spec.grid import MIN_CHANNELS, MIN_ROWS
from trackmod.spec.levels import (
    MAX_INSTRUMENT_VOLUME,
    MAX_PANNING,
    MAX_VOLUME,
    MIN_FADEOUT,
    MIN_INSTRUMENT_VOLUME,
    MIN_PANNING,
    MIN_VOLUME,
)

Index = Annotated[int, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]
Offset = Annotated[int, Field(ge=0)]
Tick = Annotated[int, Field(ge=0)]
Frames = Annotated[int, Field(ge=0)]
Rate = Annotated[int, Field(gt=0)]

Volume = Annotated[int, Field(ge=MIN_VOLUME, le=MAX_VOLUME)]
Panning = Annotated[int, Field(ge=MIN_PANNING, le=MAX_PANNING)]
InstrumentVolume = Annotated[int, Field(ge=MIN_INSTRUMENT_VOLUME, le=MAX_INSTRUMENT_VOLUME)]
Fadeout = Annotated[int, Field(ge=MIN_FADEOUT)]

Speed = Annotated[int, Field(ge=MIN_SPEED)]
Tempo = Annotated[int, Field(ge=MIN_TEMPO)]

Rows = Annotated[int, Field(ge=MIN_ROWS)]
Channels = Annotated[int, Field(ge=MIN_CHANNELS)]
