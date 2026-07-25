from pydantic import BaseModel

from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Speed, Tempo


class Playback(BaseModel):
    """The clock a song starts on: ``speed`` ticks per row at ``tempo`` beats per minute.

    Both tracker formats share this clock, so the pair fixes how long a row lasts before any effect
    changes it mid-song.
    """

    model_config = FROZEN

    speed: Speed
    tempo: Tempo
