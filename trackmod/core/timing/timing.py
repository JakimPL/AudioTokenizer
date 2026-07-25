from pydantic import BaseModel

from trackmod.schema.config import FROZEN
from trackmod.schema.scalars import Frames, Speed, Tempo


class Timing(BaseModel):
    """A row length realised exactly: ``speed`` ticks at ``tempo`` span whole ``row_frames`` frames."""

    model_config = FROZEN

    speed: Speed
    tempo: Tempo
    row_frames: Frames
