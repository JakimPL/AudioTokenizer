from typing import Annotated

from pydantic import BaseModel, Field

from trackmod.it.spec.defaults import (
    DEFAULT_CHANNEL_PANNING,
    DEFAULT_CHANNEL_VOLUME,
    DEFAULT_FLAGS,
    DEFAULT_GLOBAL_VOLUME,
    DEFAULT_MIX_VOLUME,
    DEFAULT_PANNING_SEPARATION,
)
from trackmod.it.spec.flags import HeaderFlag
from trackmod.it.spec.sizes import CHANNELS_STORED
from trackmod.schema.config import FROZEN
from trackmod.spec.width import BYTE_MAX

ChannelBytes = Annotated[
    tuple[Annotated[int, Field(ge=0, le=BYTE_MAX)], ...],
    Field(min_length=CHANNELS_STORED, max_length=CHANNELS_STORED),
]


class ITSettings(BaseModel):
    """The song-wide values this format carries that the shared model leaves to each format.

    The channel tables are as wide as the header stores them, which is the format's own channel count
    rather than the song's — a module with more channels than that leaves the extra ones at the defaults
    the tracker applies.
    """

    model_config = FROZEN

    global_volume: int = DEFAULT_GLOBAL_VOLUME
    mix_volume: int = DEFAULT_MIX_VOLUME
    panning_separation: int = DEFAULT_PANNING_SEPARATION
    channel_panning: ChannelBytes = DEFAULT_CHANNEL_PANNING
    channel_volume: ChannelBytes = DEFAULT_CHANNEL_VOLUME
    flags: HeaderFlag = DEFAULT_FLAGS
