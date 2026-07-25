from pydantic import BaseModel

from trackmod.schema.config import FROZEN
from trackmod.xm.spec.defaults import DEFAULT_FLAGS, DEFAULT_TRACKER
from trackmod.xm.spec.flags import HeaderFlag


class XMSettings(BaseModel):
    """The song-wide values this format carries that the shared model leaves to each format.

    The frequency-table flag is the one that matters to a reader: every rate this format expresses is a
    transposition read against the linear table, so a module that clears the flag is asking a player to
    interpolate its samples on a different lattice than the one their tunings were derived on.
    """

    model_config = FROZEN

    tracker: str = DEFAULT_TRACKER
    flags: HeaderFlag = DEFAULT_FLAGS
