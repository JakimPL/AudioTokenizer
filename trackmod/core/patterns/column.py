from collections.abc import Mapping
from enum import StrEnum, unique

import numpy as np
from numpy.typing import NDArray


@unique
class Column(StrEnum):
    """One of the five integer planes a pattern grid is made of."""

    NOTE = "note"
    INSTRUMENT = "instrument"
    VOLUME = "volume"
    EFFECT = "effect"
    PARAMETER = "parameter"


Columns = Mapping[Column, NDArray[np.int16]]
