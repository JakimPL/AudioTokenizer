from typing import Final

from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_PANNING
from trackmod.spec.width import BYTE_MAX, NIBBLE_MAX
from trackmod.xm.spec.ranges import (
    CANONICAL_MAX_SPEED,
    CANONICAL_MAX_TEMPO,
    CANONICAL_MIN_TEMPO,
    MAX_BREAK_ROW,
    MIN_SPEED,
)

SPEED_PARAMETER: Final = Bound(minimum=MIN_SPEED, maximum=CANONICAL_MAX_SPEED)
TEMPO_PARAMETER: Final = Bound(minimum=CANONICAL_MIN_TEMPO, maximum=CANONICAL_MAX_TEMPO)
ORDER_PARAMETER: Final = Bound(minimum=0, maximum=BYTE_MAX)
ROW_PARAMETER: Final = Bound(minimum=0, maximum=MAX_BREAK_ROW)
NIBBLE_PARAMETER: Final = Bound(minimum=0, maximum=NIBBLE_MAX)
PANNING_PARAMETER: Final = Bound(minimum=0, maximum=MAX_PANNING)
