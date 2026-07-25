from typing import Final

from trackmod.it.spec.ranges import MAX_ROWS, MAX_SPEED, MAX_TEMPO, MIN_SPEED, MIN_TEMPO
from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_PANNING
from trackmod.spec.width import BYTE_MAX, NIBBLE_MAX

SPEED_PARAMETER: Final = Bound(minimum=MIN_SPEED, maximum=MAX_SPEED)
TEMPO_PARAMETER: Final = Bound(minimum=MIN_TEMPO, maximum=MAX_TEMPO)
ORDER_PARAMETER: Final = Bound(minimum=0, maximum=BYTE_MAX)
ROW_PARAMETER: Final = Bound(minimum=0, maximum=MAX_ROWS - 1)
NIBBLE_PARAMETER: Final = Bound(minimum=0, maximum=NIBBLE_MAX)
PANNING_PARAMETER: Final = Bound(minimum=0, maximum=MAX_PANNING)
