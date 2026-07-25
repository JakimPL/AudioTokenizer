from trackmod.it.spec.ranges import PAN_MAX
from trackmod.spec.levels import MAX_PANNING


def stored_panning(panning: int) -> int:
    """Scale a position on the shared 0..255 panning range onto the 0..64 range this format stores."""
    return round(panning * PAN_MAX / MAX_PANNING)


def shared_panning(stored: int) -> int:
    """Scale a stored 0..64 panning position back onto the shared 0..255 range."""
    return round(stored * MAX_PANNING / PAN_MAX)
