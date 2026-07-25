from trackmod.limits.bound import Bound


def require_range(value: int, *, bound: Bound, subject: str) -> int:
    """Return ``value`` when it lies within ``bound``.

    Raises:
        ValueError: when the value leaves the range a format leaves for it.
    """
    if not bound.contains(value):
        raise ValueError(f"{subject} {value} is outside {bound}")

    return value
