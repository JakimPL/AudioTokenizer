from trackmod.spec.width import BYTE_MAX, NIBBLE_BITS, NIBBLE_MAX


def split_nibbles(value: int) -> tuple[int, int]:
    """Split a byte into its high and low nibbles.

    Raises:
        ValueError: when ``value`` does not fit in a byte.
    """
    if not 0 <= value <= BYTE_MAX:
        raise ValueError(f"{value} does not fit in a byte")

    return value >> NIBBLE_BITS, value & NIBBLE_MAX


def join_nibbles(high: int, low: int) -> int:
    """Combine two nibbles into a byte.

    Raises:
        ValueError: when either nibble exceeds four bits.
    """
    if not 0 <= high <= NIBBLE_MAX or not 0 <= low <= NIBBLE_MAX:
        raise ValueError(f"nibbles {high}, {low} exceed four bits")

    return (high << NIBBLE_BITS) | low
