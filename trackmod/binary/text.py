from trackmod.spec.text import ENCODING, PADDING, REPLACEMENT


def encode_name(text: str, length: int) -> bytes:
    """Encode ``text`` to exactly ``length`` bytes, null-padded, truncating anything longer.

    Tracker name fields are fixed-width and a player reads the whole width, so a name is padded rather
    than terminated. Characters outside the tracker character set become ``?``.
    """
    raw = text.encode(ENCODING, errors=REPLACEMENT)[:length]
    return raw + PADDING * (length - len(raw))


def decode_name(raw: bytes) -> str:
    """Read a fixed-width name field back, dropping the padding a writer added."""
    return raw.split(PADDING, 1)[0].decode(ENCODING, errors=REPLACEMENT)
