from enum import StrEnum, unique
from typing import Final

from trackmod.core.samples.depth import BitDepth

STORED_DTYPES: Final[dict[BitDepth, str]] = {BitDepth.EIGHT: "<i1", BitDepth.SIXTEEN: "<i2"}


@unique
class PcmEncoding(StrEnum):
    """How a format lays a waveform's frames out on disk.

    Absolute storage writes each frame's amplitude; delta storage writes successive differences and the
    player integrates them with a running sum in the stored width.
    """

    ABSOLUTE = "absolute"
    DELTA = "delta"


def dtype_for(depth: BitDepth) -> str:
    """The little-endian signed numpy dtype one frame is stored in."""
    return STORED_DTYPES[depth]
