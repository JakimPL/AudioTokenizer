from trackmod.core.songs.song import Song
from trackmod.it.patterns.sizing import packed_bytes
from trackmod.it.spec.orders import ORDER_TERMINATOR_BYTES
from trackmod.it.spec.sizes import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    OFFSET_TABLE_ENTRY_BYTES,
    PATTERN_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
)
from trackmod.module.size import SizeReport


def module_bytes(song: Song) -> SizeReport:
    """How many bytes a song occupies once written as an Impulse Tracker module."""
    per_pattern = [packed_bytes(pattern) for pattern in song.patterns]
    patterns = sum(per_pattern) + PATTERN_HEADER_BYTES * len(song.patterns)
    pcm = sum(sample.stored_bytes for sample in song.samples)
    offsets = OFFSET_TABLE_ENTRY_BYTES * (len(song.instruments) + len(song.samples) + len(song.patterns))
    headers = (
        FILE_HEADER_BYTES
        + song.order.length
        + ORDER_TERMINATOR_BYTES
        + offsets
        + INSTRUMENT_HEADER_BYTES * len(song.instruments)
        + SAMPLE_HEADER_BYTES * len(song.samples)
    )
    return SizeReport(
        patterns=patterns,
        pcm=pcm,
        headers=headers,
        largest_pattern=max(per_pattern, default=0),
    )
