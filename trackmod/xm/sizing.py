from trackmod.core.songs.song import Song
from trackmod.module.size import SizeReport
from trackmod.xm.instruments.grouping import song_groups
from trackmod.xm.instruments.writer import header_bytes, waveform_bytes
from trackmod.xm.patterns.sizing import packed_bytes
from trackmod.xm.spec.sizes import FILE_HEADER_BYTES, ORDER_TABLE_BYTES, PATTERN_HEADER_BYTES


def module_bytes(song: Song) -> SizeReport:
    """How many bytes a song occupies once written as a FastTracker 2 module.

    Every instrument owns its own copy of the samples its keys reach, so a sample two instruments play
    is counted twice — which is the cost this format charges for keeping no shared sample table.
    """
    groups = song_groups(song)
    per_pattern = [packed_bytes(pattern) for pattern in song.patterns]
    patterns = sum(per_pattern) + PATTERN_HEADER_BYTES * len(song.patterns)
    return SizeReport(
        patterns=patterns,
        pcm=sum(waveform_bytes(group) for group in groups),
        headers=FILE_HEADER_BYTES + ORDER_TABLE_BYTES + sum(header_bytes(group) for group in groups),
        largest_pattern=max(per_pattern, default=0),
    )
