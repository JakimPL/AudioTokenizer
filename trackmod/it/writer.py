import struct

from trackmod.binary.text import encode_name
from trackmod.core.songs.song import Song
from trackmod.it.instruments.writer import instrument_header
from trackmod.it.layout.file import FILE_HEADER
from trackmod.it.patterns.packer import pack_pattern
from trackmod.it.samples.writer import sample_bytes, sample_header
from trackmod.it.settings import ITSettings
from trackmod.it.spec.identity import COMPATIBLE_WITH, CREATED_WITH, MAGIC_MODULE
from trackmod.it.spec.orders import ORDER_TERMINATOR, ORDER_TERMINATOR_BYTES
from trackmod.it.spec.sizes import (
    FILE_HEADER_BYTES,
    NAME_BYTES,
    OFFSET_CODE,
    OFFSET_TABLE_ENTRY_BYTES,
    SAMPLE_HEADER_BYTES,
)


def offsets(blobs: list[bytes], start: int) -> list[int]:
    """Where each blob lands when they are laid end to end from ``start``."""
    positions = []
    position = start
    for blob in blobs:
        positions.append(position)
        position += len(blob)

    return positions


def body_start(song: Song) -> int:
    """The offset the first instrument header sits at, past the file header and the offset tables."""
    entries = len(song.instruments) + len(song.samples) + len(song.patterns)
    return FILE_HEADER_BYTES + song.order.length + ORDER_TERMINATOR_BYTES + OFFSET_TABLE_ENTRY_BYTES * entries


def lay_out(song: Song, instruments: list[bytes], patterns: list[bytes], start: int) -> tuple[list[int], bytes]:
    """The three offset tables and the body they point into.

    Where the sample frames land is resolved before the sample headers are built, because each header
    carries a pointer to frames that sit after every other section of the file.
    """
    frames = [sample_bytes(sample) for sample in song.samples]
    samples_at = start + sum(len(blob) for blob in instruments)
    patterns_at = samples_at + SAMPLE_HEADER_BYTES * len(song.samples)
    data_at = patterns_at + sum(len(blob) for blob in patterns)

    data_offsets = offsets(frames, data_at)
    headers = [sample_header(sample, data_offset=offset) for sample, offset in zip(song.samples, data_offsets)]
    tables = offsets(instruments, start) + offsets(headers, samples_at) + offsets(patterns, patterns_at)
    return tables, b"".join(instruments + headers + patterns + frames)


def file_header(song: Song, settings: ITSettings, *, instrument_count: int, pattern_count: int) -> bytes:
    """Serialise the file header that opens the module."""
    return FILE_HEADER.pack(
        {
            "magic": MAGIC_MODULE,
            "name": encode_name(song.name, NAME_BYTES),
            "highlight": 0,
            "order_count": song.order.length + ORDER_TERMINATOR_BYTES,
            "instrument_count": instrument_count,
            "sample_count": len(song.samples),
            "pattern_count": pattern_count,
            "created_with": CREATED_WITH,
            "compatible_with": COMPATIBLE_WITH,
            "flags": int(settings.flags),
            "special": 0,
            "global_volume": settings.global_volume,
            "mix_volume": settings.mix_volume,
            "speed": song.playback.speed,
            "tempo": song.playback.tempo,
            "panning_separation": settings.panning_separation,
            "pitch_wheel_depth": 0,
            "message_length": 0,
            "message_offset": 0,
            "channel_pan": bytes(settings.channel_panning),
            "channel_volume": bytes(settings.channel_volume),
        }
    )


def write_module(song: Song, settings: ITSettings) -> bytes:
    """Serialise a song and its settings as a whole Impulse Tracker file."""
    instruments = [instrument_header(instrument) for instrument in song.instruments]
    patterns = [pack_pattern(pattern) for pattern in song.patterns]
    tables, body = lay_out(song, instruments, patterns, body_start(song))

    out = bytearray(file_header(song, settings, instrument_count=len(instruments), pattern_count=len(patterns)))
    out += bytes(song.order.entries) + bytes([ORDER_TERMINATOR])
    for offset in tables:
        out += struct.pack(OFFSET_CODE, offset)

    return bytes(out + body)
