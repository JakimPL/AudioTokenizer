"""Assemble the complete ``.IT`` byte stream: the file header, offset tables and record bodies.

:class:`ITModule` bundles the samples, instruments, patterns and order list; :func:`write_it_module`
lays them out per the IT layout (file header, order list, three offset tables, then the record bodies
and PCM) and :func:`write_it` writes the result to disk. Sample headers carry a pointer to their PCM,
which lands last, so PCM offsets are resolved before the sample headers are built.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from audiotokenizer.it.format import FILE_HEADER, encode_name
from audiotokenizer.it.instruments import ITInstrument, instrument_header
from audiotokenizer.it.patterns import ITPattern, ITPlayback, pack_pattern
from audiotokenizer.it.samples import ITSample, pcm_bytes, sample_header
from audiotokenizer.it.spec import (
    CHANNEL_VOLUME_FULL,
    CHANNELS_STORED,
    CMWT,
    CWT,
    FILE_HEADER_BYTES,
    FLAG_LINEAR_SLIDES,
    FLAG_USE_INSTRUMENTS,
    MAGIC_MODULE,
    MAX_GLOBAL_VOLUME,
    MAX_MIX_VOLUME,
    NAME_BYTES,
    OFFSET_TABLE_ENTRY_BYTES,
    ORDER_TERMINATOR,
    PAN_CENTER,
    PANNING_SEPARATION,
    SAMPLE_HEADER_BYTES,
)


@dataclass(frozen=True)
class ITModule:
    """A complete module ready to serialize."""

    name: str
    samples: tuple[ITSample, ...]
    instruments: tuple[ITInstrument, ...]
    patterns: tuple[ITPattern, ...]
    orders: tuple[int, ...]
    playback: ITPlayback


@dataclass(frozen=True)
class SectionCounts:
    """Entry counts for the four IT sections, named so the offset arithmetic reads structurally.

    ``orders`` counts the order-list bytes (including the ``0xFF`` terminator); the other three count
    records whose file offsets are stored as u32 entries in the three offset tables.
    """

    orders: int
    instruments: int
    samples: int
    patterns: int

    @property
    def offset_table_bytes(self) -> int:
        """Bytes the instrument, sample and pattern offset tables occupy together."""
        return OFFSET_TABLE_ENTRY_BYTES * (self.instruments + self.samples + self.patterns)


def _file_header(module: ITModule, counts: SectionCounts) -> bytes:
    """Serialize the 192-byte IMPM header (up to and including the channel pan/volume arrays)."""
    playback = module.playback
    return FILE_HEADER.pack(
        {
            "magic": MAGIC_MODULE,
            "name": encode_name(module.name, NAME_BYTES),
            "highlight": 0,
            "order_count": counts.orders,
            "instrument_count": counts.instruments,
            "sample_count": counts.samples,
            "pattern_count": counts.patterns,
            "created_with": CWT,
            "compatible_with": CMWT,
            "flags": FLAG_USE_INSTRUMENTS | FLAG_LINEAR_SLIDES,
            "global_volume": min(playback.global_volume, MAX_GLOBAL_VOLUME),
            "mix_volume": min(playback.mix_volume, MAX_MIX_VOLUME),
            "speed": playback.speed,
            "tempo": playback.tempo,
            "panning_separation": PANNING_SEPARATION,
            "channel_pan": bytes([PAN_CENTER]) * CHANNELS_STORED,
            "channel_volume": bytes([CHANNEL_VOLUME_FULL]) * CHANNELS_STORED,
        }
    )


def _offsets(blobs: list[bytes], start: int) -> list[int]:
    """The file offset each blob occupies when laid end to end from ``start``."""
    result = []
    position = start
    for blob in blobs:
        result.append(position)
        position += len(blob)
    return result


def _serialize_body(module: ITModule, start: int) -> tuple[list[int], bytes]:
    """Serialize everything after the offset tables; return the tables (instr+sample+pattern) and body.

    Sample headers carry a pointer to their PCM, which lands after the patterns, so PCM offsets are
    resolved first and the sample headers built against them.
    """
    instrument_blobs = [instrument_header(instrument) for instrument in module.instruments]
    pattern_blobs = [pack_pattern(pattern) for pattern in module.patterns]
    pcm_blobs = [pcm_bytes(sample) for sample in module.samples]

    samples_at = start + sum(len(blob) for blob in instrument_blobs)
    patterns_at = samples_at + SAMPLE_HEADER_BYTES * len(module.samples)
    data_at = patterns_at + sum(len(blob) for blob in pattern_blobs)

    data_offsets = _offsets(pcm_blobs, data_at)
    sample_blobs = [sample_header(sample, offset) for sample, offset in zip(module.samples, data_offsets)]

    tables = (
        _offsets(instrument_blobs, start) + _offsets(sample_blobs, samples_at) + _offsets(pattern_blobs, patterns_at)
    )
    body = b"".join(instrument_blobs + sample_blobs + pattern_blobs + pcm_blobs)
    return tables, body


def write_it_module(module: ITModule) -> bytes:
    """Serialize ``module`` to the complete bytes of an uncompressed ``.IT`` file."""
    orders = tuple(module.orders) + (ORDER_TERMINATOR,)
    counts = SectionCounts(
        orders=len(orders),
        instruments=len(module.instruments),
        samples=len(module.samples),
        patterns=len(module.patterns),
    )
    table_end = FILE_HEADER_BYTES + counts.orders + counts.offset_table_bytes
    tables, body = _serialize_body(module, table_end)

    out = bytearray(_file_header(module, counts))
    out += bytes(orders)
    for offset in tables:
        out += struct.pack("<I", offset)
    out += body
    return bytes(out)


def write_it(path: Path | str, module: ITModule) -> None:
    """Write ``module`` to ``path`` as an uncompressed ``.IT`` file."""
    Path(path).write_bytes(write_it_module(module))
