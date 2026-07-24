"""Assemble the complete ``.XM`` byte stream: the file header, order table, patterns and instruments.

:class:`XMModule` bundles the samples, instruments, patterns and order list; :func:`write_xm_module` lays
them out per the XM layout — the 80-byte header, the 256-byte order table, the pattern blocks, then each
instrument followed by all its sample headers and all its sample data — and :func:`write_xm` writes the
result to disk. Unlike IT, XM stores no offset tables: sections are found by walking their sizes, so this
is a straight concatenation. Every pattern shares the header's single channel count, so all pattern grids
must be the same width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audiotokenizer.xm.format import FILE_HEADER, encode_name
from audiotokenizer.xm.instruments import XMInstrument, instrument_header
from audiotokenizer.xm.patterns import XMPattern, XMPlayback, pack_pattern
from audiotokenizer.xm.samples import XMSample, pcm_bytes, sample_header
from audiotokenizer.xm.spec import (
    FLAG_LINEAR_FREQUENCY,
    HEADER_SIZE_FIELD,
    MAGIC,
    MODULE_NAME_BYTES,
    ORDER_TABLE_BYTES,
    STRIPPED_BYTE,
    TRACKER_NAME,
    TRACKER_NAME_BYTES,
    VERSION,
)


@dataclass(frozen=True)
class XMModule:
    """A complete module ready to serialize.

    ``samples`` is flat in slot order; instrument ``i`` owns the next ``instruments[i].n_samples`` of them,
    so ``sum(n_samples)`` must equal ``len(samples)``. Every pattern grid is ``channels`` wide.
    """

    name: str
    samples: tuple[XMSample, ...]
    instruments: tuple[XMInstrument, ...]
    patterns: tuple[XMPattern, ...]
    orders: tuple[int, ...]
    playback: XMPlayback
    channels: int


def _file_header(module: XMModule) -> bytes:
    """Serialize the 80-byte header prefix (magic .. bpm); the order table is appended after it."""
    playback = module.playback
    return FILE_HEADER.pack(
        {
            "magic": MAGIC,
            "name": encode_name(module.name, MODULE_NAME_BYTES),
            "stripped": STRIPPED_BYTE,
            "tracker": encode_name(TRACKER_NAME, TRACKER_NAME_BYTES),
            "version": VERSION,
            "header_size": HEADER_SIZE_FIELD,
            "song_length": len(module.orders),
            "restart_position": 0,
            "channels": module.channels,
            "pattern_count": len(module.patterns),
            "instrument_count": len(module.instruments),
            "flags": FLAG_LINEAR_FREQUENCY,
            "speed": playback.speed,
            "bpm": playback.tempo,
        }
    )


def _order_table(orders: tuple[int, ...]) -> bytes:
    """The 256-byte order table: one pattern index per played order, zero-padded to full width."""
    if len(orders) > ORDER_TABLE_BYTES:
        raise ValueError(f"{len(orders)} orders exceed the {ORDER_TABLE_BYTES}-entry XM order table")
    return bytes(order & 0xFF for order in orders) + bytes(ORDER_TABLE_BYTES - len(orders))


def _instrument_block(instrument: XMInstrument, group: tuple[XMSample, ...]) -> bytes:
    """One instrument: its header, then all its sample headers, then all its sample data (XM's order)."""
    headers = b"".join(sample_header(sample) for sample in group)
    data = b"".join(pcm_bytes(sample) for sample in group)
    return instrument_header(instrument) + headers + data


def write_xm_module(module: XMModule) -> bytes:
    """Serialize ``module`` to the complete bytes of an ``.XM`` file."""
    if sum(instrument.n_samples for instrument in module.instruments) != len(module.samples):
        raise ValueError("instrument sample counts do not sum to the number of samples")

    out = bytearray(_file_header(module))
    out += _order_table(module.orders)
    for pattern in module.patterns:
        if pattern.n_channels != module.channels:
            raise ValueError(f"pattern width {pattern.n_channels} differs from module channels {module.channels}")
        out += pack_pattern(pattern)

    offset = 0
    for instrument in module.instruments:
        group = module.samples[offset : offset + instrument.n_samples]
        offset += instrument.n_samples
        out += _instrument_block(instrument, group)
    return bytes(out)


def write_xm(path: Path | str, module: XMModule) -> None:
    """Write ``module`` to ``path`` as an ``.XM`` file."""
    Path(path).write_bytes(write_xm_module(module))
