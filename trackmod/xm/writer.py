from trackmod.binary.text import encode_name
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.song import Song
from trackmod.spec.width import BYTE_MAX
from trackmod.xm.instruments.grouping import song_groups
from trackmod.xm.instruments.writer import instrument_block
from trackmod.xm.layout.file import FILE_HEADER
from trackmod.xm.patterns.packer import pack_pattern
from trackmod.xm.settings import XMSettings
from trackmod.xm.spec.identity import MAGIC, STRIPPED_BYTE, VERSION
from trackmod.xm.spec.sizes import (
    HEADER_SIZE_FIELD,
    MODULE_NAME_BYTES,
    ORDER_TABLE_BYTES,
    TRACKER_NAME_BYTES,
)


def order_table(order: OrderList) -> bytes:
    """The order table, which this format always writes at its full width whatever the song plays."""
    entries = bytes(entry & BYTE_MAX for entry in order.entries)
    return entries + bytes(ORDER_TABLE_BYTES - len(entries))


def file_header(song: Song, settings: XMSettings) -> bytes:
    """Serialise the header that opens the module, ahead of the order table it declares the size of."""
    return FILE_HEADER.pack(
        {
            "magic": MAGIC,
            "name": encode_name(song.name, MODULE_NAME_BYTES),
            "stripped": STRIPPED_BYTE,
            "tracker": encode_name(settings.tracker, TRACKER_NAME_BYTES),
            "version": VERSION,
            "header_size": HEADER_SIZE_FIELD,
            "order_count": song.order.length,
            "restart_position": song.order.restart,
            "channels": song.channels,
            "pattern_count": len(song.patterns),
            "instrument_count": len(song.instruments),
            "flags": int(settings.flags),
            "speed": song.playback.speed,
            "tempo": song.playback.tempo,
        }
    )


def write_module(song: Song, settings: XMSettings) -> bytes:
    """Serialise a song and its settings as a whole FastTracker 2 file.

    The format keeps no offset tables: every section is found by walking the sizes of the ones before
    it, so the whole file is one concatenation in the order a reader consumes it.
    """
    out = bytearray(file_header(song, settings))
    out += order_table(song.order)
    for pattern in song.patterns:
        out += pack_pattern(pattern)

    for instrument, group in zip(song.instruments, song_groups(song)):
        out += instrument_block(instrument, group)

    return bytes(out)
