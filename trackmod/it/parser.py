import struct

from trackmod.binary.cursor import Cursor
from trackmod.binary.records.values import read_bytes, read_int
from trackmod.binary.text import decode_name
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.patterns.grid import Pattern
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.it.instruments.parser import parse_instrument
from trackmod.it.layout.file import FILE_HEADER
from trackmod.it.layout.instrument import INSTRUMENT_HEADER
from trackmod.it.layout.sample import SAMPLE_HEADER
from trackmod.it.patterns.parser import unpack_pattern
from trackmod.it.samples.parser import parse_sample, stored_frames
from trackmod.it.settings import ITSettings
from trackmod.it.spec.flags import HeaderFlag
from trackmod.it.spec.identity import MAGIC_MODULE
from trackmod.it.spec.orders import ORDER_SEPARATOR, ORDER_TERMINATOR
from trackmod.it.spec.sizes import OFFSET_TABLE_ENTRY_BYTES


class ModuleReader:
    """Walks an Impulse Tracker file, following the three offset tables that head it."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        cursor = Cursor(data)
        self._header = cursor.read(FILE_HEADER)
        if read_bytes(self._header, "magic") != MAGIC_MODULE:
            raise ValueError("data does not open with the Impulse Tracker module tag")

        self._order = self._read_order(cursor)
        self._instrument_offsets = self._read_table(cursor, "instrument_count")
        self._sample_offsets = self._read_table(cursor, "sample_count")
        self._pattern_offsets = self._read_table(cursor, "pattern_count")

    def song(self) -> Song:
        """The format-agnostic content the file carries."""
        patterns = self._patterns()
        channels = max((pattern.channels for pattern in patterns), default=1)
        return Song(
            name=decode_name(read_bytes(self._header, "name")),
            channels=channels,
            patterns=tuple(pattern.widened(channels) for pattern in patterns),
            order=self._order,
            instruments=self._instruments(),
            samples=self._samples(),
            playback=Playback(speed=read_int(self._header, "speed"), tempo=read_int(self._header, "tempo")),
        )

    def settings(self) -> ITSettings:
        """The song-wide values this format adds, as the header states them."""
        return ITSettings(
            global_volume=read_int(self._header, "global_volume"),
            mix_volume=read_int(self._header, "mix_volume"),
            panning_separation=read_int(self._header, "panning_separation"),
            channel_panning=tuple(read_bytes(self._header, "channel_pan")),
            channel_volume=tuple(read_bytes(self._header, "channel_volume")),
            flags=HeaderFlag(read_int(self._header, "flags")),
        )

    def _read_order(self, cursor: Cursor) -> OrderList:
        raw = cursor.take(read_int(self._header, "order_count"))
        return OrderList(entries=tuple(entry for entry in raw if entry not in (ORDER_SEPARATOR, ORDER_TERMINATOR)))

    def _read_table(self, cursor: Cursor, count_field: str) -> tuple[int, ...]:
        count = read_int(self._header, count_field)
        return struct.unpack(f"<{count}I", cursor.take(OFFSET_TABLE_ENTRY_BYTES * count))

    def _instruments(self) -> tuple[Instrument, ...]:
        return tuple(
            parse_instrument(INSTRUMENT_HEADER.unpack(self._data[offset:])) for offset in self._instrument_offsets
        )

    def _samples(self) -> tuple[Sample, ...]:
        samples = []
        for offset in self._sample_offsets:
            values = SAMPLE_HEADER.unpack(self._data[offset:])
            start = read_int(values, "sample_pointer")
            samples.append(parse_sample(values, self._data[start : start + stored_frames(values)]))

        return tuple(samples)

    def _patterns(self) -> tuple[Pattern, ...]:
        patterns = []
        for offset in self._pattern_offsets:
            cursor = Cursor(self._data)
            cursor.seek(offset)
            patterns.append(unpack_pattern(cursor))

        return tuple(patterns)
