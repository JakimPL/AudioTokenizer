from trackmod.binary.cursor import Cursor
from trackmod.binary.records.values import RecordValues, read_bytes, read_int
from trackmod.binary.text import decode_name
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.patterns.grid import Pattern
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.xm.instruments.parser import parse_instrument, parse_stub
from trackmod.xm.layout.file import FILE_HEADER
from trackmod.xm.layout.instrument import EMPTY_INSTRUMENT_HEADER, INSTRUMENT_HEADER
from trackmod.xm.layout.sample import SAMPLE_HEADER
from trackmod.xm.patterns.parser import unpack_pattern
from trackmod.xm.samples.parser import parse_sample, stored_bytes
from trackmod.xm.settings import XMSettings
from trackmod.xm.spec.flags import HeaderFlag
from trackmod.xm.spec.identity import MAGIC
from trackmod.xm.spec.sizes import (
    EMPTY_INSTRUMENT_HEADER_BYTES,
    FILE_HEADER_BYTES,
    HEADER_SIZE_OFFSET,
    INSTRUMENT_HEADER_BYTES,
)


class ModuleReader:
    """Walks a FastTracker 2 file, which stores no offset tables and is read strictly front to back."""

    def __init__(self, data: bytes) -> None:
        cursor = Cursor(data)
        self._header = cursor.read(FILE_HEADER)
        if read_bytes(self._header, "magic") != MAGIC:
            raise ValueError("data does not open with the FastTracker 2 module tag")

        self._channels = read_int(self._header, "channels")
        self._order = self._read_order(data)
        cursor.seek(HEADER_SIZE_OFFSET + read_int(self._header, "header_size"))
        self._patterns = self._read_patterns(cursor)
        self._samples: list[Sample] = []
        self._instruments = self._read_instruments(cursor)

    def song(self) -> Song:
        """The format-agnostic content the file carries."""
        return Song(
            name=decode_name(read_bytes(self._header, "name")),
            channels=self._channels,
            patterns=self._patterns,
            order=self._order,
            instruments=self._instruments,
            samples=tuple(self._samples),
            playback=Playback(speed=read_int(self._header, "speed"), tempo=read_int(self._header, "tempo")),
        )

    def settings(self) -> XMSettings:
        """The song-wide values this format adds, as the header states them."""
        return XMSettings(
            tracker=decode_name(read_bytes(self._header, "tracker")),
            flags=HeaderFlag(read_int(self._header, "flags")),
        )

    def _read_order(self, data: bytes) -> OrderList:
        """The played order list, whose restart position is pulled back inside it when a file overshoots."""
        count = read_int(self._header, "order_count")
        entries = tuple(data[FILE_HEADER_BYTES : FILE_HEADER_BYTES + count])
        restart = read_int(self._header, "restart_position")
        return OrderList(entries=entries, restart=min(restart, max(len(entries) - 1, 0)))

    def _read_patterns(self, cursor: Cursor) -> tuple[Pattern, ...]:
        count = read_int(self._header, "pattern_count")
        return tuple(unpack_pattern(cursor, channels=self._channels) for _ in range(count))

    def _read_instruments(self, cursor: Cursor) -> tuple[Instrument, ...]:
        count = read_int(self._header, "instrument_count")
        return tuple(self._read_instrument(cursor) for _ in range(count))

    def _read_instrument(self, cursor: Cursor) -> Instrument:
        identity = EMPTY_INSTRUMENT_HEADER.unpack(cursor.peek(EMPTY_INSTRUMENT_HEADER_BYTES))
        size = read_int(identity, "header_size")
        length = read_int(identity, "sample_count")
        extended = length > 0 and size >= INSTRUMENT_HEADER_BYTES
        values = INSTRUMENT_HEADER.unpack(cursor.peek(INSTRUMENT_HEADER_BYTES)) if extended else identity

        cursor.skip(size)
        offset = len(self._samples)
        self._samples.extend(self._read_group(cursor, length=length))
        if not extended:
            return parse_stub(values)

        return parse_instrument(values, offset=offset, length=length)

    @staticmethod
    def _read_group(cursor: Cursor, *, length: int) -> list[Sample]:
        """One instrument's samples: every header first, then every waveform, as the file lays them out."""
        headers: list[RecordValues] = [cursor.read(SAMPLE_HEADER) for _ in range(length)]
        return [parse_sample(values, cursor.take(stored_bytes(values))) for values in headers]
