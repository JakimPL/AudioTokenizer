from trackmod.core.patterns.grid import Pattern
from trackmod.spec.grid import EMPTY
from trackmod.xm.layout.pattern import PATTERN_HEADER
from trackmod.xm.note import stored_note
from trackmod.xm.patterns.encoded import EncodedCell, StatedColumn
from trackmod.xm.spec.cells import INSTRUMENT_OFFSET, VOLUME_COLUMN_BASE, CellMask
from trackmod.xm.spec.defaults import PACKING_TYPE
from trackmod.xm.spec.sizes import PATTERN_HEADER_BYTES


def encode_cell(note: int, instrument: int, volume: int, command: int, parameter: int) -> EncodedCell:
    """The columns one grid position states, in the order a stored cell lists them.

    Instrument numbers are stored one above the shared numbering, because zero is what a cell writes to
    leave the channel on the instrument it already carries.
    """
    columns: list[StatedColumn] = []
    if note != EMPTY:
        columns.append((CellMask.NOTE, stored_note(note)))

    if instrument != EMPTY:
        columns.append((CellMask.INSTRUMENT, instrument + INSTRUMENT_OFFSET))

    if volume != EMPTY:
        columns.append((CellMask.VOLUME, VOLUME_COLUMN_BASE + volume))

    if command != EMPTY:
        columns.append((CellMask.EFFECT, command))

    if parameter != EMPTY:
        columns.append((CellMask.PARAMETER, parameter))

    return EncodedCell(columns=tuple(columns))


def pack_cells(pattern: Pattern) -> bytes:
    """Serialise a pattern grid into this format's mask-byte stream.

    Every position of every row is written and nothing terminates a row, so a player reads exactly rows
    times channels cells. A channel carrying nothing still costs the one byte its empty mask occupies,
    and the format keeps no memory between rows, so what a cell states is what the grid holds.
    """
    notes, instruments = pattern.note, pattern.instrument
    volumes, commands, parameters = pattern.volume, pattern.effect, pattern.parameter

    stream = bytearray()
    for row in range(pattern.rows):
        for channel in range(pattern.channels):
            encoded = encode_cell(
                int(notes[row, channel]),
                int(instruments[row, channel]),
                int(volumes[row, channel]),
                int(commands[row, channel]),
                int(parameters[row, channel]),
            )
            stream.extend(encoded.to_bytes())

    return bytes(stream)


def pack_pattern(pattern: Pattern) -> bytes:
    """Serialise a pattern: its header, then the packed cell stream."""
    stream = pack_cells(pattern)
    header = PATTERN_HEADER.pack(
        {
            "header_length": PATTERN_HEADER_BYTES,
            "packing_type": PACKING_TYPE,
            "rows": pattern.rows,
            "packed_size": len(stream),
        }
    )
    return header + stream
