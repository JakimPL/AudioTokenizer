from trackmod.core.patterns.grid import Pattern
from trackmod.it.layout.pattern import PATTERN_HEADER
from trackmod.it.note import NOTE_BYTES
from trackmod.it.patterns.encoded import EncodedCell
from trackmod.it.patterns.memory import ChannelMemory
from trackmod.it.spec.cells import CHANNEL_MARKER, END_OF_ROW, INSTRUMENT_OFFSET, CellMask
from trackmod.spec.grid import EMPTY
from trackmod.spec.width import BYTE_MAX


def encode_column(value: int, remembered: int, *, fresh: CellMask, reuse: CellMask, encoded: EncodedCell) -> int:
    """Add one column to a cell and return the value the channel goes on remembering."""
    if value == EMPTY:
        return remembered

    if value == remembered:
        encoded.mask |= reuse
        return remembered

    encoded.mask |= fresh
    encoded.payload.append(value & BYTE_MAX)
    return value


def encode_effect(command: int, parameter: int, memory: ChannelMemory, encoded: EncodedCell) -> None:
    """Add the effect column to a cell, spending two bytes only when the pair has changed."""
    if command == EMPTY:
        return

    if command == memory.command and parameter == memory.parameter:
        encoded.mask |= CellMask.LAST_EFFECT
        return

    encoded.mask |= CellMask.EFFECT
    encoded.payload.append(command & BYTE_MAX)
    encoded.payload.append(parameter & BYTE_MAX)
    memory.command, memory.parameter = command, parameter


def encode_cell(
    note: int,
    instrument: int,
    volume: int,
    command: int,
    parameter: int,
    memory: ChannelMemory,
) -> EncodedCell:
    """The mask and payload one grid position contributes, updating what its channel remembers."""
    encoded = EncodedCell()
    memory.note = encode_column(
        note,
        memory.note,
        fresh=CellMask.NOTE,
        reuse=CellMask.LAST_NOTE,
        encoded=encoded,
    )
    memory.instrument = encode_column(
        instrument,
        memory.instrument,
        fresh=CellMask.INSTRUMENT,
        reuse=CellMask.LAST_INSTRUMENT,
        encoded=encoded,
    )
    memory.volume = encode_column(
        volume,
        memory.volume,
        fresh=CellMask.VOLUME,
        reuse=CellMask.LAST_VOLUME,
        encoded=encoded,
    )
    encode_effect(command, parameter, memory, encoded)
    return encoded


def stored_note(value: int) -> int:
    """The note byte a grid note code stores as, leaving an absent note absent."""
    return EMPTY if value == EMPTY else NOTE_BYTES[value]


def stored_instrument(value: int) -> int:
    """The instrument byte a grid instrument index stores as, leaving an absent instrument absent.

    Instrument numbers are stored one above the shared numbering, because zero is what a cell writes to
    leave the channel on the instrument it already carries.
    """
    return EMPTY if value == EMPTY else value + INSTRUMENT_OFFSET


def stored_parameter(command: int, parameter: int) -> int:
    """The parameter byte a stated command carries, which is zero when the grid left it absent."""
    return 0 if command != EMPTY and parameter == EMPTY else parameter


def pack_cells(pattern: Pattern) -> bytes:
    """Serialise a pattern grid into this format's channel-marker byte stream.

    A row lists only the channels that carry something and ends with a zero byte, so a silent channel
    costs nothing. Each listed channel spends a mask byte only when its mask differs from the one it
    last used, and a column byte only when that column's value differs from the channel's last — which is
    what makes a channel holding steady settle to a single byte per row.
    """
    notes, instruments = pattern.note, pattern.instrument
    volumes, commands, parameters = pattern.volume, pattern.effect, pattern.parameter
    occupied = pattern.occupied
    memories = [ChannelMemory() for _ in range(pattern.channels)]

    stream = bytearray()
    for row in range(pattern.rows):
        for channel in range(pattern.channels):
            if not occupied[row, channel]:
                continue

            memory = memories[channel]
            command = int(commands[row, channel])
            encoded = encode_cell(
                stored_note(int(notes[row, channel])),
                stored_instrument(int(instruments[row, channel])),
                int(volumes[row, channel]),
                command,
                stored_parameter(command, int(parameters[row, channel])),
                memory,
            )

            if encoded.mask == memory.mask:
                stream.append((channel + 1) & BYTE_MAX)
            else:
                stream.append(((channel + 1) | CHANNEL_MARKER) & BYTE_MAX)
                stream.append(encoded.mask)
                memory.mask = encoded.mask

            stream.extend(encoded.payload)

        stream.append(END_OF_ROW)

    return bytes(stream)


def pack_pattern(pattern: Pattern) -> bytes:
    """Serialise a pattern: its header, then the packed cell stream."""
    stream = pack_cells(pattern)
    header = PATTERN_HEADER.pack({"packed_size": len(stream), "rows": pattern.rows, "reserved": 0})
    return header + stream
