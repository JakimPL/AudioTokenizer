from trackmod.binary.records.values import RecordValues, read_bytes, read_int
from trackmod.binary.text import decode_name
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import Keymap
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.xm.instruments.envelope import parse_envelope
from trackmod.xm.instruments.keymap import parse_keymap

SILENT_KEYMAP: Keymap = (None,) * NOTE_COUNT


def instrument_keymap(values: RecordValues, *, offset: int, length: int) -> Keymap:
    """The keyboard routing a stored instrument carries, silent when it owns nothing to route to."""
    if length == 0:
        return SILENT_KEYMAP

    return parse_keymap(read_bytes(values, "keymap"), offset=offset, length=length)


def parse_instrument(values: RecordValues, *, offset: int, length: int) -> Instrument:
    """Rebuild an instrument from its unpacked header fields and where its samples land in the module."""
    return Instrument(
        name=decode_name(read_bytes(values, "name")),
        keymap=instrument_keymap(values, offset=offset, length=length),
        volume_envelope=parse_envelope(EnvelopeKind.VOLUME, values),
        panning_envelope=parse_envelope(EnvelopeKind.PANNING, values),
        fadeout=read_int(values, "fadeout"),
    )


def parse_stub(values: RecordValues) -> Instrument:
    """Rebuild an instrument written in the short form, which owns nothing and so shapes nothing."""
    return Instrument(name=decode_name(read_bytes(values, "name")), keymap=SILENT_KEYMAP)
