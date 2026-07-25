from trackmod.binary.records.values import RecordValues, read_bytes, read_int, read_rows
from trackmod.binary.text import decode_name
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.core.instruments.behaviour import DuplicateAction, DuplicateCheck, NewNoteAction
from trackmod.core.instruments.instrument import Instrument
from trackmod.it.instruments.envelope import parse_envelope
from trackmod.it.instruments.keymap import parse_keymap
from trackmod.it.panning import shared_panning
from trackmod.it.spec.flags import SamplePanning


def parse_instrument(values: RecordValues) -> Instrument:
    """Rebuild an instrument from its unpacked header fields."""
    panning = read_int(values, "default_pan")
    return Instrument(
        name=decode_name(read_bytes(values, "name")),
        keymap=parse_keymap(read_rows(values, "note_map")),
        volume_envelope=parse_envelope(EnvelopeKind.VOLUME, values),
        panning_envelope=parse_envelope(EnvelopeKind.PANNING, values),
        pitch_envelope=parse_envelope(EnvelopeKind.PITCH, values),
        fadeout=read_int(values, "fadeout"),
        global_volume=read_int(values, "global_volume"),
        panning=None if panning & SamplePanning.ENABLED else shared_panning(panning),
        new_note_action=NewNoteAction(read_int(values, "new_note_action")),
        duplicate_check=DuplicateCheck(read_int(values, "duplicate_check")),
        duplicate_action=DuplicateAction(read_int(values, "duplicate_action")),
    )
