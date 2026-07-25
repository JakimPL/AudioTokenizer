from trackmod.binary.records.values import ArrayValue, FieldValue
from trackmod.binary.text import encode_name
from trackmod.core.envelopes.kind import EnvelopeKind
from trackmod.core.instruments.instrument import Instrument
from trackmod.it.instruments.envelope import envelope_values
from trackmod.it.instruments.keymap import note_map
from trackmod.it.layout.instrument import INSTRUMENT_HEADER
from trackmod.it.panning import stored_panning
from trackmod.it.spec.defaults import C5_NOTE, PANNING_DISABLED
from trackmod.it.spec.identity import MAGIC_INSTRUMENT
from trackmod.it.spec.sizes import FILENAME_BYTES, NAME_BYTES


def instrument_header(instrument: Instrument) -> bytes:
    """Serialise an instrument header, its keyboard routing and its three envelopes."""
    panning = PANNING_DISABLED if instrument.panning is None else stored_panning(instrument.panning)
    values: dict[str, FieldValue | ArrayValue] = {
        "magic": MAGIC_INSTRUMENT,
        "filename": encode_name(instrument.name, FILENAME_BYTES),
        "new_note_action": int(instrument.new_note_action),
        "duplicate_check": int(instrument.duplicate_check),
        "duplicate_action": int(instrument.duplicate_action),
        "fadeout": instrument.fadeout,
        "pitch_pan_separation": 0,
        "pitch_pan_center": C5_NOTE,
        "global_volume": instrument.global_volume,
        "default_pan": panning,
        "random_volume": 0,
        "random_panning": 0,
        "tracker_version": 0,
        "sample_count": len(instrument.samples),
        "name": encode_name(instrument.name, NAME_BYTES),
        "filter_cutoff": 0,
        "filter_resonance": 0,
        "midi_channel": 0,
        "midi_program": 0,
        "midi_bank": 0,
        "note_map": note_map(instrument.keymap),
    }
    for kind in EnvelopeKind:
        values.update(envelope_values(kind, instrument.envelope(kind)))

    return INSTRUMENT_HEADER.pack(values)
