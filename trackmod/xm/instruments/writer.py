from trackmod.binary.records.values import ArrayValue, FieldValue
from trackmod.binary.text import encode_name
from trackmod.core.instruments.instrument import Instrument
from trackmod.xm.instruments.envelope import envelope_values
from trackmod.xm.instruments.group import SampleGroup
from trackmod.xm.instruments.keymap import stored_keymap
from trackmod.xm.layout.envelope import ENVELOPE_KINDS
from trackmod.xm.layout.instrument import EMPTY_INSTRUMENT_HEADER, INSTRUMENT_HEADER
from trackmod.xm.samples.writer import sample_bytes, sample_header
from trackmod.xm.spec.defaults import INSTRUMENT_TYPE
from trackmod.xm.spec.sizes import (
    EMPTY_INSTRUMENT_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    NAME_BYTES,
    SAMPLE_HEADER_BYTES,
)


def identity_values(instrument: Instrument, group: SampleGroup, *, size: int) -> dict[str, FieldValue]:
    """The four fields both forms of the instrument header open with."""
    return {
        "header_size": size,
        "name": encode_name(instrument.name, NAME_BYTES),
        "type": INSTRUMENT_TYPE,
        "sample_count": group.length,
    }


def empty_header(instrument: Instrument, group: SampleGroup) -> bytes:
    """The short header a stored instrument that owns nothing is written as."""
    return EMPTY_INSTRUMENT_HEADER.pack(identity_values(instrument, group, size=EMPTY_INSTRUMENT_HEADER_BYTES))


def instrument_header(instrument: Instrument, group: SampleGroup) -> bytes:
    """Serialise an instrument header, its keyboard routing and its two envelopes.

    An instrument that owns no samples is written in the short form the format reserves for it, which
    stops after the sample count rather than reserving room for a keymap nothing would route through.

    Raises:
        ValueError: when the instrument carries a pitch envelope, which this format has no room for.
    """
    if instrument.pitch_envelope is not None:
        raise ValueError(f"instrument {instrument.name!r} carries a pitch envelope, which this format cannot store")

    if group.length == 0:
        return empty_header(instrument, group)

    values: dict[str, FieldValue | ArrayValue] = {
        **identity_values(instrument, group, size=INSTRUMENT_HEADER_BYTES),
        "sample_header_size": SAMPLE_HEADER_BYTES,
        "keymap": stored_keymap(group.keymap),
        "vibrato_type": 0,
        "vibrato_sweep": 0,
        "vibrato_depth": 0,
        "vibrato_rate": 0,
        "fadeout": instrument.fadeout,
        "reserved": 0,
    }
    for kind in ENVELOPE_KINDS:
        values.update(envelope_values(kind, instrument.envelope(kind)))

    return INSTRUMENT_HEADER.pack(values)


def instrument_block(instrument: Instrument, group: SampleGroup) -> bytes:
    """One instrument as the file lays it out: its header, then every sample header, then every waveform."""
    headers = b"".join(sample_header(sample, tuning=tuning) for sample, tuning in zip(group.samples, group.tunings))
    waveforms = b"".join(sample_bytes(sample) for sample in group.samples)
    return instrument_header(instrument, group) + headers + waveforms


def header_bytes(group: SampleGroup) -> int:
    """How many bytes one instrument spends on records, before any waveform it carries."""
    header = EMPTY_INSTRUMENT_HEADER_BYTES if group.length == 0 else INSTRUMENT_HEADER_BYTES
    return header + SAMPLE_HEADER_BYTES * group.length


def waveform_bytes(group: SampleGroup) -> int:
    """How many bytes one instrument's waveforms occupy, counting a shared sample once per owner."""
    return sum(sample.stored_bytes for sample in group.samples)
