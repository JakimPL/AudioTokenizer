"""The instrument record: the note->sample keymap and the 263-byte extended instrument header.

An XM instrument is a router, like its IT counterpart, but XM routes by *pitch*: the note a cell plays
both selects the sample (through this keymap) and sets its pitch (which each sample's relative-note trims
back to native). So an instrument owns up to :data:`SAMPLES_PER_INSTRUMENT` samples and gives each its own
key — slot ``k`` answers to note ``NOTE_BASE + k``. :func:`note_keymap` and the pattern writer must agree
on that key, so both derive it from :data:`NOTE_BASE`; the keymap is indexed by the 0-based note the player
looks up (``keymap[note - 1]``). Envelopes are left disabled, so a one-shot sample plays at the row's
volume-column level (XM's column overrides the sample volume byte) and falls silent when it ends — exactly
at the row boundary, since it is one row long.
"""

from __future__ import annotations

from dataclasses import dataclass

from audiotokenizer.xm.format import INSTRUMENT_HEADER, encode_name
from audiotokenizer.xm.spec import (
    INSTRUMENT_HEADER_BYTES,
    INSTRUMENT_TYPE,
    KEYMAP_NOTES,
    NAME_BYTES,
    NOTE_BASE,
    SAMPLE_HEADER_SIZE_FIELD,
    SAMPLES_PER_INSTRUMENT,
)


@dataclass(frozen=True)
class XMInstrument:
    """One instrument owning ``n_samples`` (1..16) consecutive samples, each on its own key."""

    name: str
    n_samples: int


def play_note_for(slot: int) -> int:
    """The 1-based pattern note that triggers slot ``slot`` (0-based) of an instrument."""
    return NOTE_BASE + slot


def note_keymap(n_samples: int) -> bytes:
    """A 96-byte keymap routing key ``NOTE_BASE + k`` to the instrument's ``k``-th sample.

    XM keymap entries are **0-based** sample indices within the instrument (unlike IT, whose sample
    numbers are 1-based): key ``NOTE_BASE + k`` selects sample ``k``. Indexed by the 0-based note the
    player derives from the 1-based pattern note (``keymap[note - 1]``). Unmapped keys stay ``0``, which
    is simply the first sample rather than "no sample" — harmless here because the writer only ever
    triggers the mapped keys ``NOTE_BASE .. NOTE_BASE + n_samples - 1``.
    """
    if not 1 <= n_samples <= SAMPLES_PER_INSTRUMENT:
        raise ValueError(f"instrument holds {n_samples} samples, expected 1..{SAMPLES_PER_INSTRUMENT}")
    keymap = bytearray(KEYMAP_NOTES)
    for slot in range(n_samples):
        keymap[play_note_for(slot) - 1] = slot  # 0-based sample index within the instrument
    return bytes(keymap)


def instrument_header(instrument: XMInstrument) -> bytes:
    """Serialize a 263-byte extended instrument header with disabled envelopes."""
    return INSTRUMENT_HEADER.pack(
        {
            "header_size": INSTRUMENT_HEADER_BYTES,
            "name": encode_name(instrument.name, NAME_BYTES),
            "type": INSTRUMENT_TYPE,
            "sample_count": instrument.n_samples,
            "sample_header_size": SAMPLE_HEADER_SIZE_FIELD,
            "keymap": note_keymap(instrument.n_samples),
        }
    )
