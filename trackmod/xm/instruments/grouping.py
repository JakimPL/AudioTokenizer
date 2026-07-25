from collections.abc import Sequence

from trackmod.core.instruments.instrument import Instrument
from trackmod.core.notes.pitch import Note
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.song import Song
from trackmod.xm.instruments.group import SampleGroup
from trackmod.xm.spec.sizes import KEYMAP_NOTES
from trackmod.xm.tuning import Tuning, tuning_for

FIRST_SLOT = 0


def local_slots(instrument: Instrument) -> dict[int, int]:
    """Which position each sample the keys reach takes within the instrument, in first-named order."""
    slots: dict[int, int] = {}
    for key in range(KEYMAP_NOTES):
        assignment = instrument.keymap[key]
        if assignment is not None:
            slots.setdefault(assignment.sample, len(slots))

    return slots


def slot_tuning(instrument: Instrument, sample: Sample, *, index: int) -> Tuning:
    """The one transposition that serves every key routed to a sample.

    A stored sample is transposed once, so the keys that reach it must all want the same shift. Keys
    that sound their own pitch always agree, and so does a whole-instrument transposition; a keymap that
    shifts one key differently from another is asking for something the format cannot store.

    Raises:
        ValueError: when two keys routed to the same sample want different transpositions.
    """
    agreed: Tuning | None = None
    for key in range(KEYMAP_NOTES):
        assignment = instrument.keymap[key]
        if assignment is None or assignment.sample != index:
            continue

        tuning = tuning_for(sample.rate, key=Note(key), sounded=assignment.note)
        if agreed is None:
            agreed = tuning
        elif tuning != agreed:
            raise ValueError(
                f"instrument {instrument.name!r} transposes sample {index} differently on key {Note(key)}, "
                "which this format stores one transposition for"
            )

    if agreed is None:
        raise ValueError(f"instrument {instrument.name!r} routes no key to sample {index}")

    return agreed


def group_samples(instrument: Instrument, samples: Sequence[Sample]) -> SampleGroup:
    """The samples one instrument owns, the keymap that reaches them and how each one is tuned."""
    slots = local_slots(instrument)
    owned = tuple(samples[index] for index in slots)
    tunings = tuple(slot_tuning(instrument, samples[index], index=index) for index in slots)
    keymap = tuple(
        FIRST_SLOT if (assignment := instrument.keymap[key]) is None else slots[assignment.sample]
        for key in range(KEYMAP_NOTES)
    )
    return SampleGroup(samples=owned, tunings=tunings, keymap=keymap)


def song_groups(song: Song) -> tuple[SampleGroup, ...]:
    """Every instrument's samples, in the order the file lays them out."""
    return tuple(group_samples(instrument, song.samples) for instrument in song.instruments)
