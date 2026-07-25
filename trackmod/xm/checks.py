import numpy as np

from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.patterns.grid import Pattern
from trackmod.core.songs.song import Song
from trackmod.limits.capability import Capability
from trackmod.limits.checklist import Checklist
from trackmod.limits.table import Limits
from trackmod.limits.violation import Violation
from trackmod.spec.grid import EMPTY
from trackmod.spec.pitch import NOTE_COUNT
from trackmod.xm.patterns.sizing import packed_bytes


def check_envelope(checklist: Checklist, envelope: Envelope | None, *, subject: str) -> None:
    """Grade one envelope's length and every breakpoint it carries."""
    if envelope is None:
        return

    checklist.check(Capability.ENVELOPE_POINTS, envelope.length, subject=subject)
    for point in envelope.points:
        checklist.check(Capability.ENVELOPE_TICK, point.tick, subject=subject)
        checklist.check(Capability.ENVELOPE_VALUE, point.value, subject=subject)


def check_keys(checklist: Checklist, pattern: Pattern, *, subject: str) -> None:
    """Grade the highest key a pattern plays, which this format numbers fewer of than the model.

    Only keys are graded; the note column's commands are numbered past the keyboard in the shared model
    and are the writer's business, not a quantity to bound.
    """
    notes = pattern.note
    keys = notes[(notes != EMPTY) & (notes < NOTE_COUNT)]
    if keys.size:
        checklist.check(Capability.NOTE, int(np.max(keys)), subject=subject)


def check_song(checklist: Checklist, song: Song) -> None:
    """Grade the counts and the starting clock a song declares."""
    checklist.check(Capability.CHANNELS, song.channels, subject="song")
    checklist.check(Capability.PATTERNS, len(song.patterns), subject="song")
    checklist.check(Capability.ORDERS, song.order.length, subject="song")
    checklist.check(Capability.INSTRUMENTS, len(song.instruments), subject="song")
    checklist.check(Capability.SAMPLES, stored_samples(song), subject="song")
    checklist.check(Capability.SPEED, song.playback.speed, subject="song")
    checklist.check(Capability.TEMPO, song.playback.tempo, subject="song")


def stored_samples(song: Song) -> int:
    """How many sample slots the file holds, which counts a shared sample once per instrument."""
    return sum(len(instrument.samples) for instrument in song.instruments)


def check_patterns(checklist: Checklist, song: Song) -> None:
    """Grade each pattern's height, the size of the stream it packs into and the keys it plays."""
    for index, pattern in enumerate(song.patterns):
        subject = f"pattern {index}"
        checklist.check(Capability.PATTERN_ROWS, pattern.rows, subject=subject)
        checklist.check(Capability.PATTERN_BYTES, packed_bytes(pattern), subject=subject)
        check_keys(checklist, pattern, subject=subject)


def check_samples(checklist: Checklist, song: Song) -> None:
    """Grade each sample's waveform length, playback rate and two volume levels.

    The gain bound is what tells a caller this format has no per-sample multiplier: a sample asking for
    anything below full gain is reporting that the scaling has to be baked into its waveform instead.
    """
    for index, sample in enumerate(song.samples):
        subject = f"sample {index} ({sample.name!r})"
        checklist.check(Capability.SAMPLE_FRAMES, sample.frames, subject=subject)
        checklist.check(Capability.SAMPLE_RATE, sample.rate, subject=subject)
        checklist.check(Capability.SAMPLE_VOLUME, sample.volume, subject=subject)
        checklist.check(Capability.SAMPLE_GAIN, sample.gain, subject=subject)


def check_instruments(checklist: Checklist, song: Song) -> None:
    """Grade each instrument's level, fadeout, sample fan-out and envelopes."""
    for index, instrument in enumerate(song.instruments):
        subject = f"instrument {index} ({instrument.name!r})"
        checklist.check(Capability.INSTRUMENT_VOLUME, instrument.global_volume, subject=subject)
        checklist.check(Capability.FADEOUT, instrument.fadeout, subject=subject)
        checklist.check(Capability.SAMPLES_PER_INSTRUMENT, len(instrument.samples), subject=subject)
        check_envelope(checklist, instrument.volume_envelope, subject=subject)
        check_envelope(checklist, instrument.panning_envelope, subject=subject)


def violations(song: Song, *, limits: Limits) -> tuple[Violation, ...]:
    """Every bound a song breaks, in the order the checks find them.

    This format adds no song-wide levels of its own, so its settings carry nothing to bound and the
    whole report comes from the song.
    """
    checklist = Checklist(limits)
    check_song(checklist, song)
    check_patterns(checklist, song)
    check_samples(checklist, song)
    check_instruments(checklist, song)
    return checklist.violations
