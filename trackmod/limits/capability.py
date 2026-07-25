from enum import StrEnum, unique


@unique
class Capability(StrEnum):
    """A quantity a tracker format bounds, named once so every format states the same vocabulary."""

    CHANNELS = "channels"
    PATTERNS = "patterns"
    ORDERS = "orders"
    PATTERN_ROWS = "pattern_rows"
    PATTERN_BYTES = "pattern_bytes"
    INSTRUMENTS = "instruments"
    SAMPLES = "samples"
    SAMPLES_PER_INSTRUMENT = "samples_per_instrument"
    SAMPLE_FRAMES = "sample_frames"
    SAMPLE_RATE = "sample_rate"
    SAMPLE_VOLUME = "sample_volume"
    SAMPLE_GAIN = "sample_gain"
    INSTRUMENT_VOLUME = "instrument_volume"
    ENVELOPE_POINTS = "envelope_points"
    ENVELOPE_VALUE = "envelope_value"
    ENVELOPE_TICK = "envelope_tick"
    FADEOUT = "fadeout"
    NOTE = "note"
    TEMPO = "tempo"
    SPEED = "speed"
    VOLUME = "volume"
    SONG_VOLUME = "song_volume"
    MIX_VOLUME = "mix_volume"
    PANNING = "panning"
