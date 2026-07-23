from __future__ import annotations

from typing import Final

# -- record sizes (bytes) --------------------------------------------------

FILE_HEADER_BYTES: Final = 192
SAMPLE_HEADER_BYTES: Final = 80
INSTRUMENT_HEADER_BYTES: Final = 554  # IT 2.14 instrument, envelopes included
PATTERN_HEADER_BYTES: Final = 8

# -- format limits ---------------------------------------------------------

MAX_VOLUME: Final = 64  # volume column and every 0..64 volume field
MAX_GLOBAL_VOLUME: Final = 128
MAX_MIX_VOLUME: Final = 128

MIN_ROWS: Final = 32  # ITTECH: "Ranges from 32->200"
MAX_ROWS: Final = 200
MAX_PATTERNS: Final = 200  # order-list values 0..199 are patterns

MIN_TEMPO: Final = 32
MAX_TEMPO: Final = 255
MIN_SPEED: Final = 1
MAX_SPEED: Final = 255
#: A tick lasts ``TICK_SECONDS_NUMERATOR / tempo`` seconds and a row lasts
#: ``speed`` ticks, so the shortest row is 2.5/255 s -- 432 frames at 44100 Hz.
TICK_SECONDS_NUMERATOR: Final = 2.5

KEYBOARD_NOTES: Final = 120  # IT keys C-0..B-9
IT_C5_NOTE: Final = 60  # C-5 plays a sample at its C5Speed
MAX_C5SPEED: Final = 9_999_999  # ITTECH: "ranges from 0->9999999"

MAX_SAMPLES: Final = 255  # a pattern cell stores the instrument in one byte

#: Impulse Tracker masks the pattern channel byte to 6 bits (64 channels);
#: OpenMPT masks to 7 bits, verified by rendering a 127-channel probe module.
STRICT_MAX_CHANNELS: Final = 64
HACKED_MAX_CHANNELS: Final = 127

# -- magic tags ------------------------------------------------------------

MAGIC_MODULE: Final = b"IMPM"
MAGIC_SAMPLE: Final = b"IMPS"
MAGIC_INSTRUMENT: Final = b"IMPI"

NAME_BYTES: Final = 26  # song, sample and instrument names are all 26 bytes

CWT: Final = 0x0214  # "created with" IT 2.14 -> 554-byte instrument format
CMWT: Final = 0x0214  # "compatible with"; >= 0x0200 required for instruments

# -- pattern cell mask bits ------------------------------------------------

MASK_NOTE: Final = 0x01
MASK_INSTRUMENT: Final = 0x02
MASK_VOLUME: Final = 0x04
MASK_EFFECT: Final = 0x08
MASK_LAST_NOTE: Final = 0x10  # re-trigger the channel's previous note, no byte spent
MASK_LAST_INSTRUMENT: Final = 0x20
MASK_LAST_VOLUME: Final = 0x40
CHANNEL_MARKER: Final = 0x80  # the marker carries a fresh mask byte
