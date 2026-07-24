"""The fixed byte values of the Impulse Tracker ``.IT`` layout: the single home for every format
constant the writer, the size model and the timing lattice reason against.

These are the spec-defined facts from ITTECH.TXT — magic tags, header flags, pattern-cell mask bits,
record sizes and the volume/tempo ceilings. They carry no encoder knowledge; the record serializers in
:mod:`audiotokenizer.it.format` and its siblings, and the byte model in
:mod:`audiotokenizer.coding.cost`, supply values against them.
"""

from __future__ import annotations

from typing import Final

# -- record sizes (bytes) --------------------------------------------------

FILE_HEADER_BYTES: Final = 192
SAMPLE_HEADER_BYTES: Final = 80
INSTRUMENT_HEADER_BYTES: Final = 554  # IT 2.14 instrument, envelopes included
PATTERN_HEADER_BYTES: Final = 8
OFFSET_TABLE_ENTRY_BYTES: Final = 4  # each instrument/sample/pattern offset is a little-endian u32

# -- volume ceilings -------------------------------------------------------

MAX_VOLUME: Final = 64  # the volume column and every 0..64 volume field
MAX_GLOBAL_VOLUME: Final = 128  # file-header and instrument global volume
MAX_MIX_VOLUME: Final = 128  # file-header mix volume
PANNING_SEPARATION: Final = 128  # full stereo separation in the file header

# -- pattern / row limits --------------------------------------------------

MIN_ROWS: Final = 32  # ITTECH: "Ranges from 32->200"
MAX_ROWS: Final = 200
MAX_PATTERNS: Final = 200  # order-list values 0..199 are patterns
MAX_PATTERN_BYTES: Final = 0xFFFF  # a pattern header stores its packed row stream's length as a u16

# -- tempo / speed ---------------------------------------------------------

MIN_TEMPO: Final = 32
MAX_TEMPO: Final = 255
MIN_SPEED: Final = 1
MAX_SPEED: Final = 255
#: A tick lasts ``TICK_SECONDS_NUMERATOR / tempo`` seconds and a row lasts ``speed`` ticks, so a row
#: spans ``speed * TICK_SECONDS_NUMERATOR * frame_rate / tempo`` frames.
TICK_SECONDS_NUMERATOR: Final = 2.5

# -- keyboard / channels ---------------------------------------------------

KEYBOARD_NOTES: Final = 120  # IT keys C-0..B-9 (0..119); one (play_note, sample) pair each
MAX_IT_NOTE: Final = KEYBOARD_NOTES - 1
IT_C5_NOTE: Final = 60  # note 60 = C-5, the reference key played at a sample's C5Speed
MAX_C5SPEED: Final = 9_999_999  # ITTECH: "ranges from 0->9999999"
MAX_SAMPLES: Final = 255  # a pattern cell stores the sample/instrument in one byte

CHANNELS_STORED: Final = 64  # the file header always carries 64 channel pan + 64 channel volume bytes
#: Impulse Tracker masks the pattern channel byte to 6 bits (64 channels); OpenMPT masks to 7 bits,
#: verified by rendering a 127-channel probe module through openmpt123.
STRICT_MAX_CHANNELS: Final = 64
HACKED_MAX_CHANNELS: Final = 127

STRICT_MAX_TEMPO: Final = 255
HACKED_MAX_TEMPO: Final = 65535

# -- header defaults -------------------------------------------------------

PAN_CENTER: Final = 32  # centred channel pan (IT pan spans 0..64)
CHANNEL_VOLUME_FULL: Final = MAX_VOLUME  # every stored channel plays at full volume
NOTE_ACTION_CUT: Final = 0  # new-note action: cut the previous note (0=cut, 1=continue, 2=off, 3=fade)
NOTE_CUT: Final = 254  # pattern note value that cuts the playing note instantly

# -- magic tags ------------------------------------------------------------

MAGIC_MODULE: Final = b"IMPM"
MAGIC_SAMPLE: Final = b"IMPS"
MAGIC_INSTRUMENT: Final = b"IMPI"

NAME_BYTES: Final = 26  # song, sample and instrument names are all 26 bytes, null-padded

CWT: Final = 0x0214  # "created with" IT 2.14 -> selects the 554-byte instrument + envelope format
CMWT: Final = 0x0214  # "compatible with"; >= 0x0200 is required for the instrument format

# -- file-header flags -----------------------------------------------------

FLAG_USE_INSTRUMENTS: Final = 0x04
FLAG_LINEAR_SLIDES: Final = 0x08

# -- sample-header flags ---------------------------------------------------

SMP_FLAG_DATA: Final = 0x01  # sample data present
SMP_FLAG_16BIT: Final = 0x02  # 16-bit (else 8-bit)
SMP_FLAG_LOOP: Final = 0x10  # forward loop enabled (loop begin/end fields are read)
CVT_SIGNED: Final = 0x01  # signed PCM, the standard IT storage

# -- pattern cell mask bits ------------------------------------------------

MASK_NOTE: Final = 0x01
MASK_INSTRUMENT: Final = 0x02
MASK_VOLUME: Final = 0x04
MASK_EFFECT: Final = 0x08
MASK_LAST_NOTE: Final = 0x10  # re-trigger the channel's previous note, no note byte spent
MASK_LAST_INSTRUMENT: Final = 0x20
MASK_LAST_VOLUME: Final = 0x40

# -- pattern byte stream ---------------------------------------------------

CHANNEL_MARKER: Final = 0x80  # set on a packed cell's channel byte when a fresh mask byte follows
END_OF_ROW: Final = 0x00  # a zero byte terminates a packed pattern row
ORDER_TERMINATOR: Final = 0xFF  # ends the order list
