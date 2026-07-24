"""The fixed byte values of the FastTracker 2 ``.XM`` layout: the single home for every format constant
the writer, the size model and the timing lattice reason against.

These are the spec-defined facts from the XM (Extended Module) format — magic tags, record sizes, the
pattern-cell mask bits, and the tempo/channel ceilings. They carry no encoder knowledge; the record
serializers in :mod:`audiotokenizer.xm.format` and the byte model in :mod:`audiotokenizer.coding.xm_cost`
supply values against them.

Two things set XM apart from Impulse Tracker and shape this whole package:

* **Tempo is a 16-bit word** (``bpm`` at offset 78), not IT's single byte. Writing a large value there is
  the FastTracker 2 "hack" the codec uses to reach rows shorter than IT's 255-tempo floor allows — the
  reason XM exists here at all. Real FT2 honours only 32..255; OpenMPT reads the full word, so the honoured
  ceiling is a profile knob pinned empirically against ``openmpt123``.
* **Pitch is note-relative, not an explicit rate.** XM has no C5Speed field; a sample plays at its recorded
  rate only when its ``note + relative_note`` lands on a reference the finetune trims. So every stored atom
  carries a :data:`RELATIVE_NOTE_TARGET`/:data:`SAMPLE_FINETUNE` tuning that puts native 44100 Hz on the
  key the pattern triggers. The reference and finetune are quantised on a log-frequency lattice, so 44100
  is reachable only to within ~170 ppm (IT stores it exactly); :data:`SAMPLE_C5_HZ` records the intent.
"""

from __future__ import annotations

from typing import Final

# -- magic / version -------------------------------------------------------

MAGIC: Final = b"Extended Module: "  # 17 bytes, the file's opening tag
MAGIC_BYTES: Final = len(MAGIC)  # 17
STRIPPED_BYTE: Final = 0x1A  # the DOS end-of-text byte written right after the 20-byte module name
VERSION: Final = 0x0104  # format version 1.04, the only one the packed-pattern layout here targets
TRACKER_NAME: Final = "AudioTokenizer"  # written into the 20-byte "tracker name" field

# -- record sizes (bytes) --------------------------------------------------

FILE_HEADER_BYTES: Final = 80  # the fixed header prefix (magic .. bpm); the 256-byte order table follows it
HEADER_SIZE_FIELD: Final = 276  # the u32 at offset 60: 20 fixed bytes (offsets 60..80) + the 256 order bytes
ORDER_TABLE_BYTES: Final = 256  # the order table is always written full-width
PATTERN_HEADER_BYTES: Final = 9  # header length (4) + packing type (1) + rows (2) + packed size (2)
INSTRUMENT_HEADER_BYTES: Final = 263  # the extended instrument header (samples > 0): keymap + envelopes
SAMPLE_HEADER_BYTES: Final = 40

# -- name field widths -----------------------------------------------------

MODULE_NAME_BYTES: Final = 20
TRACKER_NAME_BYTES: Final = 20
NAME_BYTES: Final = 22  # instrument and sample names are both 22 bytes, null-padded

# -- volume / panning ------------------------------------------------------

MAX_VOLUME: Final = 64  # the sample volume byte and the pattern volume column both span 0..64
PAN_CENTER: Final = 128  # XM sample panning spans 0..255; 128 is dead centre
VOLUME_COLUMN_BASE: Final = 0x10  # a volume-column byte of 0x10..0x50 sets volume 0..64 (base + level)

# -- pattern / row / order limits ------------------------------------------

MIN_ROWS: Final = 1
MAX_ROWS: Final = 256  # a pattern header stores its row count as a u16 but FT2 caps a pattern at 256 rows
MAX_PATTERNS: Final = 256  # the order table is 256 entries and pattern indices are one byte
MAX_ORDERS: Final = 256
MAX_PATTERN_BYTES: Final = 0xFFFF  # a pattern header stores its packed row stream's length as a u16
PATTERN_PACKING_TYPE: Final = 0  # the only packing type; the 0x80 mask scheme below is implied

# -- tempo / speed ---------------------------------------------------------

MIN_TEMPO: Final = 32
STRICT_MAX_TEMPO: Final = 255  # real FastTracker 2 honours a BPM of at most 255
HACKED_MAX_TEMPO: Final = 65535  # the header word is 16-bit; this is the hack the codec reaches for
MIN_SPEED: Final = 1
MAX_SPEED: Final = 0xFFFF  # speed shares the 16-bit header word; the codec uses speed 1 in practice
#: A tick lasts ``TICK_SECONDS_NUMERATOR / tempo`` seconds and a row lasts ``speed`` ticks — the same
#: clock IT uses, so the shared timing core derives the row length identically; only the ceiling differs.
TICK_SECONDS_NUMERATOR: Final = 2.5

# -- channels --------------------------------------------------------------

STRICT_MAX_CHANNELS: Final = 32  # the canonical FastTracker 2 channel ceiling
HACKED_MAX_CHANNELS: Final = 255  # OpenMPT's reach; pinned against openmpt123 the way IT's 127 was

# -- instrument / sample routing -------------------------------------------

MAX_INSTRUMENTS: Final = 128
SAMPLES_PER_INSTRUMENT: Final = 16  # 8 atoms x 2 polarities per instrument; XM keys select the sample
MAX_SAMPLES_TOTAL: Final = MAX_INSTRUMENTS * SAMPLES_PER_INSTRUMENT  # 2048 samples -> 1024 atoms
KEYMAP_NOTES: Final = 96  # the instrument keymap has one sample byte per playable key (C-0..B-7)

# -- note / tuning ---------------------------------------------------------
#: XM notes are 1-based (1 = C-0, 96 = B-7). Slot ``k`` of an instrument is triggered by note
#: ``NOTE_BASE + k``; the keymap routes that key to sample ``k`` and the sample's relative-note re-tunes
#: it to native rate. ``NOTE_BASE + SAMPLES_PER_INSTRUMENT - 1`` (40) stays well inside 1..96.
NOTE_BASE: Final = 25
#: A sample plays at its recorded rate when ``note + relative_note`` equals this reference (with
#: :data:`SAMPLE_FINETUNE` trimming the residual). Derived from FT2's linear-frequency table for 44100 Hz
#: and pinned empirically in the render check; see :data:`SAMPLE_C5_HZ`.
RELATIVE_NOTE_TARGET: Final = 77
SAMPLE_FINETUNE: Final = 100  # signed -128..127; the log-lattice trim onto 44100 Hz (~ -170 ppm residual)
SAMPLE_C5_HZ: Final = 44_100  # the rate the tuning targets; XM cannot store it exactly (unlike IT's C5Speed)

# -- file-header flags -----------------------------------------------------

FLAG_LINEAR_FREQUENCY: Final = 0x01  # 1 = linear frequency table (the table the tuning above assumes)

# -- sample-header type flags ----------------------------------------------

SMP_LOOP_NONE: Final = 0x00  # bits 0-1: 0 none, 1 forward, 2 ping-pong
SMP_LOOP_FORWARD: Final = 0x01
SMP_TYPE_16BIT: Final = 0x10  # bit 4: 16-bit sample data (else 8-bit)

# -- pattern cell mask bits ------------------------------------------------
#: A packed cell's first byte has bit 7 set and the low bits select which fields follow. With bit 7 clear
#: the byte is itself the note and all five fields follow uncompressed. XM keeps no per-channel "reuse
#: last" memory (unlike IT), so every present cell re-states its note — which is also what re-triggers the
#: one-shot sample each row — while the instrument byte is spent only when the channel's instrument changes.
MASK_PACKED: Final = 0x80
MASK_NOTE: Final = 0x01
MASK_INSTRUMENT: Final = 0x02
MASK_VOLUME: Final = 0x04
MASK_EFFECT: Final = 0x08
MASK_PARAMETER: Final = 0x10
EMPTY_CELL: Final = 0x80  # a packed cell with no field bits: one byte for a silent channel

# -- instrument header fixed fields ----------------------------------------

INSTRUMENT_TYPE: Final = 0  # the only instrument type
SAMPLE_HEADER_SIZE_FIELD: Final = SAMPLE_HEADER_BYTES  # the u32 sample-header-size inside an instrument
