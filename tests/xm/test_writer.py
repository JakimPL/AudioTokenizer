from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotokenizer.xm import spec
from audiotokenizer.xm.format import FILE_HEADER, INSTRUMENT_HEADER, PATTERN_HEADER, SAMPLE_HEADER
from audiotokenizer.xm.instruments import XMInstrument, note_keymap, play_note_for
from audiotokenizer.xm.module import XMModule, write_xm_module
from audiotokenizer.xm.patterns import XMPattern, XMPlayback, pack_pattern
from audiotokenizer.xm.reader import read_xm_samples
from audiotokenizer.xm.samples import XMSample, pcm_bytes, relative_note_for


def _tiny_module(*, tempo: int = 1000, depth_bits: int = 8) -> tuple[XMModule, np.ndarray, np.ndarray]:
    """A 2-channel, 4-sample (2-atom, 1-instrument) module with a known sparse grid."""
    block = 8
    n_slots = 4
    rng = np.random.default_rng(1)
    pcm = rng.uniform(-1, 1, size=(n_slots, block))
    samples = tuple(
        XMSample(
            name=f"s{i}",
            pcm=pcm[i],
            depth_bits=depth_bits,
            volume=40 + i,
            relative_note=relative_note_for(play_note_for(i)),
        )
        for i in range(n_slots)
    )
    instruments = (XMInstrument(name="dict0", n_samples=n_slots),)
    rows = 5
    sample_no = np.zeros((rows, 2), dtype=np.int64)
    volume = np.zeros((rows, 2), dtype=np.int64)
    for r in range(rows):
        volume[r, 0] = 30 + r  # channel 0: slot 0 every row
    for r in range(3):
        sample_no[r, 1] = 2  # channel 1: slot 2 for first three rows, then silent
        volume[r, 1] = 50
    pattern = XMPattern(rows=rows, sample_no=sample_no, volume=volume)
    module = XMModule(
        name="tiny",
        samples=samples,
        instruments=instruments,
        patterns=(pattern,),
        orders=(0,),
        playback=XMPlayback(speed=1, tempo=tempo),
        channels=2,
    )
    return module, sample_no, volume


def test_record_sizes_match_the_spec() -> None:
    assert FILE_HEADER.size == spec.FILE_HEADER_BYTES == 80
    assert PATTERN_HEADER.size == spec.PATTERN_HEADER_BYTES == 9
    assert INSTRUMENT_HEADER.size == spec.INSTRUMENT_HEADER_BYTES == 263
    assert SAMPLE_HEADER.size == spec.SAMPLE_HEADER_BYTES == 40


@pytest.mark.parametrize("depth_bits", [8, 16])
def test_delta_encoding_round_trips_through_a_running_sum(depth_bits: int) -> None:
    rng = np.random.default_rng(0)
    pcm = rng.uniform(-1.0, 1.0, size=97)
    sample = XMSample(name="t", pcm=pcm, depth_bits=depth_bits)
    raw = pcm_bytes(sample)
    dtype = np.int16 if depth_bits == 16 else np.int8
    scale = 32768.0 if depth_bits == 16 else 128.0
    delta = np.frombuffer(raw, dtype=dtype).astype(np.int64)
    recovered = np.cumsum(delta).astype(dtype)  # the player integrates deltas in the stored width
    expected = np.clip(np.round(pcm * scale), -scale, scale - 1).astype(dtype)
    assert np.array_equal(recovered, expected)


def test_header_fields_and_the_16bit_tempo_hack() -> None:
    module, _, _ = _tiny_module(tempo=1000)
    data = write_xm_module(module)
    assert data[: spec.MAGIC_BYTES] == spec.MAGIC
    assert data[37] == spec.STRIPPED_BYTE
    assert struct.unpack_from("<H", data, 58)[0] == spec.VERSION
    assert struct.unpack_from("<I", data, 60)[0] == spec.HEADER_SIZE_FIELD == 276
    assert struct.unpack_from("<H", data, 68)[0] == 2  # channels
    assert struct.unpack_from("<H", data, 70)[0] == 1  # patterns
    assert struct.unpack_from("<H", data, 72)[0] == 1  # instruments
    assert struct.unpack_from("<H", data, 78)[0] == 1000  # bpm stored as a 16-bit word (> IT's 255)


def test_sections_walk_to_the_exact_end_of_file() -> None:
    module, _, _ = _tiny_module()
    data = write_xm_module(module)
    pattern_start = 60 + struct.unpack_from("<I", data, 60)[0]
    assert pattern_start == 336  # header (80) + order table (256)
    packed = struct.unpack_from("<H", data, pattern_start + 7)[0]
    instr_start = pattern_start + spec.PATTERN_HEADER_BYTES + packed
    header_size = struct.unpack_from("<I", data, instr_start)[0]
    sample_count = struct.unpack_from("<H", data, instr_start + 27)[0]
    assert header_size == 263 and sample_count == 4
    sample_headers = instr_start + header_size
    pcm_total = sum(struct.unpack_from("<I", data, sample_headers + k * 40)[0] for k in range(sample_count))
    end = sample_headers + sample_count * 40 + pcm_total
    assert end == len(data)


def test_packed_pattern_matches_the_header_and_has_no_row_terminator() -> None:
    module, sample_no, volume = _tiny_module()
    pattern = module.patterns[0]
    blob = pack_pattern(pattern)
    packed = struct.unpack_from("<H", blob, 7)[0]
    assert len(blob) == spec.PATTERN_HEADER_BYTES + packed
    # ch0: 4 (first, instrument change) + 3*4 kept = 16; ch1: 4 + 3*2 kept + 1*2 empty = 12; total 28.
    assert packed == 28


def test_pack_pattern_rejects_rows_out_of_range() -> None:
    grid = np.zeros((1, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        pack_pattern(XMPattern(rows=spec.MAX_ROWS + 1, sample_no=grid, volume=grid))


def test_pack_pattern_rejects_a_stream_over_the_u16_limit() -> None:
    rng = np.random.default_rng(9)
    rows, channels = spec.MAX_ROWS, 255
    volume = rng.integers(1, 65, size=(rows, channels)).astype(np.int64)  # every cell present
    sample_no = rng.integers(0, 64, size=(rows, channels)).astype(np.int64)
    with pytest.raises(ValueError):
        pack_pattern(XMPattern(rows=rows, sample_no=sample_no, volume=volume))


def test_keymap_routes_each_slot_to_its_own_key() -> None:
    keymap = note_keymap(spec.SAMPLES_PER_INSTRUMENT)
    for slot in range(spec.SAMPLES_PER_INSTRUMENT):
        note = play_note_for(slot)
        assert keymap[note - 1] == slot + 1  # 0-based note index -> 1-based sample
    assert len(keymap) == spec.KEYMAP_NOTES


def test_write_rejects_inconsistent_sample_counts() -> None:
    module, _, _ = _tiny_module()
    broken = XMModule(
        name=module.name,
        samples=module.samples[:-1],  # one fewer sample than the instrument claims
        instruments=module.instruments,
        patterns=module.patterns,
        orders=module.orders,
        playback=module.playback,
        channels=module.channels,
    )
    with pytest.raises(ValueError):
        write_xm_module(broken)


@pytest.mark.parametrize("depth_bits", [8, 16])
def test_xmodits_recovers_the_stored_pcm(depth_bits: int) -> None:
    block = 48
    t = np.linspace(0, 1, block, endpoint=False)
    pcm0 = np.sin(2 * np.pi * 3 * t) * 0.9
    pcm1 = (t - 0.5) * 1.6  # constant-slope ramp: the worst case for delta encoding
    samples = (
        XMSample(name="sin", pcm=pcm0, depth_bits=depth_bits, volume=64),
        XMSample(name="ramp", pcm=pcm1, depth_bits=depth_bits, volume=64),
    )
    module = XMModule(
        name="fid",
        samples=samples,
        instruments=(XMInstrument(name="d0", n_samples=2),),
        patterns=(XMPattern(rows=2, sample_no=np.array([[0], [1]]), volume=np.array([[40], [40]])),),
        orders=(0,),
        playback=XMPlayback(speed=1, tempo=500),
        channels=1,
    )
    scale = 32768.0 if depth_bits == 16 else 128.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fid.xm"
        path.write_bytes(write_xm_module(module))
        ripped = read_xm_samples(path)
    assert len(ripped) == 2
    for original, recovered in zip((pcm0, pcm1), ripped):
        expected = np.clip(np.round(original * scale), -scale, scale - 1) / scale
        n = min(len(expected), recovered.frames)
        assert np.max(np.abs(expected[:n] - recovered.pcm[:n])) < 1.5 / scale
