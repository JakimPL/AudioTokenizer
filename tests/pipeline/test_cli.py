from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from trackmod.trackers.it.spec.identity import MAGIC_MODULE
from trackmod.trackers.xm.spec.identity import MAGIC

from audiotokenizer.audio.io import SAMPLE_RATE, save_audio
from audiotokenizer.cli import _build_parser, _resolve_format, main
from audiotokenizer.module.format import Format


def _resolve(argv: list[str]) -> Format:
    return _resolve_format(_build_parser().parse_args(argv))


def test_format_resolves_from_flag_then_suffix_then_default() -> None:
    assert _resolve(["in.wav"]) is Format.IT  # nothing given -> IT
    assert _resolve(["in.wav", "-o", "out.xm"]) is Format.XM  # inferred from the suffix
    assert _resolve(["in.wav", "-o", "out.it"]) is Format.IT
    assert _resolve(["in.wav", "-o", "out.xm", "--format", "it"]) is Format.IT  # explicit flag wins over suffix


def _write_tone(path: Path, *, seconds: float = 0.5) -> None:
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = 0.3 * sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    save_audio(path, np.asarray(tone, dtype=np.float64), SAMPLE_RATE)


def test_cli_writes_a_playable_xm(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    output = tmp_path / "out.xm"
    _write_tone(source)
    assert main([str(source), "-o", str(output), "--tempo", "441", "--atoms", "8"]) == 0
    assert output.read_bytes()[: len(MAGIC)] == MAGIC


def test_cli_infers_xm_from_output_suffix(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    output = tmp_path / "song.xm"
    _write_tone(source)
    main([str(source), "-o", str(output), "--tempo", "441", "--atoms", "8"])  # no --format
    assert output.read_bytes()[: len(MAGIC)] == MAGIC


def test_cli_writes_an_it_by_default_and_names_it_after_the_input(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    _write_tone(source)
    assert main([str(source), "--tempo", "125", "--atoms", "8"]) == 0
    assert (tmp_path / "in.it").read_bytes()[: len(MAGIC_MODULE)] == MAGIC_MODULE


def test_strict_holds_the_module_to_the_canonical_channel_count(tmp_path: Path) -> None:
    # --strict is the compliance lever: the same pool that spreads past 32 FastTracker 2 channels by default
    # is held to what the tracker's own editor accepts, so the file stays a canonical module.
    source = tmp_path / "in.wav"
    _write_tone(source, seconds=3.0)
    strict = tmp_path / "strict.xm"
    loose = tmp_path / "loose.xm"
    main([str(source), "-o", str(strict), "--tempo", "125", "--atoms", "88", "--strict"])
    main([str(source), "-o", str(loose), "--tempo", "125", "--atoms", "88"])
    assert strict.read_bytes() != loose.read_bytes()


def test_strict_compiles_the_widest_canonical_pool_by_default(tmp_path: Path) -> None:
    # Canonical Impulse Tracker stores 99 samples, so the default pool narrows to the 49 atoms they hold.
    source = tmp_path / "in.wav"
    _write_tone(source, seconds=3.0)
    assert main([str(source), "-o", str(tmp_path / "strict.it"), "--tempo", "125", "--strict"]) == 0


def test_a_module_the_format_refuses_is_reported_rather_than_traced(tmp_path: Path) -> None:
    # Half a second at 882 frames a row is 25 rows, under the 32 a canonical Impulse Tracker pattern
    # needs. That is a real refusal, and the caller should be told which bound and at which level rather
    # than handed a traceback from the writer.
    source = tmp_path / "in.wav"
    _write_tone(source)
    with pytest.raises(SystemExit) as refused:
        main([str(source), "-o", str(tmp_path / "out.it"), "--tempo", "125", "--atoms", "8", "--strict"])
    message = str(refused.value)
    assert "pattern_rows" in message
    assert "--strict" in message


def test_cli_also_writes_the_reconstruction_when_asked(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    reference = tmp_path / "ref.wav"
    _write_tone(source)
    main([str(source), "-o", str(tmp_path / "out.it"), "--tempo", "125", "--atoms", "8", "--wav", str(reference)])
    assert reference.exists()


def test_planning_xm_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    _write_tone(source)
    with pytest.raises(SystemExit):
        main([str(source), "-o", str(tmp_path / "out.xm"), "--plan"])
