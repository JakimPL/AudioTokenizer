from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audiotokenizer.audio.io import SAMPLE_RATE, save_audio
from audiotokenizer.cli import _build_parser, _resolve_format, main
from audiotokenizer.xm.spec import MAGIC


def _resolve(argv: list[str]) -> str:
    return _resolve_format(_build_parser().parse_args(argv))


def test_format_resolves_from_flag_then_suffix_then_default() -> None:
    assert _resolve(["in.wav"]) == "it"  # nothing given -> IT
    assert _resolve(["in.wav", "-o", "out.xm"]) == "xm"  # inferred from the suffix
    assert _resolve(["in.wav", "-o", "out.it"]) == "it"
    assert _resolve(["in.wav", "-o", "out.xm", "--format", "it"]) == "it"  # explicit flag wins over suffix


def _write_tone(path: Path) -> None:
    time = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE  # half a second
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


def test_planning_xm_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    _write_tone(source)
    with pytest.raises(SystemExit):
        main([str(source), "-o", str(tmp_path / "out.xm"), "--plan"])
