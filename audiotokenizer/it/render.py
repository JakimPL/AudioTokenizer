"""Render ``.IT`` modules to audio with ``openmpt123`` — the ground-truth playback engine.

The codec reasons about the module with numpy; this wrapper drives the *real* tracker so a written
module can be checked as it will actually sound. ``openmpt123`` is an external system binary, not a
Python dependency, so this module degrades gracefully: :func:`openmpt123_available` reports whether it
is installed, and the render functions raise a clear, actionable error when it is missing.

Rendering is mono float at a fixed interpolation so the ground truth is reproducible — the IT format
stores no interpolation filter, it is a player setting, and the default here is 8-tap sinc, OpenMPT's
highest-quality mode. Rendering happens in a temporary directory (``openmpt123 --render`` writes
``<input>.wav`` next to its input), so the caller's files stay untouched.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.audio.io import load_audio
from audiotokenizer.it.module import ITModule, write_it

Interpolation = Literal["none", "linear", "cubic", "sinc"]

_BINARY: Final = "openmpt123"
# openmpt123 --filter takes interpolation *taps*; more taps = higher-quality (sinc) interpolation.
_INTERPOLATION_TAPS: Final[dict[Interpolation, int]] = {"none": 1, "linear": 2, "cubic": 4, "sinc": 8}
DEFAULT_INTERPOLATION: Final[Interpolation] = "sinc"
DEFAULT_GAIN_DB: Final = 0.0


def openmpt123_available(*, binary: str = _BINARY) -> bool:
    """Return whether the ``openmpt123`` binary is discoverable on ``PATH``."""
    return shutil.which(binary) is not None


def _require_binary(binary: str) -> None:
    if not openmpt123_available(binary=binary):
        raise RuntimeError(
            f"{binary!r} not found on PATH; install it (e.g. `apt install openmpt123`) to render .IT files"
        )


def _render_in_place(
    it_path: Path, *, sample_rate: int, interpolation: Interpolation, gain_db: float, binary: str
) -> tuple[NDArray[np.float64], int]:
    """Run ``openmpt123 --render`` on ``it_path`` and read back the ``<it_path>.wav`` it produces."""
    command = [
        binary,
        "--render",
        "--samplerate",
        str(sample_rate),
        "--channels",
        "1",
        "--filter",
        str(_INTERPOLATION_TAPS[interpolation]),
        "--gain",
        str(gain_db),
        "--force",
        "--quiet",
        str(it_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} render failed (exit {result.returncode}): {result.stderr.strip()}")
    rendered, rate = load_audio(it_path.with_name(it_path.name + ".wav"), mono=True)
    return np.asarray(rendered, dtype=np.float64).ravel(), rate


def render_it(
    path: Path | str,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = _BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Render an existing ``.IT`` file to mono float PCM, returning ``(samples, sample_rate)``.

    Raises:
        RuntimeError: when ``openmpt123`` is missing or the render fails.
    """
    _require_binary(binary)
    source = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / source.name
        local.write_bytes(source.read_bytes())
        return _render_in_place(
            local, sample_rate=sample_rate, interpolation=interpolation, gain_db=gain_db, binary=binary
        )


def render_module(
    module: ITModule,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = _BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Write ``module`` to a temporary ``.IT`` file and render it (see :func:`render_it`)."""
    _require_binary(binary)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "module.it"
        write_it(local, module)
        return _render_in_place(
            local, sample_rate=sample_rate, interpolation=interpolation, gain_db=gain_db, binary=binary
        )
