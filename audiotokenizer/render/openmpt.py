"""Render tracker modules to audio with ``openmpt123`` — the ground-truth playback engine.

The codec reasons about a module with numpy; this wrapper drives the *real* tracker so a written module can
be checked as it will actually sound. ``openmpt123`` renders every format libopenmpt supports (``.it``,
``.xm``, …), so the driver here is format-agnostic: it only needs the file's bytes and its extension, which
is exactly what a :class:`~trackmod.module.protocol.TrackerModule` offers.

``openmpt123`` is an external system binary, not a Python dependency, so this module degrades gracefully:
:func:`openmpt123_available` reports whether it is installed, and the render functions raise a clear,
actionable error when it is missing. Rendering is mono float at a fixed interpolation so the ground truth is
reproducible — the tracker formats store no interpolation filter, it is a player setting, and the default
here is 8-tap sinc, OpenMPT's highest-quality mode. Rendering happens in a temporary directory
(``openmpt123 --render`` writes ``<input>.wav`` next to its input), so the caller's files stay untouched.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from trackmod.module.protocol import TrackerModule

from audiotokenizer.audio.io import load_audio

Interpolation = Literal["none", "linear", "cubic", "sinc"]

BINARY: Final = "openmpt123"
# openmpt123 --filter takes interpolation *taps*; more taps = higher-quality (sinc) interpolation.
_INTERPOLATION_TAPS: Final[dict[Interpolation, int]] = {"none": 1, "linear": 2, "cubic": 4, "sinc": 8}
DEFAULT_INTERPOLATION: Final[Interpolation] = "sinc"
DEFAULT_GAIN_DB: Final = 0.0


def openmpt123_available(*, binary: str = BINARY) -> bool:
    """Return whether the ``openmpt123`` binary is discoverable on ``PATH``."""
    return shutil.which(binary) is not None


def _require_binary(binary: str) -> None:
    if not openmpt123_available(binary=binary):
        raise RuntimeError(
            f"{binary!r} not found on PATH; install it (e.g. `apt install openmpt123`) to render modules"
        )


def _render_in_place(
    module_path: Path, *, sample_rate: int, interpolation: Interpolation, gain_db: float, binary: str
) -> tuple[NDArray[np.float64], int]:
    """Run ``openmpt123 --render`` on ``module_path`` and read back the ``<module_path>.wav`` it produces."""
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
        str(module_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} render failed (exit {result.returncode}): {result.stderr.strip()}")
    rendered, rate = load_audio(module_path.with_name(module_path.name + ".wav"), mono=True)
    return np.asarray(rendered, dtype=np.float64).ravel(), rate


def render_file(
    path: Path | str,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Render an existing module file to mono float PCM, returning ``(samples, sample_rate)``.

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


def render_bytes(
    data: bytes,
    *,
    suffix: str,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Write ``data`` to a temporary ``module<suffix>`` file and render it (see :func:`render_file`)."""
    _require_binary(binary)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / f"module{suffix}"
        local.write_bytes(data)
        return _render_in_place(
            local, sample_rate=sample_rate, interpolation=interpolation, gain_db=gain_db, binary=binary
        )


def render_module(
    module: TrackerModule,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Serialize a module and render it, whichever tracker format it happens to be."""
    return render_bytes(
        module.to_bytes(),
        suffix=module.extension,
        sample_rate=sample_rate,
        interpolation=interpolation,
        gain_db=gain_db,
        binary=binary,
    )
