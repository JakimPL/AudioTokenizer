"""Render ``.XM`` modules to audio with ``openmpt123`` — a thin XM binding over the shared render driver.

The openmpt123 driver is format-agnostic and lives in :mod:`audiotokenizer.tracker.render`; this module
only binds it to the XM writer, so :func:`render_module` renders an :class:`~audiotokenizer.xm.module.
XMModule` and :func:`render_xm` renders an existing ``.xm`` file. See the shared module for the render
settings and the graceful degradation when ``openmpt123`` is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.tracker.render import (
    BINARY,
    DEFAULT_GAIN_DB,
    DEFAULT_INTERPOLATION,
    Interpolation,
    openmpt123_available,
    render_bytes,
    render_file,
)
from audiotokenizer.xm.module import XMModule, write_xm_module

__all__ = ["Interpolation", "openmpt123_available", "render_xm", "render_module", "DEFAULT_INTERPOLATION"]


def render_xm(
    path: Path | str,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Render an existing ``.XM`` file to mono float PCM, returning ``(samples, sample_rate)``."""
    return render_file(path, sample_rate=sample_rate, interpolation=interpolation, gain_db=gain_db, binary=binary)


def render_module(
    module: XMModule,
    *,
    sample_rate: int,
    interpolation: Interpolation = DEFAULT_INTERPOLATION,
    gain_db: float = DEFAULT_GAIN_DB,
    binary: str = BINARY,
) -> tuple[NDArray[np.float64], int]:
    """Write ``module`` to a temporary ``.XM`` file and render it (see :func:`render_xm`)."""
    return render_bytes(
        write_xm_module(module),
        suffix=".xm",
        sample_rate=sample_rate,
        interpolation=interpolation,
        gain_db=gain_db,
        binary=binary,
    )
