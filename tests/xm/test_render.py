"""Ground-truth render checks for the XM writer, run only when ``openmpt123`` is installed.

These are the render-dependent half of the XM verification: they drive the real tracker and A/B its output
against the reconstruction the codec predicts. A high correlation validates the whole XM playback chain at
once — the keymap routing, the relative-note/finetune tuning, the gain baked into the PCM against the
volume-column coefficient, and the delta encoding — because any of those being wrong would make openmpt
play different audio than the model expects. They ``skip`` where the binary is absent, matching the rest of the codebase's graceful
degradation around ``openmpt123``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.tracker.render import openmpt123_available

pytestmark = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 is not installed")


@pytest.fixture
def signal() -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    time = np.arange(2 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def _normalized_correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Zero-lag correlation of two signals, invariant to openmpt's absolute mixing gain."""
    n = min(a.size, b.size)
    x = a[:n] - a[:n].mean()
    y = b[:n] - b[:n].mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def test_xm_render_tracks_the_reconstruction(signal: NDArray[np.float64]) -> None:
    # A short-row XM (tempo 441 -> 250 frames/row, impossible in IT) must render to what the model predicts.
    compiled = compile_signal(signal, TokenizerConfig.load(profile="hacked", format="xm", tempo=441, n_atoms=24))
    rendered, rate = compiled.render()
    assert rate == 44100
    assert float(np.sqrt(np.mean(rendered**2))) > 1e-4  # not silence
    assert _normalized_correlation(rendered, compiled.estimate) > 0.5


def test_xm_render_is_comparable_to_it_on_the_same_signal(signal: NDArray[np.float64]) -> None:
    # At a tempo both formats can store, XM and IT should render to nearly the same thing.
    common = dict(profile="strict", tempo=250, n_atoms=24)
    xm_rendered, _ = compile_signal(signal, TokenizerConfig(format="xm", **common)).render()
    it_compiled = compile_signal(signal, TokenizerConfig(format="it", **common))
    it_rendered, _ = it_compiled.render()
    assert _normalized_correlation(xm_rendered, it_compiled.estimate) > 0.5
    assert _normalized_correlation(xm_rendered, it_rendered) > 0.5
