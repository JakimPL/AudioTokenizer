"""Ground-truth render checks, run only when ``openmpt123`` is installed.

These are the render-dependent half of the verification: they drive the real tracker and A/B its output
against the reconstruction the codec predicts. A high correlation validates a whole playback chain at once
— the keymap routing, the tuning each format derives from a sample's rate, where the atom's gain lives
against the volume-column coefficient, and the PCM encoding — because any of those being wrong would make
openmpt play different audio than the model expects. They ``skip`` where the binary is absent, matching
the rest of the codebase's graceful degradation around ``openmpt123``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.limits.compliance import Compliance

from audiotokenizer.module.format import Format
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.render.openmpt import openmpt123_available

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


@pytest.mark.parametrize("module_format", tuple(Format))
def test_render_tracks_the_reconstruction(signal: NDArray[np.float64], module_format: Format) -> None:
    config = TokenizerConfig(compliance=Compliance.CANONICAL, format=module_format, tempo=250, n_atoms=24)
    compiled = compile_signal(signal, config)
    rendered, rate = compiled.render()
    assert rate == 44100
    assert float(np.sqrt(np.mean(rendered**2))) > 1e-4  # not silence
    assert _normalized_correlation(rendered, compiled.estimate) > 0.5


def test_a_short_row_xm_renders_what_the_model_predicts(signal: NDArray[np.float64]) -> None:
    # Tempo 441 gives 250 frames per row, a resolution IT's single-byte tempo cannot ask for.
    compiled = compile_signal(
        signal, TokenizerConfig(compliance=Compliance.EXTENDED, format=Format.XM, tempo=441, n_atoms=24)
    )
    rendered, rate = compiled.render()
    assert rate == 44100
    assert float(np.sqrt(np.mean(rendered**2))) > 1e-4
    assert _normalized_correlation(rendered, compiled.estimate) > 0.5


def test_the_two_formats_render_to_nearly_the_same_thing(signal: NDArray[np.float64]) -> None:
    # At a tempo both formats store and a compliance level both honour, the only differences left are how
    # each one encodes the same song — so their renders must agree.
    common = dict(compliance=Compliance.CANONICAL, tempo=250, n_atoms=24)
    xm_rendered, _ = compile_signal(signal, TokenizerConfig(format=Format.XM, **common)).render()
    it_compiled = compile_signal(signal, TokenizerConfig(format=Format.IT, **common))
    it_rendered, _ = it_compiled.render()
    assert _normalized_correlation(xm_rendered, it_compiled.estimate) > 0.5
    assert _normalized_correlation(xm_rendered, it_rendered) > 0.5
