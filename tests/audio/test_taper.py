from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from audiotokenizer.audio.framing import frame, tukey_window
from audiotokenizer.audio.metrics import boundary_click_db
from audiotokenizer.coding.quantisation import store_atoms
from audiotokenizer.dictionary.learned import LearnedDictionary


def test_tukey_window_zeroes_the_endpoints_and_holds_the_middle() -> None:
    window = tukey_window(512, 0.1)
    assert window[0] == 0.0 and window[-1] == 0.0  # exactly silent edges, not merely small
    assert window[256] == pytest.approx(1.0)  # flat, unattenuated interior
    assert np.allclose(window, window[::-1])  # symmetric
    assert np.all((window >= 0.0) & (window <= 1.0))


def test_tukey_alpha_zero_is_a_rectangle() -> None:
    assert np.array_equal(tukey_window(64, 0.0), np.ones(64))


def test_boundary_click_flags_block_boundary_jumps() -> None:
    block_len = 100
    ramp = np.linspace(0.0, 1.0, block_len, endpoint=False)
    clicking = np.tile(ramp, 20)  # resets to 0 each block -> a jump at every boundary
    smooth = np.sin(2 * np.pi * np.arange(20 * block_len) / block_len)  # continuous across boundaries
    # A per-block reset is a strong boundary jump; a signal that flows across boundaries is not.
    assert boundary_click_db(clicking, block_len) > 0.0
    assert boundary_click_db(clicking, block_len) > boundary_click_db(smooth, block_len) + 12.0


def _tapered_atoms(seed: int, block_len: int, alpha: float, n_atoms: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(50 * block_len)
    matrix = frame(signal, block_len) * tukey_window(block_len, alpha)
    return LearnedDictionary().learn(matrix, n_atoms)


def test_tapered_frames_learn_atoms_that_vanish_at_the_edges() -> None:
    # Every tapered frame is exactly zero at its endpoints, so the row space — and every atom — is too.
    atoms = _tapered_atoms(seed=0, block_len=320, alpha=0.06, n_atoms=24)
    assert np.allclose(atoms[:, 0], 0.0, atol=1e-12)
    assert np.allclose(atoms[:, -1], 0.0, atol=1e-12)


def test_stored_pcm_is_exactly_silent_at_the_edges() -> None:
    # After peak-normalisation and quantisation the tiny residual rounds to a hard zero, so the played
    # one-shot sample starts and ends in silence — the property that makes a retrigger click-free.
    atoms = _tapered_atoms(seed=1, block_len=320, alpha=0.06, n_atoms=24)
    stored = store_atoms(atoms, np.ones(atoms.shape[0]), bits=8)
    assert np.all(stored.pcm[:, [0, -1]] == 0.0)
