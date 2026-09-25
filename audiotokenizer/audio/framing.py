"""Cut a signal into the ``(n_blocks, block_len)`` matrix the whole codec reads, and stitch it back.

Rows are time steps and columns are frames within a block — the ``(rows, ...)`` layout the dictionary,
selection, quantiser and cost model all share, so a block maps one-to-one onto a pattern row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def n_blocks_for(n_samples: int, block_len: int) -> int:
    """Number of blocks needed to cover ``n_samples``, padding the last one."""
    return int(np.ceil(n_samples / block_len)) if n_samples else 0


def frame(signal: NDArray[np.float64], block_len: int) -> NDArray[np.float64]:
    """Cut ``signal`` into ``(n_blocks, block_len)`` rows, zero-padding the tail."""
    flat = np.asarray(signal, dtype=np.float64).ravel()
    count = n_blocks_for(flat.size, block_len)
    padded = np.zeros(count * block_len, dtype=np.float64)
    padded[: flat.size] = flat
    return padded.reshape(count, block_len)


def unframe(blocks: NDArray[np.float64], n_samples: int | None = None) -> NDArray[np.float64]:
    """Concatenate ``(n_blocks, block_len)`` rows back into a signal, trimmed to ``n_samples`` if given."""
    flat = np.asarray(blocks, dtype=np.float64).reshape(-1)
    return flat if n_samples is None else flat[:n_samples]


def tukey_window(block_len: int, alpha: float) -> NDArray[np.float64]:
    """A Tukey (tapered-cosine) window of ``block_len`` frames.

    ``alpha`` is the fraction of the window spent tapering, split between the two ends: ``0`` is a
    rectangle (no taper — the endpoints stay wherever the atom lands), ``1`` is a full Hann. For any
    ``alpha > 0`` the window is exactly zero at both endpoints and rises smoothly, so a block matrix
    multiplied by it lands entirely in the zero-endpoint subspace: every learned atom then starts and ends
    at silence and a retriggered one-shot sample cannot click.
    """
    if block_len <= 1 or alpha <= 0.0:
        return np.ones(max(block_len, 0), dtype=np.float64)
    span = min(alpha, 1.0)
    edge = span / 2.0
    position = np.linspace(0.0, 1.0, block_len)
    window = np.ones(block_len, dtype=np.float64)
    rising = position < edge
    falling = position > 1.0 - edge
    window[rising] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * position[rising] / span - 1.0)))
    window[falling] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * position[falling] / span - 2.0 / span + 1.0)))
    return window
