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
