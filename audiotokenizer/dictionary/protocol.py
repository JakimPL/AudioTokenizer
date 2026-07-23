from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Dictionary"]


@runtime_checkable
class Dictionary(Protocol):
    """A source of atoms to approximate a block matrix with.

    An implementation turns the ``(n_blocks, block_len)`` matrix into up to
    ``n_atoms`` unit-norm atoms of shape ``(n_atoms, block_len)``, the shared
    vocabulary every channel draws its per-row sample from.
    """

    def learn(self, blocks: NDArray[np.float64], n_atoms: int) -> NDArray[np.float64]: ...
