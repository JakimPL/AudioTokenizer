from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["LearnedDictionary"]


class LearnedDictionary:
    """The leading right-singular directions of the block matrix.

    Eckart-Young makes these the best rank-``k`` basis for this exact block set
    under Frobenius error, and they are orthonormal, so keeping a row's largest
    projections is exactly the optimal sparse choice. The atoms carry a whole
    waveform shape, sub-row structure included, which is what a stationary sine
    cannot.
    """

    def learn(self, blocks: NDArray[np.float64], n_atoms: int) -> NDArray[np.float64]:
        matrix = np.asarray(blocks, dtype=np.float64)
        _, _, right = np.linalg.svd(matrix, full_matrices=False)
        atoms = right[: min(n_atoms, right.shape[0])]
        return np.ascontiguousarray(atoms, dtype=np.float64)
