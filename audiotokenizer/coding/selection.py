"""Choose, per block, which atoms approximate it — the rate lever of the codec.

The dictionary is orthonormal, so a block's squared error from keeping a set of atoms is its energy minus
the kept squared projections; keeping the largest projections is therefore exactly optimal, and this is
what makes "sort by coefficient energy" sound rather than a heuristic. Two levers narrow the set below the
dense case: a hard cap per row (the channel count) and a global energy floor that drops projections too
small to be worth a cell. Both trade fidelity for a smaller file along the same optimal ordering.

A third lever, ``persistence`` (λ), trades a little of that optimality for pattern bytes: when a row must
cull to the cap it penalises every atom the previous row did not keep, so a channel holds its atom instead
of switching and paying a note byte. It only breaks ties for contested slots — a row with spare capacity
keeps every eligible atom regardless — and at ``persistence = 0`` selection stays independent per row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Selection:
    """Which atoms each block keeps: ``mask`` is ``(n_blocks, n_atoms)``, ``kept_per_row`` its row sums."""

    mask: NDArray[np.bool_]

    @property
    def kept_per_row(self) -> NDArray[np.int64]:
        return np.asarray(self.mask.sum(axis=1), dtype=np.int64)


def select(
    coef: NDArray[np.float64], *, max_per_row: int, min_energy: float = 0.0, persistence: float = 0.0
) -> Selection:
    """Keep the ``max_per_row`` largest-energy atoms per block, dropping any below ``min_energy``.

    ``coef`` is ``(n_blocks, n_atoms)`` of projections onto unit atoms. A row keeps fewer than
    ``max_per_row`` atoms when the energy floor removes some, which is how a smaller budget spends fewer
    cells on quiet blocks. A positive ``persistence`` (λ) favours atoms the previous row kept when a row
    is over the cap, trading a little captured energy for fewer atom switches.
    """
    matrix = np.asarray(coef, dtype=np.float64)
    n_blocks, n_atoms = matrix.shape
    energy = matrix * matrix
    if persistence > 0.0:
        return Selection(
            mask=_select_persistent(energy, max_per_row=max_per_row, min_energy=min_energy, penalty=persistence)
        )
    keep = energy > min_energy
    if max_per_row < n_atoms:
        cut = n_atoms - max_per_row
        top = np.argpartition(energy, cut, axis=1)[:, cut:]  # (n_blocks, max_per_row) largest-energy indices
        top_mask = np.zeros((n_blocks, n_atoms), dtype=bool)
        np.put_along_axis(top_mask, top, True, axis=1)
        keep &= top_mask
    return Selection(mask=keep)


def _select_persistent(
    energy: NDArray[np.float64], *, max_per_row: int, min_energy: float, penalty: float
) -> NDArray[np.bool_]:
    """Row-by-row selection that penalises by ``penalty`` any atom the previous row did not keep.

    Where a row has more eligible atoms than the cap, an atom active last row keeps its full energy score
    while a newcomer must clear ``energy − penalty``; below the cap every eligible atom is kept, so the
    penalty only ever settles contested slots. ``energy`` is the squared coefficients ``(n_blocks, n_atoms)``.
    """
    n_blocks, n_atoms = energy.shape
    keep = np.zeros((n_blocks, n_atoms), dtype=bool)
    active = np.zeros(n_atoms, dtype=bool)
    for row in range(n_blocks):
        eligible = np.flatnonzero(energy[row] > min_energy)
        if eligible.size > max_per_row:
            score = energy[row, eligible] - np.where(active[eligible], 0.0, penalty)
            eligible = eligible[np.argsort(score, kind="stable")[::-1][:max_per_row]]
        keep[row, eligible] = True
        active = keep[row]
    return keep
