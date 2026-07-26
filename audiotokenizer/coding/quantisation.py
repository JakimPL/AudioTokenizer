"""Turn continuous coefficients and atoms into what the module actually stores: the 6-bit volume column
and the signed PCM samples.

Two quantizers meet here. :func:`quantise` rounds each block's projection onto the 0..64 volume column,
giving every atom its own static gain so the column spans that atom's range at full resolution.
:func:`store_atoms` writes the atoms as signed PCM: each atom is peak-normalized to fill the sample's
range at full bit depth, and its static gain rides on the sample's own global-volume field — a second
0..64 lattice — so a quiet atom keeps every bit of resolution instead of collapsing toward zero. One
shared headroom scales those gains onto that lattice, so the tracker reproduces the fitted signal
uniformly attenuated by ``1 / scale``. The negation of each atom is stored as its second sample, which is
how a signed coefficient plays through the volume column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from trackmod.spec.levels import MAX_VOLUME

_DEFAULT_PCM_BITS = 8
_MIN_HEADROOM = 1.0  # keep the summed reconstruction at or below full scale


@dataclass(frozen=True)
class Quantised:
    """The volume-column encoding of a coefficient matrix, all arrays aligned to ``coef``'s shape."""

    codes: NDArray[np.int64]  # (n_blocks, n_atoms) volume levels 0..levels
    signs: NDArray[np.int64]  # (n_blocks, n_atoms) polarity -1/0/+1
    gains: NDArray[np.float64]  # (n_atoms,) per-atom static gain, baked into the stored PCM
    values: NDArray[np.float64]  # (n_blocks, n_atoms) the fitted coefficient the column reproduces


@dataclass(frozen=True)
class StoredAtoms:
    """Atoms as they are stored: peak-normalized PCM plus the per-atom global volume carrying the gain.

    The tracker's channel sum equals the fitted reconstruction divided by ``scale``; a caller comparing
    against the reference multiplies its render (or the reconstruction) back by ``scale``.
    """

    pcm: NDArray[np.float64]  # (n_atoms, block_len) peak-normalized, 8-bit-rounded, in [-1, 1]
    global_volume: NDArray[np.int64]  # (n_atoms,) per-atom sample global volume 0..levels (the gain lattice)
    scale: float


def quantise(coef: NDArray[np.float64], *, levels: int = MAX_VOLUME, signed: bool = True) -> Quantised:
    """Round coefficients onto the 0..``levels`` volume column against each atom's own static gain.

    ``coef`` is ``(n_blocks, n_atoms)``. Each atom's gain is its largest coefficient over the blocks, so
    the column spans that atom's range at full resolution; ``values`` is the coefficient the column
    reproduces, so a caller reconstructs with exactly what the module plays.
    """
    matrix = np.asarray(coef, dtype=np.float64)
    gains = np.max(np.abs(matrix), axis=0)
    gains[gains == 0.0] = 1.0
    scaled = matrix / gains
    if not signed:
        scaled = np.maximum(scaled, 0.0)
    codes = np.clip(np.round(np.abs(scaled) * levels).astype(np.int64), 0, levels)
    signs = np.where(scaled < 0.0, -1, 1).astype(np.int64)
    signs[codes == 0] = 0
    values = signs * codes / levels * gains
    return Quantised(codes=codes, signs=signs, gains=gains, values=values)


def store_atoms(
    unit_atoms: NDArray[np.float64],
    gains: NDArray[np.float64],
    *,
    bits: int = _DEFAULT_PCM_BITS,
    levels: int = MAX_VOLUME,
    min_headroom: float = _MIN_HEADROOM,
) -> StoredAtoms:
    """Peak-normalize unit atoms to full bit depth and map each gain onto the sample global-volume lattice.

    ``unit_atoms`` is ``(n_atoms, block_len)`` with unit L2 norm; ``gains`` is per atom. Each atom fills
    the sample range at full ``bits``-bit resolution, and its amplitude (gain times peak) is quantized to
    a 1..``levels`` global-volume against the loudest atom's amplitude (floored at ``min_headroom``), so
    the stored samples stay within full scale and the reconstruction is attenuated by ``1 / headroom``.
    """
    atoms = np.asarray(unit_atoms, dtype=np.float64)
    atom_gains = np.asarray(gains, dtype=np.float64)
    peaks = np.max(np.abs(atoms), axis=1) if atoms.size else np.zeros(atoms.shape[0])
    safe_peaks = np.where(peaks > 0.0, peaks, 1.0)
    amplitude = atom_gains * safe_peaks
    headroom = max(float(np.max(amplitude)) if atoms.size else 0.0, min_headroom)
    global_volume = np.clip(np.round(levels * amplitude / headroom), 1, levels).astype(np.int64)
    normalized = atoms / safe_peaks[:, None]
    full = float(2 ** (bits - 1))
    quantized = np.clip(np.round(normalized * full), -full, full - 1) / full
    return StoredAtoms(
        pcm=np.asarray(quantized, dtype=np.float64),
        global_volume=global_volume,
        scale=headroom,
    )
