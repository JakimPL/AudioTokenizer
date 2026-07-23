from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..it.spec import MAX_VOLUME

__all__ = ["Quantised", "quantise", "quantise_pcm"]


@dataclass(frozen=True)
class Quantised:
    codes: NDArray[np.int64]  # (n_rows, n_atoms) volume levels 0..levels
    signs: NDArray[np.int64]  # (n_rows, n_atoms) polarity -1/0/+1
    gains: NDArray[np.float64]  # (n_atoms,) per-atom static gain, baked into PCM
    values: NDArray[np.float64]  # (n_rows, n_atoms) the coefficient actually played


def quantise(coef: NDArray[np.float64], *, levels: int = MAX_VOLUME, signed: bool = True) -> Quantised:
    """Round coefficients onto the 0..``levels`` volume column.

    Each atom carries one static gain -- its largest coefficient over the song --
    baked into the stored PCM, so the volume column spans that atom's own range
    at full resolution. The gain is applied to the reconstruction here exactly as
    the module plays it, so ``values`` is what the tracker produces, not the fit.
    """
    matrix = np.asarray(coef, dtype=np.float64)
    gains = np.max(np.abs(matrix), axis=0)
    gains[gains == 0.0] = 1.0
    scaled = matrix / gains
    if not signed:
        scaled = np.maximum(scaled, 0.0)
    magnitude = np.abs(scaled)
    codes = np.clip(np.round(magnitude * levels).astype(np.int64), 0, levels)
    signs = np.where(scaled < 0.0, -1, 1).astype(np.int64)
    signs[codes == 0] = 0
    values = signs * codes / levels * gains
    return Quantised(codes=codes, signs=signs, gains=gains, values=values)


def quantise_pcm(atoms: NDArray[np.float64], *, bits: int = 8) -> NDArray[np.float64]:
    """Round each atom to signed ``bits``-bit PCM, peak-normalised per atom.

    At 8 bits the quantisation floor sits ~48 dB below a coefficient error that
    is already near -13 dB, so it is inaudible while halving the stored bytes.
    """
    matrix = np.asarray(atoms, dtype=np.float64)
    peak = np.max(np.abs(matrix), axis=1, keepdims=True)
    peak[peak == 0.0] = 1.0
    top = 2 ** (bits - 1) - 1
    stored = np.round(matrix / peak * top) / top * peak
    return np.asarray(stored, dtype=np.float64)
