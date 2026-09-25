"""Place each block's chosen atoms onto tracker channels so a surviving atom keeps its channel.

Channels are interchangeable for fidelity — the reconstruction is the same whichever channel plays an
atom — so this step is pure byte-saving: an atom that keeps its channel across rows keeps its note, which
lets the packer re-use it for two bytes a row instead of re-stating it. The rule is greedy and stable: a
still-selected atom holds its channel, a dropped atom frees it, and a newly selected atom takes any free
channel. ``sample_no`` is the 0-based atom-and-polarity slot ``2 * atom + (sign < 0)``, the second sample
being the atom's negation, so a polarity flip re-points the cell and pays a note byte.

``sticky_threshold`` recovers those flips at the margin: a channel keeping its atom holds its previous
polarity through a sign flip whose volume code is that small or smaller, playing the wrong sign on a
near-zero cell (negligible error) rather than paying the note byte. The reconstruction must therefore be
read back off these grids, not the pre-assignment signs, since a sticky cell plays a sign the coefficient
did not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_FREE = -1


@dataclass(frozen=True)
class Assignment:
    """The compiled pattern grids: ``sample_no`` and ``volume`` are both ``(n_blocks, n_channels)``."""

    sample_no: NDArray[np.int64]
    volume: NDArray[np.int64]

    @property
    def n_channels(self) -> int:
        return int(self.sample_no.shape[1])


def assign(
    codes: NDArray[np.int64], signs: NDArray[np.int64], *, n_channels: int, sticky_threshold: int = 0
) -> Assignment:
    """Lay the active cells (``codes > 0``) of each block onto ``n_channels`` channels, keeping survivors put.

    A channel that keeps its atom holds its previous polarity through a sign flip whose code is at or below
    ``sticky_threshold`` (``0`` disables it), which drops the note byte the flip would otherwise cost.

    Raises:
        ValueError: when a block has more active atoms than there are channels.
    """
    code_matrix = np.asarray(codes, dtype=np.int64)
    sign_matrix = np.asarray(signs, dtype=np.int64)
    n_blocks, n_atoms = code_matrix.shape
    active = code_matrix > 0

    sample_no = np.zeros((n_blocks, n_channels), dtype=np.int64)
    volume = np.zeros((n_blocks, n_channels), dtype=np.int64)

    channel_of_atom = [_FREE] * n_atoms
    atom_of_channel = [_FREE] * n_channels
    polarity_of_channel = [0] * n_channels
    for row in range(n_blocks):
        selected = np.flatnonzero(active[row])
        if selected.size > n_channels:
            raise ValueError(f"block {row} has {selected.size} active atoms, over {n_channels} channels")
        selected_set = set(selected.tolist())
        for channel in range(n_channels):
            atom = atom_of_channel[channel]
            if atom != _FREE and atom not in selected_set:
                channel_of_atom[atom] = _FREE
                atom_of_channel[channel] = _FREE
        free = [channel for channel in range(n_channels) if atom_of_channel[channel] == _FREE]
        next_free = 0
        for atom in selected.tolist():
            channel = channel_of_atom[atom]
            held = channel != _FREE
            if not held:
                channel = free[next_free]
                next_free += 1
                channel_of_atom[atom] = channel
                atom_of_channel[channel] = atom
            code = int(code_matrix[row, atom])
            new_polarity = 1 if sign_matrix[row, atom] < 0 else 0
            if held and new_polarity != polarity_of_channel[channel] and code <= sticky_threshold:
                polarity = polarity_of_channel[channel]
            else:
                polarity = new_polarity
                polarity_of_channel[channel] = new_polarity
            sample_no[row, channel] = 2 * atom + polarity
            volume[row, channel] = code
    return Assignment(sample_no=sample_no, volume=volume)
