from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.audio.framing import frame, unframe
from audiotokenizer.audio.io import SAMPLE_RATE, load_audio, normalise
from audiotokenizer.audio.metrics import (
    crest_factor,
    envelope_error_db,
    log_spectral_distance,
    mel_distance_db,
    snr_db,
)
from audiotokenizer.coding.cost import module_bytes
from audiotokenizer.coding.quantisation import quantise, quantise_pcm
from audiotokenizer.dictionary.learned import LearnedDictionary

DEFAULT_SONG = Path("/mnt/d/Projekty/Python/SampleToWave/audio/song.wav")

# The dense design point: every atom owns a channel for the whole song, so the
# note never changes and selection collapses to "use them all". These are the
# rows the byte model and the reconstruction must reproduce.
CASES: Tuple[Tuple[str, int, int], ...] = (
    ("hacked 127ch", 735, 127),
    ("hacked 127ch", 882, 127),
    ("hacked 127ch", 1102, 127),
    ("strict  64ch", 882, 64),
)


def dense_assignment(
    quant_codes: NDArray[np.int64], quant_signs: NDArray[np.int64]
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Atom ``a`` lives on channel ``a`` every row; polarity picks +s / -s."""
    n_atoms = quant_codes.shape[1]
    atom = np.arange(n_atoms, dtype=np.int64)[None, :]
    sample_no = 2 * atom + (quant_signs < 0).astype(np.int64)
    sample_no = np.where(quant_codes > 0, sample_no, -1)
    return sample_no, quant_codes


def run(signal: NDArray[np.float64]) -> None:
    dictionary = LearnedDictionary()
    duration = signal.size / SAMPLE_RATE
    columns = ("profile", "T", "atoms", "KB", "melLSD", "LSD", "SNR", "env", "crest")
    widths = (14, 6, 7, 8, 8, 7, 7, 7, 7)
    header = "".join(f"{name:>{width}}" for name, width in zip(columns, widths))
    print(f"song {duration:.1f} s  --  dense, one 8-bit atom per channel")
    print(header)
    print("-" * len(header))

    for label, block_len, n_atoms in CASES:
        matrix = frame(signal, block_len)
        atoms = quantise_pcm(dictionary.learn(matrix, n_atoms))
        norm = np.linalg.norm(atoms, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        unit = atoms / norm

        coef = matrix @ unit.T
        quant = quantise(coef, signed=True)
        estimate = unframe(quant.values @ unit, signal.size)

        sample_no, volume = dense_assignment(quant.codes, quant.signs)
        cost = module_bytes(sample_no, volume, n_stored_samples=2 * n_atoms, pcm_frames=block_len)
        print(
            f"{label:<14}{block_len:>6}{n_atoms:>7}{cost.kilobytes:>8.0f}"
            f"{mel_distance_db(signal, estimate):>8.2f}"
            f"{log_spectral_distance(signal, estimate):>7.2f}"
            f"{snr_db(signal, estimate):>7.2f}"
            f"{envelope_error_db(signal, estimate):>7.2f}"
            f"{crest_factor(estimate):>7.2f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song", nargs="?", type=Path, default=DEFAULT_SONG)
    args = parser.parse_args()
    signal, sample_rate = load_audio(args.song)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sample_rate}")
    run(normalise(signal))


if __name__ == "__main__":
    main()
