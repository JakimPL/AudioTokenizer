"""Rip the samples back out of a written module with ``xmodits``, to validate what the writer stored.

``xmodits`` is an independent, Rust-backed sample ripper that reads every format ``trackmod`` writes, so
decoding a file with a parser we did *not* write is the strongest cheap check that the byte layout is
correct. This module is a thin wrapper: it dumps every sample to WAV in a temp dir and reads them back as
float PCM. It depends on the dev-only ``xmodits-py`` and stays out of the package's runtime surface, so an
install without the validation tooling still imports cleanly.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xmodits
from numpy.typing import NDArray

from audiotokenizer.audio.io import load_audio


@dataclass(frozen=True)
class RippedSample:
    """One sample recovered from a module file: its index, name, tagged rate and float PCM."""

    index: int
    name: str
    sample_rate: int
    pcm: NDArray[np.float64]

    @property
    def frames(self) -> int:
        return int(self.pcm.size)


def _parse_name(stem: str) -> tuple[int, str]:
    """Split xmodits' ``"NN - name"`` filename stem into ``(index, name)`` (index 0 if unprefixed)."""
    head, separator, tail = stem.partition(" - ")
    if separator and head.strip().isdigit():
        return int(head), tail
    return 0, stem


def read_samples(path: Path | str) -> list[RippedSample]:
    """Extract every sample from ``path`` with xmodits and return them as float PCM, ordered by index."""
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        xmodits.dump(str(Path(path)), str(destination), format="wav")
        ripped: list[RippedSample] = []
        for wav in sorted(destination.glob("*.wav")):
            data, rate = load_audio(wav, mono=True)
            index, name = _parse_name(wav.stem)
            pcm = np.asarray(data, dtype=np.float64).ravel()
            ripped.append(RippedSample(index=index, name=name, sample_rate=rate, pcm=pcm))
    return sorted(ripped, key=lambda sample: sample.index)
