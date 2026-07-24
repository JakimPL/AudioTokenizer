"""Command-line entry point: convert a WAV into an Impulse Tracker module under a byte budget.

``audiotokenizer song.wav -o song.it`` compiles the signal with the default OpenMPT-capable profile and
prints what it spent; ``--strict`` keeps the file a canonical 64-channel Impulse Tracker module. Tuning
knobs (tempo, dictionary size, sparsity floor, and the ``--persistence``/``--sticky`` rate levers) are
exposed so the same command drives experimentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from audiotokenizer.audio.io import SAMPLE_RATE, load_audio
from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import (
    DEFAULT_BUDGET_BYTES,
    DEFAULT_MIN_ENERGY,
    DEFAULT_PCM_BITS,
    DEFAULT_PERSISTENCE,
    DEFAULT_POLARITY_STICKY,
    TokenizerConfig,
)

_DEFAULT_TEMPO: Final = 125  # speed 1, tempo 125 -> 882 frames per row at 44100 Hz
_DEFAULT_ATOMS: Final = 88  # dense 88-channel pool fits the 2 MB budget on a ~170 s song


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a WAV into an Impulse Tracker module under a byte budget")
    parser.add_argument("input", type=Path, help="input WAV at 44100 Hz")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .it path (default: input with .it)")
    parser.add_argument("--strict", action="store_true", help="canonical 64-channel IT (default: 127-channel hacked)")
    parser.add_argument("--tempo", type=int, default=_DEFAULT_TEMPO, help="IT tempo; sets the row length in frames")
    parser.add_argument("--atoms", type=int, default=_DEFAULT_ATOMS, help="dictionary size (the atom pool)")
    parser.add_argument("--min-energy", type=float, default=DEFAULT_MIN_ENERGY, help="drop projections below this")
    parser.add_argument(
        "--persistence", type=float, default=DEFAULT_PERSISTENCE, help="λ penalty for switching atoms (0 = off)"
    )
    parser.add_argument(
        "--sticky",
        type=int,
        default=DEFAULT_POLARITY_STICKY,
        help="keep polarity through sign flips at this code or less",
    )
    parser.add_argument("--pcm-bits", type=int, choices=(8, 16), default=DEFAULT_PCM_BITS, help="stored sample depth")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_BYTES, help="byte budget to report against")
    parser.add_argument("--wav", type=Path, default=None, help="also write the reconstruction as a WAV for A/B")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    signal, sample_rate = load_audio(args.input, mono=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sample_rate}")

    config = TokenizerConfig(
        profile="strict" if args.strict else "hacked",
        tempo=args.tempo,
        n_atoms=args.atoms,
        min_energy=args.min_energy,
        persistence=args.persistence,
        polarity_sticky=args.sticky,
        pcm_bits=args.pcm_bits,
        budget_bytes=args.budget,
        name=args.input.stem,
    )
    compiled = compile_signal(signal, config)

    output = args.output or args.input.with_suffix(".it")
    compiled.save(output)
    print(compiled.summary())
    print(f"  wrote         {output}")
    if args.wav is not None:
        compiled.save_reference(args.wav)
        print(f"  reference     {args.wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
