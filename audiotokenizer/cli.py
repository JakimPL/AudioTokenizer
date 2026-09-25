"""Command-line entry point: convert a WAV into an Impulse Tracker or FastTracker 2 module.

``audiotokenizer song.wav -o song.it`` compiles the signal at the default, extended compliance level, which
plays in OpenMPT and libopenmpt, and prints what it spent; ``-o song.xm`` (or ``--format xm``) writes a
FastTracker 2 module instead, whose tempo reaches 1000 for rows far shorter than IT's 255 ceiling allows.
``--strict`` keeps the file a canonical tracker module (IT: 64 channels and 49 atoms; XM: 32 channels).
Tuning knobs (tempo, speed, dictionary size, sparsity floor, and the ``--persistence``/``--sticky`` rate
levers) are exposed so the same command drives experimentation. ``--plan`` instead searches the config
lattice for the best-sounding IT module that still fits ``--budget``, printing the cost-vs-quality frontier
it settled on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray
from trackmod.limits.compliance import Compliance

from audiotokenizer.audio.io import SAMPLE_RATE, load_audio
from audiotokenizer.module.format import Format
from audiotokenizer.pipeline.compiler import CompiledModule, compile_signal
from audiotokenizer.pipeline.config import (
    DEFAULT_BUDGET_BYTES,
    DEFAULT_COMPLIANCE,
    DEFAULT_MIN_ENERGY,
    DEFAULT_PCM_BITS,
    DEFAULT_PERSISTENCE,
    DEFAULT_POLARITY_STICKY,
    DEFAULT_SPEED,
    DEFAULT_TAPER_ALPHA,
    TokenizerConfig,
    max_atoms_for,
)
from audiotokenizer.pipeline.planner import PLANNED_FORMAT, format_frontier, plan_compilation

_DEFAULT_TEMPO: Final = 125  # speed 1, tempo 125 -> 882 frames per row at 44100 Hz
_DEFAULT_ATOMS: Final = 88  # dense 88-channel pool fits the 2 MB budget on a ~170 s song


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a WAV into an Impulse Tracker or FastTracker 2 module")
    parser.add_argument("input", type=Path, help="input WAV at 44100 Hz")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="output path (default: input with the format suffix)"
    )
    parser.add_argument(
        "--format",
        type=Format,
        choices=tuple(Format),
        default=None,
        help="output format: it (Impulse Tracker) or xm (FastTracker 2); default inferred from -o suffix, else it",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="hold the module to what the tracker's own editor accepts (IT: 64 channels, 49 atoms; XM: 32 channels); "
        "the default plays in OpenMPT",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="search configs for the best quality under --budget (IT only; ignores --tempo/--atoms/--min-energy)",
    )
    parser.add_argument(
        "--tempo", type=int, default=_DEFAULT_TEMPO, help="tempo (BPM); sets the row length. XM reaches past 255"
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="ticks per row; raise to lengthen the row")
    parser.add_argument(
        "--atoms",
        type=int,
        default=None,
        help=f"dictionary size (the atom pool); default {_DEFAULT_ATOMS}, capped at what the compliance level routes",
    )
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
    parser.add_argument(
        "--taper",
        type=float,
        default=DEFAULT_TAPER_ALPHA,
        help="Tukey edge-taper fraction that zeros atom endpoints against clicks (0 = raw, clicking)",
    )
    parser.add_argument("--pcm-bits", type=int, choices=(8, 16), default=DEFAULT_PCM_BITS, help="stored sample depth")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_BYTES, help="byte budget to report against")
    parser.add_argument("--wav", type=Path, default=None, help="also write the reconstruction as a WAV for A/B")
    return parser


def _plan(args: argparse.Namespace, signal: NDArray[np.float64], output: Path) -> CompiledModule:
    """Run the config search and print the frontier it chose from, returning the winning module.

    Each time the best improves it is written to ``output``, so an interrupt, crash, or kill still leaves
    the best-so-far module on disk instead of discarding the whole sweep.
    """

    def checkpoint(module: CompiledModule) -> None:
        module.save(output)

    plan = plan_compilation(
        signal,
        budget_bytes=args.budget,
        compliances=(Compliance.CANONICAL,) if args.strict else (Compliance.EXTENDED, Compliance.CANONICAL),
        pcm_bits=args.pcm_bits,
        taper_alpha=args.taper,
        name=args.input.stem,
        on_best=checkpoint,
    )
    print("planner frontier (cost vs quality; last row chosen):")
    print(format_frontier(plan.frontier))
    print()
    if plan.interrupted:
        print("  note          interrupted — reporting the best module found before the stop")
    if plan.failures:
        print(f"  note          {plan.failures} candidate(s) raised while assembling and were skipped")
    if not plan.within_budget:
        print(f"  note          nothing fit {args.budget / 1024:.0f} KB; this is the closest-to-budget module")
    return plan.module


def _refuse(compiled: CompiledModule) -> str:
    """The message for a module the chosen format will not store, listing every bound it breaks.

    A compilation can be perfectly good audio and still be unwritable — a canonical Impulse Tracker
    pattern has a 32-row floor, so a very short signal at a long row simply is not one. Naming the bounds
    and the level they were read at is what tells a caller whether to change the timing or drop
    ``--strict``, so it is reported rather than raised as a traceback.
    """
    lines = [f"  {violation}" for violation in compiled.violations]
    hint = " (drop --strict to write an extended module)" if compiled.config.compliance is Compliance.CANONICAL else ""
    return "\n".join([f"cannot write this {compiled.config.format} module{hint}:", *lines])


def _compliance(args: argparse.Namespace) -> Compliance:
    """The level every bound is read at: canonical under ``--strict``, the default level otherwise."""
    return Compliance.CANONICAL if args.strict else DEFAULT_COMPLIANCE


def _atoms(args: argparse.Namespace, module_format: Format, compliance: Compliance) -> int:
    """The dictionary size to compile with: ``--atoms`` when given, else the default pool.

    The default pool is capped at what the format routes at ``compliance``, so a plain ``--strict`` run
    compiles the widest canonical dictionary. An explicit ``--atoms`` is checked by the config as stated.
    """
    stated: int | None = args.atoms
    if stated is not None:
        return stated

    return min(_DEFAULT_ATOMS, max_atoms_for(module_format, compliance))


def _resolve_format(args: argparse.Namespace) -> Format:
    """Pick the output format: an explicit ``--format``, else the ``-o`` suffix, else IT."""
    if args.format is not None:
        return Format(args.format)
    if args.output is not None and args.output.suffix.lower() == f".{Format.XM}":
        return Format.XM
    return Format.IT


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    signal, sample_rate = load_audio(args.input, mono=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sample_rate}")

    module_format = _resolve_format(args)
    output = args.output or args.input.with_suffix(f".{module_format}")
    if args.plan:
        if module_format is not PLANNED_FORMAT:
            raise SystemExit("the --plan config search is IT-only; drop --format xm (or -o *.xm) to plan")
        compiled = _plan(args, signal, output)
    else:
        compliance = _compliance(args)
        config = TokenizerConfig(
            compliance=compliance,
            format=module_format,
            tempo=args.tempo,
            speed=args.speed,
            n_atoms=_atoms(args, module_format, compliance),
            min_energy=args.min_energy,
            persistence=args.persistence,
            polarity_sticky=args.sticky,
            taper_alpha=args.taper,
            pcm_bits=args.pcm_bits,
            budget_bytes=args.budget,
            name=args.input.stem,
        )
        compiled = compile_signal(signal, config)

    if not compiled.writable:
        raise SystemExit(_refuse(compiled))

    compiled.save(output)
    print(compiled.summary())
    print(f"  wrote         {output}")
    if args.wav is not None:
        compiled.save_reference(args.wav)
        print(f"  reference     {args.wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
