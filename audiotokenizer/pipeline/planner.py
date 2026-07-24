"""Search the configuration lattice for the best-sounding module under a byte budget.

The codec has more knobs than a person should tune by hand — timing (the row length), pool size, profile,
the sparsity floor, the storage depth — and their interaction with the byte budget is not obvious: a
richer 127-atom dictionary that keeps its best 64 per row can beat a fixed 88, and a shorter row buys time
resolution at the cost of pattern bytes. This planner sweeps that lattice and keeps the config with the
lowest mel distance whose file still fits the budget.

Two facts make the sweep cheap enough to run on a whole song:

* **One SVD per timing.** The learned atoms are singular-ordered, so :func:`_prepare` learns the largest
  pool once per timing and every smaller ``n_atoms`` candidate is a column slice — the whole pool sweep
  costs no extra SVDs (see :class:`~audiotokenizer.pipeline.compiler.Prepared`).
* **A lower-bound prune.** :func:`~audiotokenizer.coding.cost.cost_lower_bound` bounds a candidate's size
  from cell counts alone, so the expensive per-row assignment and the metrics run only for candidates that
  can still fit. Because the bound never exceeds the true size, no feasible candidate is ever pruned.

The sweep is also built never to waste a long run. It holds only the running best module and cheap
frontier records — not every reconstruction — so memory stays flat, and it checkpoints each new best
through an ``on_best`` callback so a crash, kill, or ``Ctrl-C`` still leaves the best-so-far on disk. A
candidate that raises is skipped, and when nothing fits the budget the closest-to-budget module is
returned rather than the run being thrown away. See :func:`plan_compilation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Iterator, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from audiotokenizer.audio.io import SAMPLE_RATE, normalise
from audiotokenizer.coding.cost import cost_lower_bound
from audiotokenizer.coding.quantisation import quantise
from audiotokenizer.coding.selection import select
from audiotokenizer.it.timing import Timing, exact_timings
from audiotokenizer.pipeline.compiler import _POLARITIES, CompiledModule, Prepared, _assemble, _prepare
from audiotokenizer.pipeline.config import DEFAULT_NAME, DEFAULT_TAPER_ALPHA, MAX_ATOMS, Profile, TokenizerConfig

_DEFAULT_N_ATOMS_GRID: Final = (24, 40, 56, 72, 88, 104, MAX_ATOMS)
_DEFAULT_MIN_ENERGY_GRID: Final = (0.0,)
_DEFAULT_PROFILES: Final = ("hacked", "strict")
_DEFAULT_PCM_BITS: Final[Literal[8, 16]] = 8
_MEL_EPS: Final = 1e-9  # a Pareto step must lower mel by at least this to count as an improvement


@dataclass(frozen=True)
class Candidate:
    """One evaluated config and the numbers that decide whether it wins: its size and its mel distance."""

    config: TokenizerConfig
    total_bytes: int
    mel_db: float
    n_atoms_used: int
    n_channels_used: int


@dataclass(frozen=True)
class Plan:
    """The planner's outcome: the chosen module, the cost-vs-mel frontier, and how the sweep ended.

    ``module`` is always a real, compiled module — the lowest-mel one that fit the budget, or, when
    nothing fit, the closest-to-budget fallback (``within_budget`` is then ``False``). A sweep never
    throws its work away: ``interrupted`` marks a run a ``Ctrl-C`` cut short but still finalized on its
    best-so-far, and ``failures`` counts candidates skipped because assembling them raised.
    """

    module: CompiledModule
    frontier: tuple[Candidate, ...]
    interrupted: bool = False
    failures: int = 0

    @property
    def config(self) -> TokenizerConfig:
        return self.module.config

    @property
    def within_budget(self) -> bool:
        """Whether the chosen module actually fits the budget (``False`` for a best-effort fallback)."""
        return self.module.within_budget


def plan_compilation(
    signal: NDArray[np.float64],
    *,
    budget_bytes: int,
    profiles: Sequence[Profile] = _DEFAULT_PROFILES,
    n_atoms_grid: Sequence[int] = _DEFAULT_N_ATOMS_GRID,
    min_energy_grid: Sequence[float] = _DEFAULT_MIN_ENERGY_GRID,
    pcm_bits: Literal[8, 16] = _DEFAULT_PCM_BITS,
    taper_alpha: float = DEFAULT_TAPER_ALPHA,
    name: str = DEFAULT_NAME,
    on_best: Callable[[CompiledModule], None] | None = None,
) -> Plan:
    """Compile ``signal`` with the lowest-mel config whose file fits ``budget_bytes``.

    Sweeps every exact-row timing, and within each the profiles, pool sizes and sparsity floors given.
    The rate levers (persistence, polarity-sticky) stay off — they are last-resort byte knobs, not quality
    wins, so the planner leaves them for a person to pin. ``taper_alpha`` is a single run-level value shared
    by every candidate (it feeds the one SVD per timing), so it is not a search axis and adds no SVDs.

    The sweep never discards its work. Only the running best module is held (not every candidate), so a
    long run stays within one signal's memory rather than accumulating hundreds of reconstructions. Each
    time the best improves, ``on_best`` is called with it — the CLI writes it to disk there, so a crash or
    kill leaves the best-so-far module on disk. A ``KeyboardInterrupt`` stops the sweep and finalizes on the
    best found so far, and a candidate that raises while assembling is counted and skipped, not fatal. When
    nothing fits the budget, the closest-to-budget candidate is compiled and returned (flagged over budget)
    rather than the whole run being thrown away.

    Raises:
        ValueError: only when the grid or signal yields no candidate to compile at all.
    """
    reference = normalise(np.asarray(signal, dtype=np.float64).ravel())
    best: CompiledModule | None = None  # lowest-mel feasible module so far — the only reconstruction held
    best_mel = float("inf")
    candidates: list[Candidate] = []  # cheap records only — the Pareto frontier, no held reconstructions
    fallback: tuple[Prepared, TokenizerConfig] | None = None  # smallest-bound seed, compiled only if needed
    fallback_bound = 0
    failures = 0
    interrupted = False

    try:
        for timing in tqdm(exact_timings(frame_rate=SAMPLE_RATE)):
            prepared = _prepare(reference, timing, MAX_ATOMS, taper_alpha=taper_alpha)
            for config in _sweep_configs(
                timing,
                available=prepared.n_atoms,
                profiles=profiles,
                n_atoms_grid=n_atoms_grid,
                min_energy_grid=min_energy_grid,
                pcm_bits=pcm_bits,
                taper_alpha=taper_alpha,
                budget_bytes=budget_bytes,
                name=name,
            ):
                bound = _lower_bound(prepared, config)
                fallback, fallback_bound = _closer(fallback, fallback_bound, prepared, config, bound)
                if bound > budget_bytes:
                    continue
                try:
                    compiled = _assemble(prepared, config)
                except Exception:  # pylint: disable=broad-except  # one bad config must not abort the sweep
                    failures += 1
                    continue
                # Fitting the budget is not enough: a pattern over IT's u16 length field cannot be
                # serialized, so an unwritable candidate is skipped just like an over-budget one.
                if compiled.cost.total > budget_bytes or not compiled.cost.writable:
                    continue
                candidates.append(_candidate(compiled))
                best, best_mel = _keep_better(best, best_mel, compiled, on_best)
    except KeyboardInterrupt:
        interrupted = True  # stop searching, but finalize on whatever we already found

    if best is None:
        # Nothing fit the budget (or Ctrl-C landed before a feasible point). Compile the closest-to-budget
        # seed anyway and hand it back flagged over budget, rather than throwing the whole run away. It goes
        # through the same record path, so it lands in the frontier and fires on_best exactly once.
        if fallback is None:
            raise ValueError("no candidate to compile; the grid or signal is empty")
        prepared, config = fallback
        compiled = _assemble(prepared, config)
        candidates.append(_candidate(compiled))
        best, best_mel = _keep_better(best, best_mel, compiled, on_best)

    assert best is not None  # the sweep or the fallback always sets it
    return Plan(module=best, frontier=_pareto(candidates), interrupted=interrupted, failures=failures)


def _sweep_configs(
    timing: Timing,
    *,
    available: int,
    profiles: Sequence[Profile],
    n_atoms_grid: Sequence[int],
    min_energy_grid: Sequence[float],
    pcm_bits: Literal[8, 16],
    taper_alpha: float,
    budget_bytes: int,
    name: str,
) -> Iterator[TokenizerConfig]:
    """Every config to try at one timing — the pool sizes (clamped to the SVD's atoms) × profiles × floors.

    Flattening the three grids into one generator keeps :func:`plan_compilation`'s loop shallow; the timing
    is fixed here because it alone drives the SVD that the caller shares across the whole yield.
    """
    for n_atoms in _atom_options(n_atoms_grid, available):
        for profile in profiles:
            for min_energy in min_energy_grid:
                yield TokenizerConfig.load(
                    profile=profile,
                    tempo=timing.tempo,
                    speed=timing.speed,
                    n_atoms=n_atoms,
                    min_energy=min_energy,
                    pcm_bits=pcm_bits,
                    taper_alpha=taper_alpha,
                    budget_bytes=budget_bytes,
                    name=name,
                )


def _atom_options(grid: Sequence[int], available: int) -> tuple[int, ...]:
    """The distinct pool sizes to try: every grid value clamped to the atoms the SVD actually produced."""
    return tuple(sorted({min(n, available) for n in grid if n >= 1}))


def _lower_bound(prepared: Prepared, config: TokenizerConfig) -> int:
    """The candidate's minimum byte size, from selection alone (no assignment, no metrics)."""
    coef = prepared.coef[:, : config.n_atoms]
    selection = select(coef, max_per_row=config.max_per_row, min_energy=config.min_energy)
    codes = quantise(np.where(selection.mask, coef, 0.0)).codes
    n_used = int(np.count_nonzero(np.any(codes > 0, axis=0)))
    n_cells = int(np.count_nonzero(codes > 0))
    return cost_lower_bound(
        prepared.n_rows,
        n_cells,
        n_stored_samples=_POLARITIES * n_used,
        pcm_frames=prepared.timing.row_frames,
        bits_per_frame=config.pcm_bits,
    )


def _keep_better(
    best: CompiledModule | None,
    best_mel: float,
    compiled: CompiledModule,
    on_best: Callable[[CompiledModule], None] | None,
) -> tuple[CompiledModule | None, float]:
    """Whichever of the running best and a new feasible module has the lower mel, checkpointing a win.

    The single place the running best changes: when ``compiled`` beats ``best_mel`` it becomes the new best
    and ``on_best`` is called with it, so each improvement is streamed to disk and a crash still leaves the
    best-so-far. Returning the carried ``(module, mel)`` instead of mutating an enclosing variable keeps the
    sweep's state explicit — the caller threads it through the loop, and there is nothing non-local to track.
    """
    mel = compiled.metrics.mel_distance_db
    if mel >= best_mel:
        return best, best_mel
    if on_best is not None:
        on_best(compiled)
    return compiled, mel


def _closer(
    fallback: tuple[Prepared, TokenizerConfig] | None,
    fallback_bound: int,
    prepared: Prepared,
    config: TokenizerConfig,
    bound: int,
) -> tuple[tuple[Prepared, TokenizerConfig], int]:
    """The seed with the smaller lower bound — the closest-to-budget point to fall back to when nothing fits.

    Carries one timing's prepared arrays (the current smallest-bound seed) forward, so the fallback needs no
    enclosing state and stays memory-flat: only a single ``Prepared`` is ever held for it.
    """
    if fallback is None or bound < fallback_bound:
        return (prepared, config), bound
    return fallback, fallback_bound


def _candidate(compiled: CompiledModule) -> Candidate:
    return Candidate(
        config=compiled.config,
        total_bytes=compiled.cost.total,
        mel_db=compiled.metrics.mel_distance_db,
        n_atoms_used=compiled.n_atoms_used,
        n_channels_used=compiled.n_channels_used,
    )


def _pareto(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """The cost-vs-mel frontier: cheapest first, keeping only points where more bytes buy lower mel."""
    frontier: list[Candidate] = []
    best_mel = float("inf")
    for candidate in sorted(candidates, key=lambda c: (c.total_bytes, c.mel_db)):
        if candidate.mel_db < best_mel - _MEL_EPS:
            frontier.append(candidate)
            best_mel = candidate.mel_db
    return tuple(frontier)


def format_frontier(frontier: Sequence[Candidate]) -> str:
    """A compact cost-vs-mel table, cheapest first, with the last row the lowest-mel choice."""
    header = f"{'profile':<7} {'tempo':>5} {'atoms':>5} {'chan':>4} {'KB':>6} {'mel dB':>7}"
    rows = [
        f"{c.config.profile:<7} {c.config.tempo:>5} {c.config.n_atoms:>5} "
        f"{c.n_channels_used:>4} {c.total_bytes / 1024:>6.0f} {c.mel_db:>7.2f}"
        for c in frontier
    ]
    return "\n".join([header, *rows])
