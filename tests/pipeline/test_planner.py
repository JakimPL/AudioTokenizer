from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod.limits.compliance import Compliance

from audiotokenizer.pipeline.compiler import compile_signal
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.pipeline.planner import PLANNED_FORMAT, plan_compilation

_BIG_BUDGET = 8 * 1024 * 1024  # generous enough that nothing is pruned, so the search is unconstrained
_GRID = (8, 16, 24)
_CANONICAL = (Compliance.CANONICAL,)
_BOTH = (Compliance.EXTENDED, Compliance.CANONICAL)


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A few seconds of a chord plus a little noise — enough structure for a small dictionary to fit."""
    rng = np.random.default_rng(0)
    time = np.arange(3 * 44100) / 44100.0
    tone = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.0, 330.0))
    return np.asarray(0.3 * tone + 0.02 * rng.standard_normal(time.size), dtype=np.float64)


def test_plan_fits_the_budget_and_is_a_real_module(signal: NDArray[np.float64]) -> None:
    budget = 400 * 1024
    plan = plan_compilation(signal, budget_bytes=budget, compliances=_CANONICAL, n_atoms_grid=_GRID)
    assert plan.module.within_budget
    assert plan.module.size.total <= budget
    # The chosen config is compiled into a genuine module: its size model equals the bytes it writes.
    assert plan.module.size.total == len(plan.module.to_bytes())
    assert plan.config.format is PLANNED_FORMAT


def test_plan_is_no_worse_than_a_hand_config_in_its_search_space(signal: NDArray[np.float64]) -> None:
    hand = compile_signal(
        signal, TokenizerConfig(compliance=Compliance.CANONICAL, tempo=125, n_atoms=8, budget_bytes=_BIG_BUDGET)
    )
    plan = plan_compilation(signal, budget_bytes=_BIG_BUDGET, compliances=_CANONICAL, n_atoms_grid=_GRID)
    # The hand config (canonical, tempo 125, 8 atoms) is one point the sweep visits, so the winner cannot lose.
    assert plan.module.metrics.mel_distance_db <= hand.metrics.mel_distance_db + 1e-9


def test_frontier_is_a_monotone_pareto_ending_in_the_choice(signal: NDArray[np.float64]) -> None:
    plan = plan_compilation(signal, budget_bytes=_BIG_BUDGET, compliances=_BOTH, n_atoms_grid=_GRID)
    frontier = plan.frontier
    assert frontier
    costs = [c.total_bytes for c in frontier]
    mels = [c.mel_db for c in frontier]
    assert costs == sorted(costs)  # cheapest first
    assert all(later < earlier for earlier, later in zip(mels, mels[1:]))  # each step buys lower mel
    # The lowest-mel frontier point is exactly the module the planner returns.
    assert frontier[-1].config == plan.module.config
    assert frontier[-1].mel_db == pytest.approx(plan.module.metrics.mel_distance_db)


def test_nothing_fits_returns_the_closest_module_instead_of_discarding_the_run(signal: NDArray[np.float64]) -> None:
    # A budget too small for any candidate must not throw the sweep away: the planner hands back the
    # closest-to-budget module, compiled for real and flagged over budget, so there is always something to
    # render.
    plan = plan_compilation(signal, budget_bytes=1024, compliances=_CANONICAL, n_atoms_grid=_GRID)
    assert not plan.within_budget
    assert plan.module.size.total > 1024
    assert plan.module.size.total == len(plan.module.to_bytes())  # a genuine module, not a stub


def test_on_best_checkpoints_the_running_best_and_ends_on_the_choice(signal: NDArray[np.float64]) -> None:
    # Every improvement is streamed to the callback, so a crash mid-sweep still leaves the best-so-far; the
    # final checkpoint is exactly the module the planner returns.
    seen: list[float] = []
    plan = plan_compilation(
        signal,
        budget_bytes=_BIG_BUDGET,
        compliances=_CANONICAL,
        n_atoms_grid=_GRID,
        on_best=lambda module: seen.append(module.metrics.mel_distance_db),
    )
    assert seen  # at least the first feasible candidate fired it
    assert seen == sorted(seen, reverse=True)  # each checkpoint is a strict improvement in mel
    assert seen[-1] == pytest.approx(plan.module.metrics.mel_distance_db)


def test_plan_raises_only_when_there_is_nothing_to_compile(signal: NDArray[np.float64]) -> None:
    # An empty grid leaves no candidate at all — the one case with nothing to fall back to.
    with pytest.raises(ValueError):
        plan_compilation(signal, budget_bytes=_BIG_BUDGET, compliances=_CANONICAL, n_atoms_grid=())


def test_planner_only_ever_returns_a_writable_module(signal: NDArray[np.float64]) -> None:
    # Fitting the byte budget is not enough — a pattern over the u16 length field cannot be serialized.
    # The planner must reject those, so the module it hands back always writes without raising.
    plan = plan_compilation(signal, budget_bytes=_BIG_BUDGET, compliances=_BOTH, n_atoms_grid=_GRID)
    assert plan.module.writable
    assert plan.module.violations == ()
    assert plan.module.size.total == len(plan.module.to_bytes())
