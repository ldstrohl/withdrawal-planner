"""Monte Carlo orchestration. Runs N independent paths and aggregates outcomes.

The returns model is pluggable: any object satisfying the `ReturnsModel` protocol
(see planner.returns) can drive the simulation. Today: LognormalReturns. Future:
HistoricalPlayback (Shiller resampling), regime-switching, fat-tailed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .returns import ReturnsModel
from .simulate import SimulationInputs, YearResult, simulate, summarize


@dataclass
class PathSummary:
    path_index: int
    years_funded: int
    ending_total: float
    depleted: bool
    total_shortfall: float


@dataclass
class MCSummary:
    inputs: SimulationInputs
    n_runs: int
    success_rate: float          # fraction of paths with zero shortfall over full horizon
    median_ending: float
    p5_ending: float
    p95_ending: float
    median_depletion_age: Optional[float]
    paths: List[PathSummary] = field(default_factory=list)
    age: List[int] = field(default_factory=list)
    p5_balance: List[float] = field(default_factory=list)
    p25_balance: List[float] = field(default_factory=list)
    p50_balance: List[float] = field(default_factory=list)
    p75_balance: List[float] = field(default_factory=list)
    p95_balance: List[float] = field(default_factory=list)


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def run_monte_carlo(
    inputs: SimulationInputs,
    returns_model: ReturnsModel,
    n_runs: int = 1000,
) -> MCSummary:
    """Run n_runs independent paths and collect summary stats per year and overall."""
    per_year_balances: List[List[float]] = [[] for _ in range(inputs.horizon_years)]
    paths: List[PathSummary] = []

    for i in range(n_runs):
        results: List[YearResult] = simulate(inputs, returns_model=returns_model, path_index=i)
        s = summarize(results)
        paths.append(
            PathSummary(
                path_index=i,
                years_funded=s["years_funded"],
                ending_total=s["ending_total"],
                depleted=s["depleted"],
                total_shortfall=s["total_shortfall"],
            )
        )
        for y, r in enumerate(results):
            per_year_balances[y].append(r.ending_total)
        # Pad early-terminated paths with 0 for the rest of the horizon.
        for y in range(len(results), inputs.horizon_years):
            per_year_balances[y].append(0.0)

    success = sum(1 for p in paths if p.total_shortfall <= 0.01) / n_runs
    endings = sorted(p.ending_total for p in paths)
    median_ending = _percentile(endings, 0.5)
    p5_ending = _percentile(endings, 0.05)
    p95_ending = _percentile(endings, 0.95)

    depletion_ages = [inputs.start_age + p.years_funded for p in paths if p.depleted]
    if depletion_ages:
        depletion_ages.sort()
        median_depletion_age = _percentile([float(a) for a in depletion_ages], 0.5)
    else:
        median_depletion_age = None

    age = [inputs.start_age + y for y in range(inputs.horizon_years)]
    p5, p25, p50, p75, p95 = [], [], [], [], []
    for y in range(inputs.horizon_years):
        sb = sorted(per_year_balances[y])
        p5.append(_percentile(sb, 0.05))
        p25.append(_percentile(sb, 0.25))
        p50.append(_percentile(sb, 0.50))
        p75.append(_percentile(sb, 0.75))
        p95.append(_percentile(sb, 0.95))

    return MCSummary(
        inputs=inputs,
        n_runs=n_runs,
        success_rate=success,
        median_ending=median_ending,
        p5_ending=p5_ending,
        p95_ending=p95_ending,
        median_depletion_age=median_depletion_age,
        paths=paths,
        age=age,
        p5_balance=p5,
        p25_balance=p25,
        p50_balance=p50,
        p75_balance=p75,
        p95_balance=p95,
    )
