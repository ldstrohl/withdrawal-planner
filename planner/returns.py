"""Returns model abstraction: pluggable per-asset-class real-return generator.

A `ReturnsModel` produces a year's real returns for (stocks, bonds, cash).
Implementations:
  - `ConstantReturns`: deterministic flat rates (default for single-path simulation)
  - `LognormalReturns`: stochastic, used by Monte Carlo
  - `HistoricalPlayback`: rolling-start replay of Shiller annual real returns

Per-account growth is computed by combining the year's asset returns with each
account's stock allocation. Cash uses its own rate.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Tuple


@dataclass(frozen=True)
class YearReturns:
    """Real returns for one simulation year, per asset class."""
    stocks: float
    bonds: float
    cash: float


class ReturnsModel(Protocol):
    """A returns model produces a YearReturns for a given (year_index, path_index).

    `year_index` is the offset from retirement start.
    `path_index` is the simulation path id (0 for single-path; 0..N-1 for Monte Carlo).
    Implementations should be deterministic given (year_index, path_index, seed).
    """

    def get(self, year_index: int, path_index: int = 0) -> YearReturns: ...


@dataclass(frozen=True)
class ConstantReturns:
    """Deterministic flat real returns. Default model."""
    stocks: float = 0.07
    bonds: float = 0.02
    cash: float = 0.0

    def get(self, year_index: int, path_index: int = 0) -> YearReturns:
        return YearReturns(stocks=self.stocks, bonds=self.bonds, cash=self.cash)


def blended_return(
    year_returns: YearReturns,
    stock_allocation: float,
) -> float:
    """Convex combination of stock/bond real returns based on allocation."""
    a = max(0.0, min(1.0, stock_allocation))
    return a * year_returns.stocks + (1 - a) * year_returns.bonds


@dataclass(frozen=True)
class LognormalReturns:
    """Stochastic real returns sampled per (year, path) from independent lognormal distributions.

    `mu_*` and `sigma_*` parameterize the log of (1 + real_return). Defaults derived from
    long-run US 1928–2024 statistics (real, after-CPI):
      - Stocks: mu=0.06, sigma=0.18
      - Bonds:  mu=0.02, sigma=0.07
      - Cash:   mu=0.00, sigma=0.01
    Set sigma=0 for deterministic behavior.

    mu is the **log-space** mean (location parameter). The arithmetic mean of (1 + return)
    is exp(mu + sigma^2/2), so arithmetic real return ≈ exp(mu + sigma^2/2) − 1.
    With mu_stocks=0.06 and sigma_stocks=0.18, arithmetic real ≈ exp(0.0762) − 1 ≈ 7.92%,
    **not** 6%.

    `stock_bond_correlation` (typical US: -0.05 to +0.20) injected via 2x2 Cholesky.

    Reproducibility: `random.Random((seed, path_index, year_index))` makes draws
    deterministic and order-independent — get(y, p) returns the same value regardless
    of call order.
    """
    mu_stocks: float = 0.06
    sigma_stocks: float = 0.18
    mu_bonds: float = 0.02
    sigma_bonds: float = 0.07
    mu_cash: float = 0.00
    sigma_cash: float = 0.01
    seed: int = 42
    stock_bond_correlation: float = 0.05

    def get(self, year_index: int, path_index: int = 0) -> YearReturns:
        # Mix seed/path/year into a single int seed (Python 3.12+ rejects tuple seeds).
        rng_seed = (self.seed * 1_000_003 + path_index * 9973 + year_index) & 0xFFFFFFFF
        rng = random.Random(rng_seed)
        z_s = rng.gauss(0.0, 1.0)
        z_b_indep = rng.gauss(0.0, 1.0)
        z_c = rng.gauss(0.0, 1.0)
        rho = self.stock_bond_correlation
        z_b = rho * z_s + math.sqrt(max(1 - rho * rho, 0.0)) * z_b_indep
        return YearReturns(
            stocks=math.exp(self.mu_stocks + self.sigma_stocks * z_s) - 1.0,
            bonds=math.exp(self.mu_bonds + self.sigma_bonds * z_b) - 1.0,
            cash=math.exp(self.mu_cash + self.sigma_cash * z_c) - 1.0,
        )


# --- Historical playback -----------------------------------------------------

DEFAULT_HISTORICAL_CSV = Path(__file__).parent.parent / "data" / "historical_real_returns.csv"


def _load_historical_csv(path: Path) -> List[Tuple[int, float, float, float]]:
    """Return [(year, real_stocks, real_bonds, real_cash), ...] sorted by year."""
    rows: List[Tuple[int, float, float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((
                int(r["year"]),
                float(r["real_stocks"]),
                float(r["real_bonds"]),
                float(r["real_cash"]),
            ))
    rows.sort(key=lambda x: x[0])
    return rows


@dataclass
class HistoricalPlayback:
    """Rolling-start replay of historical annual real returns.

    Each `path_index` corresponds to a distinct start year; `get(year_index, path_index)`
    returns the historical year `start_years[path_index] + year_index`.

    Start years are filtered so each path has full `horizon_years` of data — start years
    that would run off the end are dropped. With Shiller data 1928–2022 and a 60-year
    horizon, valid starts are 1928..1963 → 36 paths. With a 30-year horizon, 1928..1993
    → 66 paths. The Monte Carlo orchestrator should clamp `n_runs <= n_paths`.

    Not stochastic (despite living in the MC tab) — the same start-year ordering replays
    deterministically. The "MC" framing is a UX simplification.
    """
    horizon_years: int
    start_year_floor: int = 1928
    csv_path: Path = field(default_factory=lambda: DEFAULT_HISTORICAL_CSV)
    _rows: List[Tuple[int, float, float, float]] = field(init=False, repr=False)
    _by_year: dict[int, Tuple[float, float, float]] = field(init=False, repr=False)
    start_years: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self._rows = _load_historical_csv(self.csv_path)
        self._by_year = {y: (s, b, c) for y, s, b, c in self._rows}
        first_year = max(self.start_year_floor, self._rows[0][0])
        last_year = self._rows[-1][0]
        # Need years [Y, Y+1, ..., Y + horizon - 1] inclusive.
        last_valid_start = last_year - self.horizon_years + 1
        self.start_years = [y for y in range(first_year, last_valid_start + 1)
                            if y in self._by_year]

    @property
    def n_paths(self) -> int:
        return len(self.start_years)

    @property
    def coverage(self) -> Tuple[int, int]:
        """First and last calendar year covered by the underlying dataset."""
        return self._rows[0][0], self._rows[-1][0]

    def get(self, year_index: int, path_index: int = 0) -> YearReturns:
        if not self.start_years:
            raise ValueError(
                f"No valid start years (horizon={self.horizon_years} exceeds dataset)"
            )
        # Modulo so callers passing path_index >= n_paths don't crash; they get a recycled path.
        start = self.start_years[path_index % len(self.start_years)]
        s, b, c = self._by_year[start + year_index]
        return YearReturns(stocks=s, bonds=b, cash=c)
