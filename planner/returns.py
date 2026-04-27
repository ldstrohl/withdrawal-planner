"""Returns model abstraction: pluggable per-asset-class real-return generator.

A `ReturnsModel` produces a year's real returns for (stocks, bonds, cash).
Implementations:
  - `ConstantReturns`: deterministic flat rates (default for single-path simulation)
  - `LognormalReturns`: stochastic, used by Monte Carlo
  - (future) `HistoricalPlayback`: Shiller-data resampling

Per-account growth is computed by combining the year's asset returns with each
account's stock allocation. Cash uses its own rate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol, Tuple


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
