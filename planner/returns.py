"""Returns model abstraction: pluggable per-asset-class real-return generator.

A `ReturnsModel` produces a year's real returns for (stocks, bonds, cash).
Implementations:
  - `ConstantReturns`: deterministic flat rates (default for single-path simulation)
  - `LognormalReturns`: stochastic, Wave 2c Monte Carlo
  - (future) `HistoricalPlayback`: Shiller-data resampling

Per-account growth is computed by combining the year's asset returns with each
account's stock allocation. Cash uses its own rate.
"""

from __future__ import annotations

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
