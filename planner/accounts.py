"""Account state, growth, and Roth conversion ladder bookkeeping.

All values are real dollars. Growth is real return. Per-account asset allocation
(stock_allocation in [0,1]) determines blended growth from a YearReturns sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .returns import YearReturns, blended_return


@dataclass
class Cash:
    balance: float = 0.0

    def apply_growth(self, rate: float) -> None:
        self.balance *= 1 + rate

    def withdraw(self, amount: float) -> float:
        take = min(self.balance, max(amount, 0.0))
        self.balance -= take
        return take


@dataclass
class Taxable:
    balance: float = 0.0
    basis: float = 0.0

    def apply_growth(self, rate: float) -> None:
        self.balance *= 1 + rate

    @property
    def gain_ratio(self) -> float:
        if self.balance <= 0:
            return 0.0
        gain = max(self.balance - self.basis, 0.0)
        return gain / self.balance

    def sell(self, amount: float) -> Tuple[float, float]:
        """Sell `amount` of market value pro-rata between basis and gain.

        Returns (cash_proceeds, ltcg_realized).
        """
        amount = max(amount, 0.0)
        if amount <= 0 or self.balance <= 0:
            return 0.0, 0.0
        amount = min(amount, self.balance)
        ratio = self.gain_ratio
        ltcg = amount * ratio
        basis_used = amount - ltcg
        self.balance -= amount
        self.basis = max(self.basis - basis_used, 0.0)
        return amount, ltcg


@dataclass
class TraditionalIRA:
    balance: float = 0.0

    def apply_growth(self, rate: float) -> None:
        self.balance *= 1 + rate

    def withdraw(self, amount: float) -> float:
        take = min(self.balance, max(amount, 0.0))
        self.balance -= take
        return take


@dataclass
class RothLadderRung:
    year_converted: int
    amount: float  # remaining (not yet withdrawn)


@dataclass
class RothIRA:
    """Tracks contribution principal and seasoned/unseasoned conversion rungs separately.

    Earnings sit on top of contributions/conversions; withdrawing earnings before 59.5
    triggers tax + penalty, so we track total balance vs. contributions/conversions.
    """

    contributions: float = 0.0  # principal from direct contributions (always available)
    rungs: List[RothLadderRung] = field(default_factory=list)
    earnings: float = 0.0  # everything above contributions + sum(rungs)
    _rung_total: float = 0.0  # invariant: equals sum(r.amount for r in rungs)

    def apply_growth(self, rate: float) -> None:
        # Apply growth proportionally; earnings absorb all growth.
        if self.balance <= 0:
            return
        growth = self.balance * rate
        self.earnings += growth

    @property
    def balance(self) -> float:
        return self.contributions + self._rung_total + self.earnings

    @property
    def total_rungs(self) -> float:
        return self._rung_total

    def add_conversion(self, year: int, amount: float) -> None:
        if amount <= 0:
            return
        # Insertion order is monotonically ascending by year (simulation loop),
        # so the rungs list stays FIFO-sorted without explicit sort calls.
        self.rungs.append(RothLadderRung(year_converted=year, amount=amount))
        self._rung_total += amount

    def seasoned_balance(self, current_year: int, seasoning_years: int = 5) -> float:
        return sum(
            r.amount for r in self.rungs if current_year - r.year_converted >= seasoning_years
        )

    def withdraw_seasoned(
        self, amount: float, current_year: int, seasoning_years: int = 5
    ) -> float:
        """Withdraw FIFO from rungs that have seasoned >= seasoning_years."""
        amount = max(amount, 0.0)
        taken = 0.0
        for r in self.rungs:
            if amount <= 0:
                break
            if current_year - r.year_converted < seasoning_years:
                continue
            chunk = min(r.amount, amount)
            r.amount -= chunk
            amount -= chunk
            taken += chunk
        if taken > 0:
            self._rung_total -= taken
            self.rungs = [r for r in self.rungs if r.amount > 1e-6]
        return taken

    def withdraw_contributions(self, amount: float) -> float:
        take = min(self.contributions, max(amount, 0.0))
        self.contributions -= take
        return take

    def withdraw_any(self, amount: float) -> float:
        """Post-59.5: any Roth dollar is tax/penalty-free.

        Pull order (cosmetic): contributions -> all rungs (seasoned first) -> earnings.
        """
        amount = max(amount, 0.0)
        taken = 0.0
        if amount > 0 and self.contributions > 0:
            chunk = min(self.contributions, amount)
            self.contributions -= chunk
            amount -= chunk
            taken += chunk
        if amount > 0 and self.rungs:
            rung_taken = 0.0
            for r in self.rungs:
                if amount <= 0:
                    break
                chunk = min(r.amount, amount)
                r.amount -= chunk
                amount -= chunk
                taken += chunk
                rung_taken += chunk
            if rung_taken > 0:
                self._rung_total -= rung_taken
                self.rungs = [r for r in self.rungs if r.amount > 1e-6]
        if amount > 0 and self.earnings > 0:
            chunk = min(self.earnings, amount)
            self.earnings -= chunk
            amount -= chunk
            taken += chunk
        return taken


@dataclass
class HSA:
    balance: float = 0.0

    def apply_growth(self, rate: float) -> None:
        self.balance *= 1 + rate

    def withdraw(self, amount: float) -> float:
        take = min(self.balance, max(amount, 0.0))
        self.balance -= take
        return take


@dataclass
class Portfolio:
    cash: Cash
    taxable: Taxable
    traditional: TraditionalIRA
    roth: RothIRA
    hsa: HSA
    # Stock allocation per investment account (0=all bonds, 1=all stocks).
    # Cash always uses YearReturns.cash directly.
    stock_allocation_taxable: float = 0.85
    stock_allocation_traditional: float = 0.85
    stock_allocation_roth: float = 0.85
    stock_allocation_hsa: float = 0.85

    def apply_growth(self, year_returns: YearReturns) -> None:
        self.cash.apply_growth(year_returns.cash)
        self.taxable.apply_growth(blended_return(year_returns, self.stock_allocation_taxable))
        self.traditional.apply_growth(blended_return(year_returns, self.stock_allocation_traditional))
        self.roth.apply_growth(blended_return(year_returns, self.stock_allocation_roth))
        self.hsa.apply_growth(blended_return(year_returns, self.stock_allocation_hsa))

    @property
    def total(self) -> float:
        return (
            self.cash.balance
            + self.taxable.balance
            + self.traditional.balance
            + self.roth.balance
            + self.hsa.balance
        )

    def snapshot(self) -> dict:
        return {
            "cash": self.cash.balance,
            "taxable": self.taxable.balance,
            "taxable_basis": self.taxable.basis,
            "traditional": self.traditional.balance,
            "roth": self.roth.balance,
            "roth_seasoned": self.roth.total_rungs,  # full ladder amount, agnostic of "now"
            "roth_contributions": self.roth.contributions,
            "roth_earnings": self.roth.earnings,
            "hsa": self.hsa.balance,
            "total": self.total,
        }
