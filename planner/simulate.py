"""Year-by-year simulation orchestrator.

Order of operations within a year (real-dollar terms):
  1. Apply real growth to all accounts.
  2. Run strategy planner -> withdrawals + conversion decision.
  3. Apply withdrawals to portfolio.
  4. Apply conversion (Trad balance -> new Roth ladder rung).
  5. Record outcomes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

from .accounts import Cash, HSA, Portfolio, RothIRA, Taxable, TraditionalIRA
from .returns import ConstantReturns, ReturnsModel
from .strategy import PlanResult, plan_year
from .tax import TAX_PARAMS_2026, TaxParams


@dataclass
class YearResult:
    year: int  # 0-indexed year of retirement
    age: int
    starting_total: float
    ending_total: float
    plan: PlanResult
    snapshot: dict  # account snapshot at end-of-year
    target_net: float
    withdrawal_rate: float  # gross withdrawals / starting_total


@dataclass(frozen=True)
class SimulationInputs:
    initial_cash: float = 25_000
    initial_taxable: float = 834_843
    taxable_basis: float = 570_659
    initial_traditional: float = 837_547
    initial_roth: float = 327_610
    roth_contributions: float = 0.0  # of the Roth balance, how much is direct contributions
    initial_hsa: float = 26_000
    target_spend: float = 80_000  # real, net
    # Real returns by asset class (used by ConstantReturns when no model is passed).
    stock_return: float = 0.07
    bond_return: float = 0.02
    cash_return: float = 0.0
    # Stock allocation per investment account (0..1). Cash account ignores allocation.
    stock_allocation: float = 0.85
    start_age: int = 35
    horizon_years: int = 60
    strategy: str = "bridge_optimal"
    custom_conversion: Optional[float] = None
    aca_mode: str = "cap"
    params: TaxParams = field(default_factory=lambda: TAX_PARAMS_2026)


def build_portfolio(inputs: SimulationInputs) -> Portfolio:
    return Portfolio(
        cash=Cash(balance=inputs.initial_cash),
        taxable=Taxable(balance=inputs.initial_taxable, basis=inputs.taxable_basis),
        traditional=TraditionalIRA(balance=inputs.initial_traditional),
        roth=RothIRA(
            contributions=inputs.roth_contributions,
            earnings=inputs.initial_roth - inputs.roth_contributions,
        ),
        hsa=HSA(balance=inputs.initial_hsa),
        stock_allocation_taxable=inputs.stock_allocation,
        stock_allocation_traditional=inputs.stock_allocation,
        stock_allocation_roth=inputs.stock_allocation,
        stock_allocation_hsa=inputs.stock_allocation,
    )


def _apply_action(portfolio: Portfolio, plan: PlanResult, year: int) -> None:
    w = plan.withdrawals
    portfolio.cash.withdraw(w.cash)
    portfolio.taxable.sell(w.taxable)
    portfolio.roth.withdraw_seasoned(w.roth_seasoned, current_year=year)
    portfolio.roth.withdraw_contributions(w.roth_contributions)
    portfolio.roth.withdraw_any(w.roth_post60)
    portfolio.traditional.withdraw(w.traditional)
    portfolio.hsa.withdraw(w.hsa)

    # Apply conversion: pull from Trad, add to Roth ladder rung
    if plan.conversion > 0:
        actually_converted = portfolio.traditional.withdraw(plan.conversion)
        portfolio.roth.add_conversion(year=year, amount=actually_converted)


def simulate(
    inputs: SimulationInputs,
    returns_model: Optional[ReturnsModel] = None,
    path_index: int = 0,
) -> List[YearResult]:
    """Run a single deterministic simulation path.

    `returns_model` defaults to ConstantReturns built from `inputs.stock_return / bond_return / cash_return`.
    For Monte Carlo, pass a stochastic model and vary `path_index` per run.
    """
    if returns_model is None:
        returns_model = ConstantReturns(
            stocks=inputs.stock_return,
            bonds=inputs.bond_return,
            cash=inputs.cash_return,
        )

    portfolio = build_portfolio(inputs)
    results: List[YearResult] = []

    for y in range(inputs.horizon_years):
        age = inputs.start_age + y

        # 1. Capture beginning-of-year balance (pre-growth) for SWR-conventional WR denominator.
        starting_total = portfolio.total
        year_returns = returns_model.get(year_index=y, path_index=path_index)
        portfolio.apply_growth(year_returns)

        # 2. Plan the year.
        plan = plan_year(
            portfolio=portfolio,
            age=age,
            year=y,
            target_net=inputs.target_spend,
            strategy_name=inputs.strategy,
            params=inputs.params,
            aca_mode=inputs.aca_mode,
            custom_conversion=inputs.custom_conversion,
        )

        # 3+4. Apply.
        _apply_action(portfolio, plan, year=y)

        # 5. Record.
        ending_total = portfolio.total
        gross_withdrawn = (
            plan.withdrawals.cash
            + plan.withdrawals.taxable
            + plan.withdrawals.roth_seasoned
            + plan.withdrawals.roth_contributions
            + plan.withdrawals.roth_post60
            + plan.withdrawals.traditional
            + plan.withdrawals.hsa
        )
        wr = gross_withdrawn / starting_total if starting_total > 0 else 0.0

        results.append(
            YearResult(
                year=y,
                age=age,
                starting_total=starting_total,
                ending_total=ending_total,
                plan=plan,
                snapshot=portfolio.snapshot(),
                target_net=inputs.target_spend,
                withdrawal_rate=wr,
            )
        )

        # Stop if portfolio is depleted.
        if ending_total < 1.0 and plan.shortfall > 0:
            break

    return results


def summarize(results: List[YearResult]) -> dict:
    """Cross-cutting metrics for a single scenario."""
    if not results:
        return {}
    total_tax = sum(r.plan.federal_tax for r in results)
    total_aca = sum(r.plan.aca_oop for r in results)
    total_penalty = sum(r.plan.penalty for r in results)
    total_conversions = sum(r.plan.conversion for r in results)
    total_shortfall = sum(r.plan.shortfall for r in results)
    last = results[-1]
    return {
        "ending_total": last.ending_total,
        "ending_age": last.age,
        "years_funded": len(results),
        "depleted": last.ending_total < 1.0,
        "total_federal_tax": total_tax,
        "total_aca_oop": total_aca,
        "total_penalty": total_penalty,
        "total_conversions": total_conversions,
        "total_shortfall": total_shortfall,
    }
