"""Withdrawal & Roth-conversion strategies.

A strategy is a function `(portfolio, age, year, target_net, params, aca_mode) -> PlanResult`
that decides how much to withdraw from each account and how much to convert,
solving the tax fixed-point internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .accounts import Portfolio
from .tax import (
    TAX_PARAMS_2026,
    TaxParams,
    aca_premium,
    medicare_premium,
    early_withdrawal_penalty,
    federal_tax,
    fpl_400_ceiling,
    zero_ltcg_ceiling,
)


@dataclass
class Withdrawals:
    cash: float = 0.0
    taxable: float = 0.0  # market value sold
    roth_seasoned: float = 0.0
    traditional: float = 0.0
    roth_contributions: float = 0.0
    roth_post60: float = 0.0  # post-59.5 free withdrawals (any Roth)
    hsa: float = 0.0


@dataclass
class PlanResult:
    withdrawals: Withdrawals
    ltcg: float
    conversion: float
    ordinary_income: float
    federal_tax: float
    penalty: float
    healthcare_oop: float
    magi: float
    target_net: float
    funded: float  # what we actually delivered (may be < target_net if portfolio insolvent)
    shortfall: float


# --- conversion sizing ------------------------------------------------------


def _conversion_for_strategy(
    name: str,
    portfolio: Portfolio,
    ltcg_estimate: float,
    params: TaxParams,
    custom_amount: Optional[float],
) -> float:
    if portfolio.traditional.balance <= 0:
        return 0.0
    if name == "minimal_convert":
        target = params.standard_deduction
    elif name == "aggressive_convert":
        # Fill top of 12% bracket. ordinary_brackets[1][0] is the upper limit of the 12%
        # bracket as *taxable* income (~$48,475). Adding standard_deduction yields the
        # target gross ordinary income that places taxable income at that ceiling.
        top_12_taxable = params.ordinary_brackets[1][0]
        target = params.standard_deduction + top_12_taxable
    elif name == "bridge_optimal":
        ceil_ltcg = zero_ltcg_ceiling(ltcg_estimate, params)
        ceil_aca = fpl_400_ceiling(params) - ltcg_estimate
        # Floor at standard deduction: filling std_ded is always free regardless of LTCG,
        # and keeps the conversion ladder fed during late-Phase-A high-gain-ratio years.
        target = max(min(ceil_ltcg, ceil_aca), params.standard_deduction)
    elif name == "custom":
        target = custom_amount or 0.0
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return min(target, portfolio.traditional.balance)


# --- withdrawal allocator ---------------------------------------------------


def _fund_priority(
    portfolio: Portfolio,
    age: int,
    year: int,
    gross_need: float,
) -> tuple[Withdrawals, float, float]:
    """Walk source priority to fund `gross_need`. Returns (withdrawals, ltcg, shortfall).

    Priority (pre-59.5):
      cash -> taxable -> seasoned Roth -> Roth contributions -> Trad (10% penalty) -> HSA
    Priority (post-59.5):
      cash -> taxable -> Roth (any) -> Trad -> HSA
    HSA is last in both phases (non-medical: tax + 20% penalty pre-65, tax-only post-65).
    """
    w = Withdrawals()
    remaining = max(gross_need, 0.0)
    ltcg = 0.0
    post_60 = age >= 60

    # 1. Cash
    take = min(portfolio.cash.balance, remaining)
    w.cash = take
    remaining -= take

    # 2. Taxable
    if remaining > 0 and portfolio.taxable.balance > 0:
        sale = min(portfolio.taxable.balance, remaining)
        w.taxable = sale
        ltcg = sale * portfolio.taxable.gain_ratio
        remaining -= sale

    if post_60:
        # 3. Roth (full balance, free)
        if remaining > 0 and portfolio.roth.balance > 0:
            take = min(portfolio.roth.balance, remaining)
            w.roth_post60 = take
            remaining -= take
        # 4. Traditional (ordinary tax, no penalty)
        if remaining > 0 and portfolio.traditional.balance > 0:
            take = min(portfolio.traditional.balance, remaining)
            w.traditional = take
            remaining -= take
    else:
        # 3. Seasoned Roth conversions (free)
        if remaining > 0:
            seasoned = portfolio.roth.seasoned_balance(year)
            take = min(seasoned, remaining)
            w.roth_seasoned = take
            remaining -= take
        # 4. Roth contributions (free)
        if remaining > 0 and portfolio.roth.contributions > 0:
            take = min(portfolio.roth.contributions, remaining)
            w.roth_contributions = take
            remaining -= take
        # 5. Traditional (10% penalty)
        if remaining > 0 and portfolio.traditional.balance > 0:
            take = min(portfolio.traditional.balance, remaining)
            w.traditional = take
            remaining -= take

    # Last resort: HSA
    if remaining > 0 and portfolio.hsa.balance > 0:
        take = min(portfolio.hsa.balance, remaining)
        w.hsa = take
        remaining -= take

    return w, ltcg, remaining


# --- main planner -----------------------------------------------------------


def plan_year(
    portfolio: Portfolio,
    age: int,
    year: int,
    target_net: float,
    strategy_name: str,
    params: TaxParams = TAX_PARAMS_2026,
    aca_mode: str = "cap",
    custom_conversion: Optional[float] = None,
    max_iter: int = 40,
    tol: float = 1.0,
) -> PlanResult:
    """Solve the year's plan: how much to withdraw from each source and convert.

    Iterates to a fixed point because taxes depend on withdrawals which depend on
    target gross need which depends on taxes.
    """
    gross_need = target_net  # initial guess
    last = None

    for _ in range(max_iter):
        # Trial conversion sized using current LTCG estimate.
        ltcg_est = (
            min(gross_need, portfolio.taxable.balance) * portfolio.taxable.gain_ratio
            if portfolio.taxable.balance > 0
            else 0.0
        )
        conversion = _conversion_for_strategy(
            strategy_name, portfolio, ltcg_est, params, custom_conversion
        )

        w, ltcg, shortfall = _fund_priority(portfolio, age, year, gross_need)

        # Ordinary income = conversion + traditional withdrawals + non-qualified HSA distribution
        # (HSA non-medical pre-65 also incurs 20% penalty; we ignore HSA penalty for v1
        # since HSA is last-resort and the simulator should never reach it under reasonable inputs.)
        ordinary_income = conversion + w.traditional + w.hsa

        tax = federal_tax(ordinary_income, ltcg, params)
        penalty = early_withdrawal_penalty(w.traditional, age)
        magi = ordinary_income + ltcg
        if age >= 65:
            healthcare = medicare_premium(magi, age)
        else:
            healthcare = aca_premium(magi, params, aca_mode)

        new_gross = target_net + tax["total"] + penalty + healthcare["out_of_pocket"]

        last = PlanResult(
            withdrawals=w,
            ltcg=ltcg,
            conversion=conversion,
            ordinary_income=ordinary_income,
            federal_tax=tax["total"],
            penalty=penalty,
            healthcare_oop=healthcare["out_of_pocket"],
            magi=magi,
            target_net=target_net,
            funded=target_net - shortfall,
            shortfall=shortfall,
        )

        if abs(new_gross - gross_need) < tol:
            break
        # Damped update: prevents oscillation near the ACA cliff discontinuity at 400% FPL.
        gross_need = 0.5 * gross_need + 0.5 * new_gross

    return last


# --- preset registry --------------------------------------------------------

STRATEGY_PRESETS = ["bridge_optimal", "minimal_convert", "aggressive_convert", "custom"]

STRATEGY_DESCRIPTIONS = {
    "bridge_optimal": (
        "Spend taxable + cash; convert Trad→Roth up to the lesser of (0% LTCG ceiling) and "
        "(400% FPL ceiling) each year. The recommended default."
    ),
    "minimal_convert": (
        "Convert only enough Trad→Roth to fill the standard deduction (~$15.7k)."
    ),
    "aggressive_convert": (
        "Convert Trad→Roth all the way to the top of the 12% bracket every year, "
        "even when LTCG gets pushed out of 0%."
    ),
    "custom": "Specify a fixed annual conversion amount.",
}
