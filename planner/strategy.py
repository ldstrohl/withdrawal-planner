"""Withdrawal & Roth-conversion strategies.

A strategy is a function `(portfolio, age, year, target_net, params, aca_mode) -> PlanResult`
that decides how much to withdraw from each account and how much to convert,
solving the tax fixed-point internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .accounts import Portfolio
from .state_tax import STATE_PRESETS, StateTaxParams, state_tax as _state_tax_fn
from .tax import (
    TAX_PARAMS_2026,
    TaxParams,
    aca_premium,
    medicare_premium,
    early_withdrawal_penalty,
    federal_tax,
    fpl_400_ceiling,
    taxable_ss,
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
    ss_income: float = 0.0
    rmd_amount: float = 0.0
    state_tax: float = 0.0
    gross_used: float = 0.0  # actual gross need for cash-flow (pre-RMD-forced bump)
    scheduled_income: float = 0.0
    scheduled_taxable_income: float = 0.0
    scheduled_expense: float = 0.0


# --- conversion sizing ------------------------------------------------------


def _conversion_for_strategy(
    name: str,
    portfolio: Portfolio,
    age: int,
    ltcg_estimate: float,
    params: TaxParams,
    custom_amount: Optional[float],
    drawdown_from_peak: float = 0.0,
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
    elif name == "bridge_guarded":
        # Like bridge_optimal but throttles conversion in down sequences. Caps the year's
        # conversion at trad_balance / years_to_60, so the Trad backstop can't be drained
        # faster than the bridge years remaining. Protects against the failure mode where
        # bracket-sized conversions empty Trad before age 60 and leave nothing to draw on
        # if the Roth ladder gap opens up. No std-ded floor — preserving Trad outranks
        # capturing free conversion room.
        ceil_ltcg = zero_ltcg_ceiling(ltcg_estimate, params)
        ceil_aca = fpl_400_ceiling(params) - ltcg_estimate
        bracket_target = max(min(ceil_ltcg, ceil_aca), params.standard_deduction)
        if age < 60:
            years_left = max(60 - age, 1)
            reserve_cap = portfolio.traditional.balance / years_left
            target = min(bracket_target, reserve_cap)
        else:
            target = bracket_target
    elif name == "bridge_responsive":
        # Linear blend between minimal_convert and bridge_optimal targets, weighted
        # by drawdown from peak. No drawdown -> minimal (preserve Trad cushion +
        # avoid tax-prepay on dollars that may crash). At >= 20% drawdown -> full
        # bridge_optimal target.
        ceil_ltcg = zero_ltcg_ceiling(ltcg_estimate, params)
        ceil_aca = fpl_400_ceiling(params) - ltcg_estimate
        bracket_target = max(min(ceil_ltcg, ceil_aca), params.standard_deduction)
        minimal_target = params.standard_deduction
        blend = min(drawdown_from_peak / 0.20, 1.0)
        target = minimal_target + blend * (bracket_target - minimal_target)
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
    ss_income: float = 0.0,
    rmd_amount: float = 0.0,
    scheduled_income: float = 0.0,
    scheduled_taxable_income: float = 0.0,
    scheduled_expense: float = 0.0,
    state_params: StateTaxParams = STATE_PRESETS["WA"],
    filing_status: str = "single",
    max_iter: int = 40,
    tol: float = 1.0,
    drawdown_from_peak: float = 0.0,
) -> PlanResult:
    """Solve the year's plan: how much to withdraw from each source and convert.

    Iterates to a fixed point because taxes depend on withdrawals which depend on
    target gross need which depends on taxes.
    """
    # SS + scheduled income reduce cash-flow requirement; scheduled expense raises it.
    effective_target = max(target_net + scheduled_expense - ss_income - scheduled_income, 0.0)
    gross_need = effective_target  # initial guess
    last = None

    for _ in range(max_iter):
        # Trial conversion sized using current LTCG estimate.
        ltcg_est = (
            min(gross_need, portfolio.taxable.balance) * portfolio.taxable.gain_ratio
            if portfolio.taxable.balance > 0
            else 0.0
        )
        conversion = _conversion_for_strategy(
            strategy_name, portfolio, age, ltcg_est, params, custom_conversion,
            drawdown_from_peak=drawdown_from_peak,
        )

        w, ltcg, shortfall = _fund_priority(portfolio, age, year, gross_need)

        # Force-bump traditional withdrawal to satisfy RMD (capped at actual balance).
        if rmd_amount > w.traditional:
            w.traditional = min(rmd_amount, portfolio.traditional.balance)

        # Only the taxable portion of SS enters ordinary income.
        other_ord = conversion + w.traditional + w.hsa + scheduled_taxable_income
        ss_taxable = taxable_ss(ss_income, other_ord, ltcg, filing_status)
        # Ordinary income = conversion + traditional withdrawals + non-qualified HSA distribution
        # + scheduled taxable income streams (rental, pension, etc.).
        # (HSA non-medical pre-65 also incurs 20% penalty; we ignore HSA penalty for v1
        # since HSA is last-resort and the simulator should never reach it under reasonable inputs.)
        ordinary_income = other_ord + ss_taxable

        tax = federal_tax(ordinary_income, ltcg, params)
        penalty = early_withdrawal_penalty(w.traditional, age)
        magi = ordinary_income + ltcg
        if age >= 65:
            healthcare = medicare_premium(magi, age, filing_status)
        else:
            healthcare = aca_premium(magi, params, aca_mode)

        state_tax_amt = _state_tax_fn(ordinary_income, ltcg, state_params)
        new_gross = effective_target + tax["total"] + penalty + healthcare["out_of_pocket"] + state_tax_amt

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
            ss_income=ss_income,
            rmd_amount=rmd_amount,
            state_tax=state_tax_amt,
            gross_used=gross_need,
            scheduled_income=scheduled_income,
            scheduled_taxable_income=scheduled_taxable_income,
            scheduled_expense=scheduled_expense,
        )

        if abs(new_gross - gross_need) < tol:
            break
        # Damped update: prevents oscillation near the ACA cliff discontinuity at 400% FPL.
        gross_need = 0.5 * gross_need + 0.5 * new_gross

    return last


# --- preset registry --------------------------------------------------------

STRATEGY_PRESETS = ["bridge_optimal", "bridge_guarded", "bridge_responsive", "minimal_convert", "aggressive_convert", "custom"]

STRATEGY_DESCRIPTIONS = {
    "bridge_optimal": (
        "Spend taxable + cash; convert Trad→Roth up to the lesser of (0% LTCG ceiling) and "
        "(400% FPL ceiling) each year."
    ),
    "bridge_guarded": (
        "Like bridge_optimal but caps each year's conversion at trad_balance ÷ years_to_60, "
        "preventing the bracket target from draining Trad before the bridge ends. More "
        "robust under bad sequences; trades some Roth-stacking upside for ladder-gap insurance."
    ),
    "bridge_responsive": (
        "Like bridge_optimal but scales conversion by drawdown from peak — minimal in good times "
        "(preserves Trad cushion, avoids tax-prepay on dollars that may crash), full bracket-fill "
        "at ≥20% drawdown (cheap conversion when prices are depressed). Targets preservation, not depletion."
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
