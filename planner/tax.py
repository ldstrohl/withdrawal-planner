"""Federal tax engine: ordinary brackets, LTCG with stacking, ACA subsidy, early-withdrawal penalty.

All math is done in nominal dollars for the supplied tax-year params. The simulator
operates in real dollars and so passes constant params each year.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Literal


@dataclass(frozen=True)
class TaxParams:
    """Single-filer tax parameters. 2025 actuals, taken as default for 2026 estimate.

    Brackets are (upper_limit, rate) pairs sorted ascending; final entry uses inf as upper.
    Standard deduction and FPL are nominal dollars.
    """

    standard_deduction: float
    ordinary_brackets: Tuple[Tuple[float, float], ...]
    ltcg_brackets: Tuple[Tuple[float, float], ...]
    fpl_single: float
    benchmark_premium: float

    label: str = ""


INF = float("inf")


TAX_PARAMS_2026 = TaxParams(
    standard_deduction=15_750.0,
    ordinary_brackets=(
        (11_925.0, 0.10),
        (48_475.0, 0.12),
        (103_350.0, 0.22),
        (197_300.0, 0.24),
        (250_525.0, 0.32),
        (626_350.0, 0.35),
        (INF, 0.37),
    ),
    ltcg_brackets=(
        (48_350.0, 0.00),
        (533_400.0, 0.15),
        (INF, 0.20),
    ),
    fpl_single=15_650.0,
    benchmark_premium=8_000.0,
    label="2026 (single, projected from 2025 actuals)",
)


def _apply_brackets(taxable: float, brackets: Tuple[Tuple[float, float], ...]) -> float:
    if taxable <= 0:
        return 0.0
    tax = 0.0
    prev_limit = 0.0
    for upper, rate in brackets:
        if taxable <= upper:
            tax += (taxable - prev_limit) * rate
            return tax
        tax += (upper - prev_limit) * rate
        prev_limit = upper
    return tax


def federal_tax(
    ordinary_income: float,
    ltcg: float,
    params: TaxParams = TAX_PARAMS_2026,
) -> dict:
    """Compute federal tax with LTCG stacking on top of ordinary taxable income.

    LTCG fills bracket space *above* ordinary taxable income. So the 0% LTCG bracket
    is consumed by ordinary taxable first, then any remaining 0% room applies to gains.
    """
    ordinary_taxable = max(ordinary_income - params.standard_deduction, 0.0)
    ordinary_tax = _apply_brackets(ordinary_taxable, params.ordinary_brackets)

    # LTCG is taxed in the brackets above ordinary_taxable.
    ltcg_tax = 0.0
    remaining = max(ltcg, 0.0)
    floor = ordinary_taxable
    prev_limit = 0.0
    for upper, rate in params.ltcg_brackets:
        if remaining <= 0:
            break
        if upper <= floor:
            prev_limit = upper
            continue
        bracket_lo = max(prev_limit, floor)
        bracket_hi = upper
        room = bracket_hi - bracket_lo
        taxed = min(remaining, room)
        ltcg_tax += taxed * rate
        remaining -= taxed
        prev_limit = upper

    return {
        "ordinary": ordinary_tax,
        "ltcg": ltcg_tax,
        "total": ordinary_tax + ltcg_tax,
        "ordinary_taxable": ordinary_taxable,
        "agi": ordinary_income + ltcg,
    }


def _premium_contribution_rate(fpl_pct: float) -> float:
    """ACA premium contribution rate as % of MAGI, IRA-expanded schedule.

    Piecewise linear:
      0-150% FPL: 0%
      150-200%: 0 -> 2%
      200-250%: 2 -> 4%
      250-300%: 4 -> 6%
      300-400%: 6 -> 8.5%
      400%+: 8.5%
    """
    if fpl_pct <= 150:
        return 0.0
    if fpl_pct <= 200:
        return (fpl_pct - 150) / 50 * 0.02
    if fpl_pct <= 250:
        return 0.02 + (fpl_pct - 200) / 50 * 0.02
    if fpl_pct <= 300:
        return 0.04 + (fpl_pct - 250) / 50 * 0.02
    if fpl_pct <= 400:
        return 0.06 + (fpl_pct - 300) / 100 * 0.025
    return 0.085


def aca_premium(
    magi: float,
    params: TaxParams = TAX_PARAMS_2026,
    mode: Literal["cliff", "cap"] = "cap",
) -> dict:
    """Out-of-pocket premium estimate.

    cliff: subsidy disappears entirely above 400% FPL (pre-IRA rules).
    cap: 8.5% of MAGI cap continues above 400% FPL (IRA-expanded).
    """
    fpl_pct = 100 * magi / params.fpl_single if magi > 0 else 0.0
    full_premium = params.benchmark_premium

    if mode == "cliff" and fpl_pct > 400:
        oop = full_premium
    else:
        rate = _premium_contribution_rate(fpl_pct)
        oop = min(rate * magi, full_premium)

    return {
        "out_of_pocket": oop,
        "fpl_pct": fpl_pct,
        "subsidy": max(full_premium - oop, 0.0),
    }


def early_withdrawal_penalty(traditional_withdrawal: float, age: int) -> float:
    """10% penalty on Traditional/401k withdrawals before age 59.5."""
    if age >= 60 or traditional_withdrawal <= 0:
        return 0.0
    return 0.10 * traditional_withdrawal


def zero_ltcg_ceiling(ltcg: float, params: TaxParams = TAX_PARAMS_2026) -> float:
    """Largest ordinary income that keeps all `ltcg` inside the 0% LTCG bracket.

    ordinary_taxable + ltcg <= ltcg_brackets[0].upper
    -> ordinary_income <= standard_deduction + (0%_ltcg_ceiling - ltcg)
    """
    zero_top = params.ltcg_brackets[0][0]
    return params.standard_deduction + max(zero_top - ltcg, 0.0)


def fpl_400_ceiling(params: TaxParams = TAX_PARAMS_2026) -> float:
    return 4.0 * params.fpl_single
