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


# 2025 Medicare Part B + Part D base + IRMAA tiers (single filer).
# IRMAA is based on MAGI from 2 years prior; for simplicity we use current-year MAGI.
MEDICARE_BASE_ANNUAL = 2_220.00   # 2025: Part B base $185.00/mo * 12
MEDICARE_PARTD_BASE = 480.0        # Part D base ~$40/mo * 12

# (MAGI upper limit, total annual surcharge above base, Part B + Part D combined)
IRMAA_TIERS_SINGLE = (
    (106_000.0, 0.0),
    (133_000.0, 1_028.40 + 165.60),
    (167_000.0, 2_569.20 + 425.40),
    (200_000.0, 4_110.00 + 686.40),
    (500_000.0, 5_650.80 + 947.40),
    (float("inf"), 6_165.60 + 1_032.00),
)


def medicare_premium(magi: float, age: int) -> dict:
    """Medicare Part B + D base + IRMAA surcharge.

    Caller is responsible for invoking only at age >= 65.
    Returns {"out_of_pocket", "irmaa_surcharge", "tier", "base"}.
    """
    base = MEDICARE_BASE_ANNUAL + MEDICARE_PARTD_BASE
    surcharge = 0.0
    tier = 0
    for i, (upper, extra) in enumerate(IRMAA_TIERS_SINGLE):
        if magi <= upper:
            surcharge = extra
            tier = i
            break
    return {
        "out_of_pocket": base + surcharge,
        "irmaa_surcharge": surcharge,
        "tier": tier,
        "base": base,
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


# IRS Uniform Lifetime Table (2022 update). RMD = prior_year_balance / divisor.
UNIFORM_LIFETIME_TABLE = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}


def required_min_distribution(traditional_balance: float, age: int) -> float:
    """RMD on Traditional accounts. Returns 0 if age < 73 or no balance."""
    if age < 73 or traditional_balance <= 0:
        return 0.0
    divisor = UNIFORM_LIFETIME_TABLE.get(age, 2.0)
    return traditional_balance / divisor


def taxable_ss(ss_benefit: float, other_ordinary_income: float, ltcg: float) -> float:
    """IRS provisional-income test for SS taxation (single filer, 2025 thresholds, NOT inflation-indexed by IRS).

    provisional = other_ordinary_income + ltcg + 0.5 * ss_benefit
    Threshold 1: $25,000 — below this, none of SS is taxable.
    Threshold 2: $34,000 — between, up to 50% of SS or 50% of (provisional - 25k), lesser.
    Above $34,000 — up to 85% of SS, plus the 50% phase-in piece.
    Returns dollars of SS that are added to ordinary taxable income.
    """
    if ss_benefit <= 0:
        return 0.0
    provisional = other_ordinary_income + ltcg + 0.5 * ss_benefit
    if provisional <= 25_000:
        return 0.0
    if provisional <= 34_000:
        return min(0.5 * ss_benefit, 0.5 * (provisional - 25_000))
    # Above 34k: 85% of (provisional - 34k), plus the lesser of 50% SS or $4,500 (= 0.5*(34k-25k)),
    # capped at 85% of total SS benefit.
    tier2 = 0.85 * (provisional - 34_000)
    tier1 = min(0.5 * ss_benefit, 4_500.0)
    return min(tier1 + tier2, 0.85 * ss_benefit)


# Kept for backward compatibility — new code should use planner.state_tax instead.
WA_LTCG_THRESHOLD = 262_000.0
WA_LTCG_RATE = 0.07


def wa_ltcg_tax(ltcg: float) -> float:
    """Deprecated: use planner.state_tax.state_tax with STATE_PRESETS["WA"]."""
    excess = max(ltcg - WA_LTCG_THRESHOLD, 0.0)
    return excess * WA_LTCG_RATE
