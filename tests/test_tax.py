"""Pin tax-engine math against the year-1 numbers worked out in conversation."""

import math

from planner.tax import (
    TAX_PARAMS_2026,
    aca_premium,
    early_withdrawal_penalty,
    federal_tax,
    fpl_400_ceiling,
    zero_ltcg_ceiling,
)


def approx(a, b, tol=1.0):
    return abs(a - b) <= tol


def test_year1_conversation_case():
    """$80k taxable sale (gain ratio 31.6%) + $37k Roth conversion.

    Expected: federal tax ~$2,310, AGI ~$62,300, all LTCG in 0% bracket.
    """
    ltcg = 80_000 * 0.3163  # ~25,304
    ordinary = 37_000
    out = federal_tax(ordinary, ltcg)
    assert approx(out["agi"], 62_304, tol=10), out
    assert approx(out["ordinary"], 2_311.5, tol=5), out
    assert out["ltcg"] == 0.0
    assert approx(out["total"], 2_311.5, tol=5), out


def test_standard_deduction_only_conversion_is_zero_tax():
    """Conversion that exactly fills standard deduction with no LTCG -> $0 tax."""
    out = federal_tax(ordinary_income=TAX_PARAMS_2026.standard_deduction, ltcg=0)
    assert out["total"] == 0.0


def test_ltcg_pushed_out_of_zero_bracket():
    """If ordinary taxable + LTCG exceeds the 0% LTCG ceiling, excess is taxed at 15%."""
    # ordinary taxable = 30k, LTCG = 30k -> total stack = 60k, 0% ceiling 48,350.
    # excess = 60,000 - 48,350 = 11,650 of LTCG at 15% = 1,747.50
    ord_income = 30_000 + TAX_PARAMS_2026.standard_deduction
    out = federal_tax(ord_income, ltcg=30_000)
    assert approx(out["ltcg"], 1_747.5, tol=2), out


def test_ltcg_stacking_when_ordinary_already_above_ceiling():
    """If ordinary taxable already exceeds 0% LTCG ceiling, ALL gains taxed at 15%+."""
    ord_income = 70_000 + TAX_PARAMS_2026.standard_deduction  # taxable = 70k > 48,350
    out = federal_tax(ord_income, ltcg=10_000)
    assert approx(out["ltcg"], 1_500, tol=1), out


def test_aca_subsidized_under_400_fpl():
    """At ~383% FPL, premium contribution should be under 8.5% of MAGI."""
    magi = 60_000
    out = aca_premium(magi, mode="cap")
    fpl_pct = 100 * magi / TAX_PARAMS_2026.fpl_single
    assert 380 < fpl_pct < 400
    # Rate at fpl_pct (300-400 band): 6% + (fpl_pct - 300)/100 * 2.5%
    expected_rate = 0.06 + (fpl_pct - 300) / 100 * 0.025
    expected = expected_rate * magi
    assert approx(out["out_of_pocket"], expected, tol=1), out


def test_aca_cliff_above_400_fpl():
    """In cliff mode, MAGI > 400% FPL pays full premium."""
    magi = 70_000
    out = aca_premium(magi, mode="cliff")
    assert out["out_of_pocket"] == TAX_PARAMS_2026.benchmark_premium


def test_aca_cap_above_400_fpl():
    """In cap mode, MAGI > 400% FPL still capped at 8.5%."""
    magi = 80_000
    out = aca_premium(magi, mode="cap")
    expected = min(0.085 * magi, TAX_PARAMS_2026.benchmark_premium)
    assert approx(out["out_of_pocket"], expected, tol=1), out


def test_early_withdrawal_penalty():
    assert early_withdrawal_penalty(10_000, age=40) == 1_000
    assert early_withdrawal_penalty(10_000, age=60) == 0


def test_zero_ltcg_ceiling_matches_year1_conversion_size():
    """With $25,304 LTCG, the 0%-LTCG-preserving ordinary income ceiling is ~$38,800."""
    ceiling = zero_ltcg_ceiling(80_000 * 0.3163)
    # std_ded (15,750) + (48,350 - 25,304) = 15,750 + 23,046 = 38,796
    assert approx(ceiling, 38_796, tol=5)


def test_fpl_400_ceiling():
    assert fpl_400_ceiling() == 4 * TAX_PARAMS_2026.fpl_single


def test_medicare_irmaa_tiers():
    from planner.tax import medicare_premium
    out = medicare_premium(magi=80_000, age=66)
    assert out["irmaa_surcharge"] == 0.0
    assert out["tier"] == 0
    out = medicare_premium(magi=120_000, age=66)
    assert out["tier"] == 1
    assert out["irmaa_surcharge"] > 0
    out = medicare_premium(magi=600_000, age=66)
    assert out["tier"] == 5


def test_rmd_below_age_73_is_zero():
    from planner.tax import required_min_distribution
    assert required_min_distribution(1_000_000, age=72) == 0.0


def test_rmd_at_age_73():
    from planner.tax import required_min_distribution
    rmd = required_min_distribution(1_000_000, age=73)
    assert approx(rmd, 1_000_000 / 26.5, tol=1)


def test_rmd_grows_with_age():
    from planner.tax import required_min_distribution
    assert required_min_distribution(500_000, age=85) > required_min_distribution(500_000, age=73)


def test_monte_carlo_smoke():
    from planner.simulate import SimulationInputs
    from planner.returns import LognormalReturns
    from planner.montecarlo import run_monte_carlo
    inputs = SimulationInputs(horizon_years=20)
    model = LognormalReturns(seed=42)
    mc = run_monte_carlo(inputs, returns_model=model, n_runs=50)
    assert mc.n_runs == 50
    assert 0.0 <= mc.success_rate <= 1.0
    assert len(mc.p50_balance) == 20
    assert mc.median_ending > 0


def test_lognormal_returns_reproducible():
    from planner.returns import LognormalReturns
    m = LognormalReturns(seed=7)
    a = m.get(year_index=3, path_index=10)
    b = m.get(year_index=3, path_index=10)
    assert a == b
    # Different paths produce different draws
    c = m.get(year_index=3, path_index=11)
    assert a != c
