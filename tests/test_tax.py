"""Pin tax-engine math against the year-1 numbers worked out in conversation."""

import math

from planner.tax import (
    TAX_PARAMS_2026,
    aca_premium,
    early_withdrawal_penalty,
    federal_tax,
    fpl_400_ceiling,
    taxable_ss,
    wa_ltcg_tax,
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


def test_historical_playback_paths_and_replay():
    from planner.returns import HistoricalPlayback
    m = HistoricalPlayback(horizon_years=60)
    # Shiller data ends 2022; floor 1928 -> valid starts 1928..1963 = 36
    assert m.n_paths == 36
    assert m.start_years[0] == 1928
    assert m.start_years[-1] == 1963
    # Replay is deterministic and indexed by start year offset
    y3 = m.get(year_index=3, path_index=0)  # 1931 — historical depression year
    assert y3.stocks < -0.30  # 1931 was -38%
    # Same (year, path) repeats identically
    assert m.get(0, 5) == m.get(0, 5)


def test_historical_playback_horizon_too_long_no_paths():
    from planner.returns import HistoricalPlayback
    m = HistoricalPlayback(horizon_years=120)
    assert m.n_paths == 0


def test_historical_playback_in_montecarlo():
    from planner.returns import HistoricalPlayback
    from planner.montecarlo import run_monte_carlo
    from planner.simulate import SimulationInputs
    inputs = SimulationInputs(horizon_years=30)
    m = HistoricalPlayback(horizon_years=30)
    mc = run_monte_carlo(inputs, returns_model=m, n_runs=m.n_paths)
    assert mc.n_runs == m.n_paths
    assert 0.0 <= mc.success_rate <= 1.0
    assert len(mc.p50_balance) == 30


def test_bridge_guarded_caps_conversion_in_pre60_years():
    """At age 40 with $100k Trad, conversion must be <= $100k / (60-40) = $5k regardless of bracket target."""
    from planner.simulate import SimulationInputs, simulate
    from planner.returns import ConstantReturns
    inputs = SimulationInputs(
        initial_cash=20_000, initial_taxable=200_000, taxable_basis=200_000,
        initial_traditional=100_000, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=40, horizon_years=2,
        strategy="bridge_guarded",
    )
    r = simulate(inputs, returns_model=ConstantReturns(stocks=0, bonds=0, cash=0))
    # year 0 (age 40): cap = 100k / 20 = 5k. Conversion <= 5k.
    assert r[0].plan.conversion <= 5_000 + 1, f"Expected conversion <= ~5k, got {r[0].plan.conversion}"


def test_income_stream_offsets_gross_need():
    """A non-taxable income stream reduces the cash-flow need 1:1, no tax effect."""
    from planner.simulate import SimulationInputs, simulate
    from planner.streams import IncomeStream
    from planner.returns import ConstantReturns
    base = SimulationInputs(
        initial_cash=20_000, initial_taxable=200_000, taxable_basis=200_000,
        initial_traditional=0, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=40, horizon_years=2,
        strategy="minimal_convert",
    )
    with_stream = SimulationInputs(
        **{k: v for k, v in base.__dict__.items() if k != "income_streams"},
        income_streams=(IncomeStream(name="gift", annual_amount=20_000, start_age=40, end_age=40, taxable=False),),
    )
    a = simulate(base, returns_model=ConstantReturns(0, 0, 0))[0]
    b = simulate(with_stream, returns_model=ConstantReturns(0, 0, 0))[0]
    # Non-taxable income should reduce gross_used by ~$20k
    assert b.plan.gross_used < a.plan.gross_used - 19_000
    # Tax should be unchanged (no taxable income from the stream)
    assert abs(b.plan.federal_tax - a.plan.federal_tax) < 5  # $0 in this scenario


def test_taxable_income_stream_enters_ordinary_income():
    """A taxable income stream (rental) shows up in ordinary_income and bumps tax."""
    from planner.simulate import SimulationInputs, simulate
    from planner.streams import IncomeStream
    from planner.returns import ConstantReturns
    inp = SimulationInputs(
        initial_cash=20_000, initial_taxable=200_000, taxable_basis=200_000,
        initial_traditional=0, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=40, horizon_years=2,
        strategy="minimal_convert",
        income_streams=(IncomeStream(name="rental", annual_amount=30_000, start_age=40, end_age=80, taxable=True),),
    )
    r = simulate(inp, returns_model=ConstantReturns(0, 0, 0))[0]
    assert r.plan.scheduled_taxable_income == 30_000
    # Ordinary income should include the rental
    assert r.plan.ordinary_income >= 30_000


def test_expense_stream_increases_gross_need():
    """An expense stream increases the gross need by its amount, no tax effect."""
    from planner.simulate import SimulationInputs, simulate
    from planner.streams import ExpenseStream
    from planner.returns import ConstantReturns
    base = SimulationInputs(
        initial_cash=20_000, initial_taxable=300_000, taxable_basis=300_000,
        initial_traditional=0, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=40, horizon_years=2,
        strategy="minimal_convert",
    )
    with_exp = SimulationInputs(
        **{k: v for k, v in base.__dict__.items() if k != "expense_streams"},
        expense_streams=(ExpenseStream(name="property", annual_amount=15_000, start_age=40, end_age=90),),
    )
    a = simulate(base, returns_model=ConstantReturns(0, 0, 0))[0]
    b = simulate(with_exp, returns_model=ConstantReturns(0, 0, 0))[0]
    assert b.plan.gross_used >= a.plan.gross_used + 14_000
    assert b.plan.scheduled_expense == 15_000


def test_state_tax_no_tax_states_zero():
    from planner.state_tax import STATE_PRESETS, state_tax
    for code in ("NONE", "TX", "FL", "NV", "AK", "SD", "TN", "WY"):
        assert state_tax(100_000, 50_000, STATE_PRESETS[code]) == 0.0, code


def test_state_tax_wa_preserves_legacy_behavior():
    from planner.state_tax import STATE_PRESETS, state_tax
    wa = STATE_PRESETS["WA"]
    # Below threshold → 0
    assert state_tax(50_000, 250_000, wa) == 0.0
    # Above threshold → 7% on the excess only
    out = state_tax(50_000, 300_000, wa)
    assert approx(out, 0.07 * (300_000 - 262_000), tol=0.01)


def test_state_tax_custom_flat_rates():
    from planner.state_tax import resolve_state_params, state_tax
    params = resolve_state_params("CUSTOM", custom_ordinary_rate=0.05, custom_ltcg_rate=0.05)
    # 5% on $100k ord + 5% on $20k LTCG (no threshold) = $5k + $1k = $6k
    out = state_tax(100_000, 20_000, params)
    assert approx(out, 6_000, tol=0.01)


def test_state_tax_threads_through_simulate():
    from planner.simulate import SimulationInputs, simulate, summarize
    s_wa = summarize(simulate(SimulationInputs(state_code="WA")))
    s_tx = summarize(simulate(SimulationInputs(state_code="TX")))
    # TX always 0; WA may be 0 or positive (depends on per-year LTCG vs threshold)
    assert s_tx["total_state_tax"] == 0.0
    assert s_wa["total_state_tax"] >= 0.0


def test_swr_mode_resolves_to_rate_times_starting_total():
    from planner.simulate import SimulationInputs, simulate
    inp = SimulationInputs(spend_mode="swr", spend_rate=0.04, horizon_years=2)
    r = simulate(inp)
    starting = (inp.initial_cash + inp.initial_taxable + inp.initial_traditional
                + inp.initial_roth + inp.initial_hsa)
    expected = 0.04 * starting
    assert approx(r[0].target_net, expected, tol=0.01), (r[0].target_net, expected)


def test_streams_age_window_inclusive():
    """Stream is active when start_age <= age <= end_age, inactive otherwise."""
    from planner.simulate import SimulationInputs, simulate
    from planner.streams import IncomeStream
    from planner.returns import ConstantReturns
    inp = SimulationInputs(
        initial_cash=10_000, initial_taxable=500_000, taxable_basis=500_000,
        initial_traditional=0, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=40, horizon_years=5,
        strategy="minimal_convert",
        income_streams=(IncomeStream(name="r", annual_amount=10_000, start_age=42, end_age=43, taxable=True),),
    )
    rs = simulate(inp, returns_model=ConstantReturns(0, 0, 0))
    assert rs[0].plan.scheduled_income == 0          # age 40
    assert rs[1].plan.scheduled_income == 0          # age 41
    assert rs[2].plan.scheduled_income == 10_000     # age 42 (start)
    assert rs[3].plan.scheduled_income == 10_000     # age 43 (end)
    assert rs[4].plan.scheduled_income == 0          # age 44


def test_bridge_guarded_relaxes_cap_post60():
    """Post-60, no cap — should match bridge_optimal target sizing."""
    from planner.simulate import SimulationInputs, simulate
    from planner.returns import ConstantReturns
    inputs = SimulationInputs(
        initial_cash=20_000, initial_taxable=200_000, taxable_basis=200_000,
        initial_traditional=100_000, initial_roth=0, initial_hsa=0,
        target_spend=40_000, start_age=65, horizon_years=2,
        strategy="bridge_guarded",
    )
    r = simulate(inputs, returns_model=ConstantReturns(stocks=0, bonds=0, cash=0))
    # Post-60 should hit bracket target (typically std_ded floor ~ 15.7k or higher).
    assert r[0].plan.conversion >= 15_000, f"Expected post-60 conversion >= 15k, got {r[0].plan.conversion}"


# --- taxable_ss tests -------------------------------------------------------

def test_taxable_ss_zero_benefit():
    assert taxable_ss(0, 50_000, 10_000) == 0.0


def test_taxable_ss_low_provisional_under_25k():
    # provisional = 10_000 + 0 + 0.5 * 5_000 = 12_500 -> 0
    assert taxable_ss(5_000, 10_000, 0) == 0.0


def test_taxable_ss_mid_tier_50pct_cap():
    # provisional = 20_000 + 0 + 0.5 * 20_000 = 30_000 -> between 25k and 34k
    # taxable = min(0.5 * 20_000, 0.5 * (30_000 - 25_000)) = min(10_000, 2_500) = 2_500
    result = taxable_ss(20_000, 20_000, 0)
    assert approx(result, 2_500, tol=1)


def test_taxable_ss_high_provisional_above_34k():
    # SS = 30_000, other_ord = 60_000, ltcg = 0
    # provisional = 60_000 + 0 + 15_000 = 75_000 > 34_000
    # tier1 = min(0.5 * 30_000, 4_500) = 4_500
    # tier2 = 0.85 * (75_000 - 34_000) = 0.85 * 41_000 = 34_850
    # total = 39_350; cap = 0.85 * 30_000 = 25_500 -> returns 25_500
    result = taxable_ss(30_000, 60_000, 0)
    assert approx(result, 25_500, tol=1)


def test_taxable_ss_85pct_cap_binds():
    # Verify that at very high provisional income the 85% cap binds
    result = taxable_ss(50_000, 200_000, 0)
    assert approx(result, 0.85 * 50_000, tol=1)


# --- wa_ltcg_tax tests -------------------------------------------------------

def test_wa_ltcg_tax_below_threshold():
    assert wa_ltcg_tax(100_000) == 0.0


def test_wa_ltcg_tax_at_threshold():
    assert wa_ltcg_tax(262_000) == 0.0


def test_wa_ltcg_tax_above_threshold():
    # $10k above threshold -> 10_000 * 0.07 = 700
    assert approx(wa_ltcg_tax(272_000), 700.0, tol=0.01)


# --- RMD table extension test -----------------------------------------------

def test_rmd_table_age_110():
    from planner.tax import required_min_distribution
    # Should use divisor 3.5 (from extended table), not the old 6.4 fallback
    rmd = required_min_distribution(350_000, age=110)
    assert approx(rmd, 350_000 / 3.5, tol=1)


# --- RMD excess cash re-credit smoke test -----------------------------------

def test_rmd_excess_credits_cash():
    """When forced RMD > gross need, the surplus should appear in cash."""
    from planner.simulate import SimulationInputs, simulate

    # start_age=70 so RMDs begin at age 73 (year 3). Use a large traditional balance
    # and low spend so RMD overshoots gross need.
    inputs = SimulationInputs(
        start_age=70,
        horizon_years=10,
        initial_traditional=2_000_000,
        initial_taxable=0,
        taxable_basis=0,
        initial_cash=0,
        initial_roth=0,
        initial_hsa=0,
        target_spend=50_000,
        ss_annual_benefit=0,
        strategy="minimal_convert",
    )
    results = simulate(inputs)
    rmd_years = [yr for yr in results if yr.plan.rmd_amount > 0]
    assert len(rmd_years) > 0, "Expected RMD years"
    # In any year where rmd_amount > gross_used, cash should have increased
    found_surplus = False
    for yr in rmd_years:
        if yr.plan.rmd_amount > yr.plan.gross_used:
            # surplus was re-credited; cash balance should be positive
            assert yr.snapshot["cash"] > 0, f"Expected cash > 0 at age {yr.age}"
            found_surplus = True
            break
    assert found_surplus, "Expected at least one year where RMD overshoots gross need"


# --- bridge_responsive tests ------------------------------------------------

def _make_portfolio_with_trad(trad_balance: float = 500_000):
    from planner.accounts import Cash, HSA, Portfolio, RothIRA, Taxable, TraditionalIRA
    return Portfolio(
        cash=Cash(balance=0),
        taxable=Taxable(balance=0, basis=0),
        traditional=TraditionalIRA(balance=trad_balance),
        roth=RothIRA(contributions=0, earnings=0),
        hsa=HSA(balance=0),
    )


def test_bridge_responsive_zero_drawdown_minimal():
    """At drawdown=0, bridge_responsive target equals standard_deduction."""
    from planner.strategy import _conversion_for_strategy
    portfolio = _make_portfolio_with_trad(500_000)
    result = _conversion_for_strategy(
        "bridge_responsive", portfolio, age=40,
        ltcg_estimate=0, params=TAX_PARAMS_2026, custom_amount=None,
        drawdown_from_peak=0.0,
    )
    assert approx(result, TAX_PARAMS_2026.standard_deduction, tol=1), result


def test_bridge_responsive_full_drawdown_bridge_optimal():
    """At drawdown=0.20, bridge_responsive target equals bridge_optimal target."""
    from planner.strategy import _conversion_for_strategy
    portfolio = _make_portfolio_with_trad(500_000)
    ltcg_estimate = 5_000  # small, doesn't bind the LTCG ceiling
    result_responsive = _conversion_for_strategy(
        "bridge_responsive", portfolio, age=40,
        ltcg_estimate=ltcg_estimate, params=TAX_PARAMS_2026, custom_amount=None,
        drawdown_from_peak=0.20,
    )
    result_optimal = _conversion_for_strategy(
        "bridge_optimal", portfolio, age=40,
        ltcg_estimate=ltcg_estimate, params=TAX_PARAMS_2026, custom_amount=None,
    )
    assert approx(result_responsive, result_optimal, tol=1), (result_responsive, result_optimal)


def test_bridge_responsive_half_drawdown_midpoint():
    """At drawdown=0.10, bridge_responsive target is the midpoint of minimal and bridge_optimal."""
    from planner.strategy import _conversion_for_strategy
    portfolio = _make_portfolio_with_trad(500_000)
    ltcg_estimate = 5_000
    minimal = TAX_PARAMS_2026.standard_deduction
    result_optimal = _conversion_for_strategy(
        "bridge_optimal", portfolio, age=40,
        ltcg_estimate=ltcg_estimate, params=TAX_PARAMS_2026, custom_amount=None,
    )
    expected_midpoint = minimal + 0.5 * (result_optimal - minimal)
    result_responsive = _conversion_for_strategy(
        "bridge_responsive", portfolio, age=40,
        ltcg_estimate=ltcg_estimate, params=TAX_PARAMS_2026, custom_amount=None,
        drawdown_from_peak=0.10,
    )
    assert approx(result_responsive, expected_midpoint, tol=1), (result_responsive, expected_midpoint)


def test_preservation_rate_basic():
    """preservation_rate equals fraction of paths with ending_total >= starting_total."""
    from planner.montecarlo import MCSummary, PathSummary, run_monte_carlo
    from planner.simulate import SimulationInputs, build_portfolio
    from planner.returns import ConstantReturns

    # Use a small deterministic MC: constant 0% returns, tiny horizon.
    # With 0% real returns and positive withdrawals the portfolio shrinks every year,
    # so ending_total < starting_total for all paths -> preservation_rate == 0.
    inputs = SimulationInputs(horizon_years=5, stock_return=0.0, bond_return=0.0, cash_return=0.0)
    model = ConstantReturns(stocks=0.0, bonds=0.0, cash=0.0)
    mc = run_monte_carlo(inputs, returns_model=model, n_runs=5)
    assert mc.starting_total == build_portfolio(inputs).total
    assert 0.0 <= mc.preservation_rate <= 1.0
    # With 0% returns and spending, portfolio shrinks: expect 0% preservation
    assert mc.preservation_rate == 0.0
