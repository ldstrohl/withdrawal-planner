"""Streamlit GUI for the Early Retirement Withdrawal Planner.

Run: streamlit run app.py
"""

from __future__ import annotations

import json as _json
import dataclasses as _dc

import streamlit as st

import charts
from planner.montecarlo import run_monte_carlo
from planner.returns import HistoricalPlayback, LognormalReturns
from planner.simulate import SimulationInputs, simulate, summarize
from planner.strategy import STRATEGY_DESCRIPTIONS, STRATEGY_PRESETS


st.set_page_config(page_title="Withdrawal Planner", layout="wide", initial_sidebar_state="expanded")


STRATEGY_DISPLAY = {
    "bridge_optimal": "Bridge optimal",
    "minimal_convert": "Minimal convert",
    "aggressive_convert": "Aggressive convert",
    "custom": "Custom",
}

# Defaults for sidebar inputs (used for scenario state initialization)
_SIDEBAR_DEFAULTS = {
    "initial_cash": 25_000,
    "initial_taxable": 834_843,
    "taxable_basis": 570_659,
    "initial_traditional": 837_547,
    "initial_roth": 327_610,
    "roth_contributions": 0,
    "initial_hsa": 26_000,
    "target_spend": 80_000,
    "stock_return": 0.07,
    "bond_return": 0.02,
    "cash_return": 0.0,
    "stock_allocation": 0.85,
    "start_age": 35,
    "horizon_years": 60,
    "ss_annual_benefit": 0,
    "ss_claim_age": 67,
    "aca_mode": "cap",
}


@st.cache_data
def cached_simulate(inputs: SimulationInputs):
    return simulate(inputs)


@st.cache_data(show_spinner="Running Monte Carlo (lognormal)...")
def cached_mc_lognormal(
    inputs: SimulationInputs,
    n_runs: int,
    seed: int,
    sigma_s: float,
    sigma_b: float,
    sigma_c: float,
    rho: float,
):
    model = LognormalReturns(
        mu_stocks=inputs.stock_return,
        sigma_stocks=sigma_s,
        mu_bonds=inputs.bond_return,
        sigma_bonds=sigma_b,
        mu_cash=inputs.cash_return,
        sigma_cash=sigma_c,
        seed=seed,
        stock_bond_correlation=rho,
    )
    return run_monte_carlo(inputs, returns_model=model, n_runs=n_runs)


@st.cache_data(show_spinner="Running historical playback...")
def cached_mc_historical(inputs: SimulationInputs, start_year_floor: int):
    """Rolling-start replay of historical real returns. n_runs is determined by data coverage."""
    model = HistoricalPlayback(
        horizon_years=inputs.horizon_years,
        start_year_floor=start_year_floor,
    )
    return run_monte_carlo(inputs, returns_model=model, n_runs=model.n_paths)


# Streamlit ≥1.32 supports @st.fragment; older falls back to identity decorator.
try:
    _fragment = st.fragment
except AttributeError:
    def _fragment(f):
        return f


def render_sidebar() -> SimulationInputs:
    # Initialize session state defaults so widgets can use key= only (no value=)
    for k, v in _SIDEBAR_DEFAULTS.items():
        st.session_state.setdefault(f"scn_{k}", v)

    # G. Scenario save/load — top of sidebar
    with st.sidebar.expander("Scenario (save / load)", expanded=False):
        uploaded = st.file_uploader("Load JSON", type="json", key="scenario_upload")
        if uploaded is not None:
            data = _json.load(uploaded)
            for k, v in data.items():
                if f"scn_{k}" in st.session_state or k in _SIDEBAR_DEFAULTS:
                    st.session_state[f"scn_{k}"] = v
            st.success("Scenario loaded — adjust below if needed.")

    st.sidebar.markdown("### Initial balances (real $)")
    cash = st.sidebar.number_input("Cash", step=1_000, key="scn_initial_cash")
    taxable = st.sidebar.number_input("Taxable brokerage", step=10_000, key="scn_initial_taxable")
    basis = st.sidebar.number_input("Taxable cost basis", step=10_000, key="scn_taxable_basis")
    traditional = st.sidebar.number_input("Traditional 401k / IRA", step=10_000, key="scn_initial_traditional")
    roth = st.sidebar.number_input("Roth IRA", step=10_000, key="scn_initial_roth")
    roth_contrib = st.sidebar.number_input(
        "...of which is direct contributions", step=1_000,
        key="scn_roth_contributions",
        help="Roth contributions can be withdrawn anytime tax/penalty free.",
    )
    hsa = st.sidebar.number_input("HSA", step=1_000, key="scn_initial_hsa")

    st.sidebar.markdown("### Spending")
    target = st.sidebar.number_input("Target net annual spend (real $)", step=1_000, key="scn_target_spend")

    # A. Collapse "Real returns by asset class"
    with st.sidebar.expander("Real returns by asset class", expanded=False):
        stock_return = st.slider(
            "Stocks", min_value=0.0, max_value=0.12, step=0.005, format="%.3f",
            key="scn_stock_return",
            help="Long-run real (after-inflation) total return on equities.",
        )
        bond_return = st.slider(
            "Bonds", min_value=-0.02, max_value=0.06, step=0.005, format="%.3f",
            key="scn_bond_return",
            help="Long-run real total return on intermediate bonds.",
        )
        cash_return = st.slider(
            "Cash", min_value=-0.02, max_value=0.04, step=0.005, format="%.3f",
            key="scn_cash_return",
            help="Real return on cash (often near 0 after inflation).",
        )
        stock_alloc = st.slider(
            "Stock allocation (all investment accounts)",
            min_value=0.0, max_value=1.0, step=0.05,
            key="scn_stock_allocation",
            help="Fraction of each tax-advantaged + taxable account held in stocks vs bonds. Cash sits separately.",
        )

    st.sidebar.markdown("### Horizon")
    start_age = st.sidebar.number_input("Retirement age", step=1, key="scn_start_age")
    horizon = st.sidebar.number_input("Years to simulate", step=1, key="scn_horizon_years")

    # A. Collapse "Social Security"
    with st.sidebar.expander("Social Security", expanded=False):
        ss_benefit = st.number_input(
            "Annual SS benefit (real $)", step=1_000,
            key="scn_ss_annual_benefit",
            help="Estimated annual benefit in today's dollars. Use ssa.gov estimator. Set 0 to disable.",
        )
        ss_claim = st.number_input(
            "SS claim age", min_value=62, max_value=70, step=1,
            key="scn_ss_claim_age",
            help="62=earliest, 67=full retirement age, 70=max delay.",
        )

    st.sidebar.markdown("### ACA assumption")
    aca_mode = st.sidebar.radio(
        "Subsidy schedule",
        options=["cap", "cliff"],
        format_func=lambda m: {"cap": "8.5% cap (IRA-extended)", "cliff": "Cliff at 400% FPL"}[m],
        key="scn_aca_mode",
        help=(
            "IRA-extended (cap): premium contribution capped at 8.5% of MAGI even above "
            "400% FPL — current law through 2025. Cliff: subsidies vanish entirely above "
            "400% FPL. Use 'cliff' for conservative planning if you expect the IRA "
            "expansion to lapse."
        ),
    )

    # G. Download button for scenario save
    inputs_dict = {
        k: v for k, v in _dc.asdict(SimulationInputs(
            initial_cash=float(cash),
            initial_taxable=float(taxable),
            taxable_basis=float(basis),
            initial_traditional=float(traditional),
            initial_roth=float(roth),
            roth_contributions=float(roth_contrib),
            initial_hsa=float(hsa),
            target_spend=float(target),
            stock_return=float(stock_return),
            bond_return=float(bond_return),
            cash_return=float(cash_return),
            stock_allocation=float(stock_alloc),
            start_age=int(start_age),
            horizon_years=int(horizon),
            aca_mode=aca_mode,
            ss_annual_benefit=float(ss_benefit),
            ss_claim_age=int(ss_claim),
        )).items()
        if k != "params"
    }
    st.sidebar.download_button(
        "Save scenario JSON",
        data=_json.dumps(inputs_dict, indent=2),
        file_name="scenario.json",
        mime="application/json",
    )

    return SimulationInputs(
        initial_cash=float(cash),
        initial_taxable=float(taxable),
        taxable_basis=float(basis),
        initial_traditional=float(traditional),
        initial_roth=float(roth),
        roth_contributions=float(roth_contrib),
        initial_hsa=float(hsa),
        target_spend=float(target),
        stock_return=float(stock_return),
        bond_return=float(bond_return),
        cash_return=float(cash_return),
        stock_allocation=float(stock_alloc),
        start_age=int(start_age),
        horizon_years=int(horizon),
        aca_mode=aca_mode,
        ss_annual_benefit=float(ss_benefit),
        ss_claim_age=int(ss_claim),
    )


def kpi_row(results, summary: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Years funded", summary.get("years_funded", 0))
    cols[1].metric("Ending balance (real $)", f"${summary.get('ending_total', 0):,.0f}")
    cols[2].metric("Lifetime federal tax", f"${summary.get('total_federal_tax', 0):,.0f}")
    cols[3].metric("Lifetime healthcare OOP", f"${summary.get('total_healthcare_oop', 0):,.0f}")
    cols[4].metric(
        "Penalty paid",
        f"${summary.get('total_penalty', 0):,.0f}",
        delta=("ladder gap" if summary.get("total_penalty", 0) > 100 else "none"),
        delta_color="inverse" if summary.get("total_penalty", 0) > 100 else "off",
    )
    if summary.get("total_shortfall", 0) > 0.01:
        st.error(f"Plan does not fully fund target: lifetime shortfall ${summary['total_shortfall']:,.0f}")


@_fragment
def single_scenario_view(base: SimulationInputs) -> None:
    st.markdown("#### Strategy")
    cols = st.columns([2, 3])
    with cols[0]:
        strategy = st.selectbox("Withdrawal/conversion strategy", STRATEGY_PRESETS, index=0)
    with cols[1]:
        st.caption(STRATEGY_DESCRIPTIONS.get(strategy, ""))

    if strategy == "custom":
        st.number_input("Annual conversion amount", step=1_000, key="custom_conversion")
    custom_amount = st.session_state["custom_conversion"] if strategy == "custom" else None

    inputs = SimulationInputs(**{**base.__dict__, "strategy": strategy, "custom_conversion": custom_amount})
    results = cached_simulate(inputs)
    summary = summarize(results)

    kpi_row(results, summary)

    st.plotly_chart(charts.balance_stack(results), use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(charts.cashflow_bars(results), use_container_width=True)
    with cols[1]:
        st.plotly_chart(charts.tax_breakdown(results), use_container_width=True)

    st.plotly_chart(charts.conversions_bars(results), use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(charts.withdrawal_rate(results), use_container_width=True)
    with cols[1]:
        st.plotly_chart(charts.ladder_status(results), use_container_width=True)

    with st.expander("Per-year detail (table)"):
        df = charts.per_year_table(results)
        st.dataframe(df, hide_index=True, use_container_width=True)


@_fragment
def comparison_view(base: SimulationInputs) -> None:
    st.markdown("#### Strategies to compare")
    chosen = st.multiselect(
        "Pick 2–4",
        STRATEGY_PRESETS,
        default=["bridge_optimal", "minimal_convert", "aggressive_convert"],
    )
    if "custom" in chosen:
        st.number_input("Custom annual conversion", step=1_000, key="custom_conversion")
    custom_amount = st.session_state["custom_conversion"]

    if len(chosen) < 2:
        st.info("Select at least 2 strategies.")
        return

    # C. Optional per-strategy MC checkbox + model choice
    mc_cols = st.columns([2, 3])
    with mc_cols[0]:
        mc_compare = st.checkbox("Run Monte Carlo per strategy (slower)", value=False)
    with mc_cols[1]:
        mc_compare_model = st.radio(
            "MC returns model",
            options=["lognormal", "historical"],
            format_func=lambda m: {"lognormal": "Lognormal (200 paths)", "historical": "Historical (rolling start)"}[m],
            horizontal=True,
            disabled=not mc_compare,
            key="mc_compare_model",
        )

    scenarios = {}
    for name in chosen:
        inputs = SimulationInputs(**{
            **base.__dict__,
            "strategy": name,
            "custom_conversion": custom_amount if name == "custom" else None,
        })
        scenarios[name] = cached_simulate(inputs)

    success_rates: dict[str, float] = {}
    mc_label = ""
    if mc_compare:
        if mc_compare_model == "lognormal":
            for name in chosen:
                mc_inputs = SimulationInputs(**{
                    **base.__dict__,
                    "strategy": name,
                    "custom_conversion": custom_amount if name == "custom" else None,
                })
                mc_result = cached_mc_lognormal(
                    inputs=mc_inputs, n_runs=200, seed=42,
                    sigma_s=0.18, sigma_b=0.07, sigma_c=0.01, rho=0.05,
                )
                success_rates[name] = mc_result.success_rate
            mc_label = "Lognormal · 200 paths"
        else:
            preview = HistoricalPlayback(horizon_years=base.horizon_years)
            if preview.n_paths == 0:
                st.error(
                    f"Horizon of {base.horizon_years} years exceeds the historical dataset "
                    f"({preview.coverage[0]}–{preview.coverage[1]}). Switch to Lognormal or reduce horizon."
                )
            else:
                for name in chosen:
                    mc_inputs = SimulationInputs(**{
                        **base.__dict__,
                        "strategy": name,
                        "custom_conversion": custom_amount if name == "custom" else None,
                    })
                    mc_result = cached_mc_historical(inputs=mc_inputs, start_year_floor=1928)
                    success_rates[name] = mc_result.success_rate
                mc_label = f"Historical · {preview.n_paths} sequences ({preview.start_years[0]}–{preview.start_years[-1]})"

    # Summary table
    rows = []
    for name, results in scenarios.items():
        s = summarize(results)
        row = {
            "Strategy": STRATEGY_DISPLAY.get(name, name),
            "Ending (real $)": f"${s['ending_total']:,.0f}",
            "Years funded": s["years_funded"],
            "Fed tax": f"${s['total_federal_tax']:,.0f}",
            "Healthcare OOP": f"${s['total_healthcare_oop']:,.0f}",
            "Penalty": f"${s['total_penalty']:,.0f}",
            "Conversions total": f"${s['total_conversions']:,.0f}",
            "Shortfall": f"${s['total_shortfall']:,.0f}",
        }
        if mc_compare:
            row["Success rate"] = f"{success_rates.get(name, 0.0):.0%}"
        rows.append(row)
    st.markdown("##### Lifetime summary")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    # C. Success-rate bar chart when mc_compare is on
    if mc_compare and success_rates:
        import plotly.graph_objects as go
        bar_fig = go.Figure(go.Bar(
            x=[STRATEGY_DISPLAY.get(n, n) for n in success_rates],
            y=[v * 100 for v in success_rates.values()],
            text=[f"{v:.0%}" for v in success_rates.values()],
            textposition="auto",
        ))
        bar_fig.update_layout(
            title=f"Monte Carlo success rate by strategy ({mc_label})",
            xaxis_title="Strategy",
            yaxis_title="Success rate (%)",
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.plotly_chart(charts.compare_balance_trajectory(scenarios), use_container_width=True)

    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(charts.compare_ending_balance(scenarios), use_container_width=True)
    with cols[1]:
        st.plotly_chart(charts.compare_cumulative_costs(scenarios), use_container_width=True)
    with cols[2]:
        st.plotly_chart(charts.compare_shortfall(scenarios), use_container_width=True)


def inputs_summary_view(base: SimulationInputs) -> None:
    st.markdown("##### Echoed parameters")
    items = [
        ("Cash", f"${base.initial_cash:,.0f}"),
        ("Taxable", f"${base.initial_taxable:,.0f}"),
        ("Taxable basis", f"${base.taxable_basis:,.0f}"),
        ("Traditional", f"${base.initial_traditional:,.0f}"),
        ("Roth", f"${base.initial_roth:,.0f}"),
        ("Roth contributions", f"${base.roth_contributions:,.0f}"),
        ("HSA", f"${base.initial_hsa:,.0f}"),
        ("Total starting", f"${base.initial_cash + base.initial_taxable + base.initial_traditional + base.initial_roth + base.initial_hsa:,.0f}"),
        ("Target net spend", f"${base.target_spend:,.0f}"),
        ("Stock return", f"{base.stock_return:.2%}"),
        ("Bond return", f"{base.bond_return:.2%}"),
        ("Cash return", f"{base.cash_return:.2%}"),
        ("Stock allocation", f"{base.stock_allocation:.0%}"),
        ("Start age", base.start_age),
        ("Horizon", f"{base.horizon_years} years"),
        # F. SS fields added after Horizon
        ("SS benefit", f"${base.ss_annual_benefit:,.0f}"),
        ("SS claim age", base.ss_claim_age),
        ("ACA mode", base.aca_mode),
        ("Tax params", base.params.label),
    ]
    for label, value in items:
        st.markdown(f"- **{label}:** {value}")

    st.markdown("##### Notes & assumptions baked in")
    st.markdown(
        "- All values are **real** (inflation-adjusted) dollars; growth rate is real return.\n"
        "- Single filer. Federal brackets are 2026 estimates (2025 actuals).\n"
        "- Growth is applied at the **start** of each year, then withdrawal/conversion happens.\n"
        "- 10% early-withdrawal penalty applies to Traditional withdrawals before age 60 (proxy for 59.5).\n"
        "- HSA non-medical withdrawals are modeled as taxable income; the additional 20% pre-65 penalty is **not** modeled (HSA is last-resort here, so impact is small in well-formed plans).\n"
        "- WA state has no income tax. The 7% capital-gains tax above $262k LTCG **is** modeled (state_tax field on PlanResult; lifetime sum in summary).\n"
        "- ACA premium model uses the IRA-expanded sliding-scale schedule (or 400% FPL cliff) on a benchmark $8k/yr unsubsidized premium.\n"
        "- Social Security is taxed using the IRS provisional-income test (0/50/85% inclusion) — not the simplified 100% inclusion.\n"
        "- Medicare IRMAA surcharge uses **current-year** MAGI (IRS actually uses MAGI from 2 years prior — simplification understates volatility-year IRMAA by ~2 years)."
    )


def how_it_works_view() -> None:
    st.markdown(
        """
### The early-retirement withdrawal problem

You retire at 35 with most of your wealth in tax-deferred accounts (Traditional 401k)
that can't be touched until 59.5 without a 10% penalty. The puzzle:

1. **Bridge** the ~25-year gap with money you *can* access (taxable, cash, Roth contributions).
2. **Move** Traditional → Roth via the **Roth conversion ladder** so it's accessible later.
3. **Minimize** the combined hit from federal tax, ACA premium clawback, and early-withdrawal penalty.

These three goals fight each other in low-bracket years. The strategies below pick
different points on that tradeoff frontier.

---

### Source priority (when funding the year's spend)

Same order under every strategy — what differs is *how much to convert*. Pre-59.5:

1. **Cash** (no tax)
2. **Taxable brokerage** — sells pro-rata between basis (no tax) and gains (LTCG, possibly 0%)
3. **Seasoned Roth conversions** (rungs ≥5 years old: tax/penalty free)
4. **Roth contributions** (always free)
5. **Traditional with 10% penalty**
6. **HSA** (last resort)

Post-59.5 the order collapses: Roth (any) and Traditional both unlock fully.

---

### What each strategy does

#### `bridge_optimal` — fill the joint ceiling
For each year, convert Trad→Roth up to the **lesser of**:
- the **0% LTCG ceiling**: `standard_deduction + (48,350 − this_year's_LTCG)` — the largest
  ordinary income that still keeps all LTCG in the 0% bracket
- the **400% FPL ceiling** (~$62.6k): the cliff above which ACA subsidies vanish

Floored at the **standard deduction** (~$15.7k), since that much conversion is always free
regardless of LTCG. Best when basis ratio is high (so LTCG room is wide) and you care
about ACA subsidies.

**Trap:** when basis ratio gets thin in late **Phase A** (the bridge years before 59.5 when Traditional accounts are still penalty-locked), the LTCG ceiling collapses and only
the std-deduction floor survives — too small to feed enough $80k-sized rungs, leaving a
ladder gap from years 16–24 that forces some pre-60 penalty Trad pulls.

#### `minimal_convert` — fill the standard deduction only
Convert exactly `standard_deduction` (~$15.7k) every year. This is **always free**: the
deduction wipes out the ordinary tax on it, and it's small enough not to push LTCG into
the 15% bracket. Lowest tax + ACA over the bridge years, but creates a starved ladder —
$80k spend can't be funded from $15.7k rungs, so you eat penalty Trad pulls in the gap.

#### `aggressive_convert` — fill the top of the 12% bracket
Convert `standard_deduction + 48,475` (~$64.2k) every year. Ordinary tax rate stays at
12%, but every dollar above the 0% LTCG ceiling pushes that much LTCG up into the 15%
bracket. **Most tax-expensive in early years**, but the ladder is fully fed (each $64k
rung covers a full year of $80k spend with margin), so **zero pre-60 penalty**.

#### `custom` — your number
Set a fixed annual conversion amount. Useful for sensitivity testing or matching an
external plan.

---

### Reading the comparison view

The interesting comparison is **(federal tax + ACA premium + penalty)** vs **ending balance**:

- `minimal_convert` minimizes the visible cost (tax + ACA) but pays it back as penalty.
- `aggressive_convert` eliminates penalty entirely at the cost of higher early-year tax.
- `bridge_optimal` aims for the sweet spot but on this portfolio it's not always dominant
  — the std-ded floor in late Phase A creates a partial ladder gap.

There is no single "right" answer; the right choice depends on whether you weight near-term
cash flow (favor minimal/bridge) or terminal certainty and spending floor (favor aggressive).

---

### Math notes

When you take capital gains and do a Roth conversion in the same year, the conversion (ordinary income) and the LTCG share the same low brackets — so they fight for the same room.

- **LTCG stacking**: federal LTCG is taxed in the brackets *above* ordinary taxable income.
  Each extra dollar of conversion that pushes ordinary taxable past the 0% LTCG ceiling
  ($48,350) costs **12% on the conversion plus 15% on the displaced LTCG = 27% marginal**.
  This is the load-bearing math behind why aggressive conversion is expensive in late Phase A.
- **ACA subsidy curve**: piecewise linear contribution rate, 0% at 150% FPL up to 8.5% at
  400% FPL. Above 400% FPL: either 8.5% cap (IRA-extended, current law through 2025) or no
  subsidy (cliff). Toggle in the sidebar.
- **Roth ladder seasoning**: 5 years from year of conversion. The simulator FIFOs from
  oldest seasoned rung first.

---

### Stochastic returns (Monte Carlo)

The Monte Carlo engine models each year's real return using a lognormal distribution applied
to `(1 + real_return)` for each asset class (stocks, bonds, cash), with a stock/bond
correlation parameter. Each path runs the full withdrawal/conversion simulation; success rate
is the fraction of 1,000 paths that reach year `horizon_years` without a funding shortfall.
The fan chart shows the 10th/25th/50th/75th/90th percentile total-portfolio balance across all
paths; the fan widens over time because terminal-balance variance compounds roughly with
√(years), amplifying early return shocks all the way to the end of the horizon.

---

### Income streams

**Social Security:** The claim-age tradeoff is significant — claiming at 62 yields roughly
70% of the full retirement age (FRA) benefit, while delaying to 70 produces roughly 124%.
The simulator applies the SS benefit starting at your chosen claim age and taxes it using
the IRS provisional-income test: 0% inclusion below the first threshold, 50% inclusion in
the middle band, and 85% inclusion above the upper threshold — not the simplified 100%
assumption that overstates your tax liability.

**Required Minimum Distributions (RMDs):** Starting at age 73, the IRS requires a minimum
annual withdrawal from Traditional accounts, sized by the IRS Uniform Lifetime Table divisor
applied to the prior year-end balance. If the forced RMD exceeds the year's net spending
need, the surplus is treated as excess cash and parked in the cash account rather than wasted.

---

### Healthcare lifecycle

**Pre-65 ACA:** The premium tax credit reduces your benchmark unsubsidized premium based
on your MAGI as a percent of the federal poverty line. Under the IRA expansion (current law),
your contribution is capped at 8.5% of MAGI even above 400% FPL; under the cliff schedule,
subsidies vanish entirely above 400% FPL. The subsidy schedule is piecewise: 0% contribution
at 150% FPL, rising linearly to 8.5% at 400% FPL and above. MAGI-heavy Roth conversion years
can sharply increase your premium.

**Post-65 Medicare + IRMAA:** Base Part B + Part D premiums are added to healthcare OOP
starting at age 65. Above income thresholds, the Income-Related Monthly Adjustment Amount
(IRMAA) adds a surcharge in discrete tiers based on MAGI — the simulator uses current-year
MAGI as a simplification; IRS actually uses MAGI from two years prior, so volatility-year
IRMAA exposure may be understated by approximately two years.
"""
    )


@_fragment
def monte_carlo_view(base: SimulationInputs) -> None:
    st.markdown("#### Monte Carlo simulation")

    model_choice = st.radio(
        "Returns model",
        options=["lognormal", "historical"],
        format_func=lambda m: {
            "lognormal": "Lognormal (synthetic, configurable σ)",
            "historical": "Historical (Shiller annual real returns, rolling start)",
        }[m],
        horizontal=True,
        key="mc_model_choice",
    )

    strategy = st.selectbox(
        "Strategy",
        STRATEGY_PRESETS,
        format_func=lambda s: STRATEGY_DISPLAY.get(s, s),
        key="mc_strategy",
    )
    inputs = SimulationInputs(**{**base.__dict__, "strategy": strategy})

    if model_choice == "lognormal":
        cols = st.columns(2)
        with cols[0]:
            n_runs = st.number_input("Paths", value=1000, min_value=100, max_value=5000, step=100,
                                     help="More paths = tighter percentile estimates, longer wait.")
        with cols[1]:
            seed = st.number_input("Random seed", value=42, step=1,
                                   help="Change to resample. Same seed reproduces same paths.")

        with st.expander("Advanced: volatility parameters", expanded=False):
            adv_cols = st.columns(2)
            with adv_cols[0]:
                sigma_s = st.slider("Stock σ", 0.0, 0.30, 0.18, 0.01,
                                    help="Stdev of log-returns. ~0.18 ≈ historical US large-cap real volatility.")
            with adv_cols[1]:
                sigma_b = st.slider("Bond σ", 0.0, 0.15, 0.07, 0.005,
                                    help="Stdev of log-returns. ~0.07 ≈ historical intermediate-bond real volatility.")
            adv_cols2 = st.columns(2)
            with adv_cols2[0]:
                sigma_c = st.slider("Cash σ", 0.0, 0.05, 0.01, 0.005,
                                    help="Real cash volatility (mostly inflation surprise).")
            with adv_cols2[1]:
                rho = st.slider("Stock/bond correlation", -0.5, 0.6, 0.05, 0.05,
                                help="Annual real-return correlation. Historical US: -0.05 to +0.20.")

        mc = cached_mc_lognormal(
            inputs=inputs,
            n_runs=int(n_runs),
            seed=int(seed),
            sigma_s=float(sigma_s),
            sigma_b=float(sigma_b),
            sigma_c=float(sigma_c),
            rho=float(rho),
        )
        run_label = f"Lognormal · {int(n_runs)} paths"
    else:
        # Preview model to surface n_paths / horizon mismatch before running.
        preview = HistoricalPlayback(horizon_years=inputs.horizon_years)
        if preview.n_paths == 0:
            st.error(
                f"Horizon of {inputs.horizon_years} years exceeds the historical dataset "
                f"({preview.coverage[0]}–{preview.coverage[1]}). Reduce horizon in the sidebar."
            )
            return
        st.caption(
            f"Replays {preview.n_paths} sequences with start years "
            f"{preview.start_years[0]}–{preview.start_years[-1]} "
            f"(data: {preview.coverage[0]}–{preview.coverage[1]}, real returns from Shiller's S&P + 10y bond series; cash held at 0% real)."
        )
        mc = cached_mc_historical(inputs=inputs, start_year_floor=1928)
        run_label = f"Historical · {preview.n_paths} sequences ({preview.start_years[0]}–{preview.start_years[-1]})"

    kpis = charts.mc_success_kpis(mc)
    kpi_cols = st.columns(len(kpis))
    for col, (label, value) in zip(kpi_cols, kpis.items()):
        col.metric(label, value)

    st.caption(f"Strategy: {STRATEGY_DISPLAY.get(strategy, strategy)}  ·  {run_label}")

    if mc.success_rate < 0.80:
        st.warning(
            f"Success rate {mc.success_rate:.0%} is below 80% — sequence-of-returns risk is "
            f"meaningful for this plan. Consider lowering target spend or extending the bridge."
        )
    elif mc.success_rate < 0.95:
        st.info(
            f"Success rate {mc.success_rate:.0%} — meaningful but not robust to severe sequences. "
            f"Consider tightening spend or lengthening bridge."
        )
    else:
        st.success(f"Success rate {mc.success_rate:.0%} — plan is robust to historical-style volatility.")

    st.plotly_chart(charts.mc_fan_chart(mc), use_container_width=True)


# --- main ----------------------------------------------------------------

st.title("Early Retirement Withdrawal Planner")
st.caption("Tax + penalty optimization for early retirees. Real dollars throughout.")

if "custom_conversion" not in st.session_state:
    st.session_state["custom_conversion"] = 40_000

base = render_sidebar()

tab1, tab2, tab_mc, tab3, tab4 = st.tabs(
    ["Single scenario", "Compare strategies", "Monte Carlo", "How it works", "Inputs & assumptions"]
)

with tab1:
    single_scenario_view(base)
with tab2:
    comparison_view(base)
with tab_mc:
    monte_carlo_view(base)
with tab3:
    how_it_works_view()
with tab4:
    inputs_summary_view(base)
