"""Streamlit GUI for the Early Retirement Withdrawal Planner.

Run: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import charts
from planner.montecarlo import run_monte_carlo
from planner.returns import LognormalReturns
from planner.simulate import SimulationInputs, simulate, summarize
from planner.strategy import STRATEGY_DESCRIPTIONS, STRATEGY_PRESETS


st.set_page_config(page_title="Withdrawal Planner", layout="wide", initial_sidebar_state="expanded")


STRATEGY_DISPLAY = {
    "bridge_optimal": "Bridge optimal",
    "minimal_convert": "Minimal convert",
    "aggressive_convert": "Aggressive convert",
    "custom": "Custom",
}


@st.cache_data
def cached_simulate(inputs: SimulationInputs):
    return simulate(inputs)


@st.cache_data(show_spinner="Running Monte Carlo...")
def cached_mc(
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


# Streamlit ≥1.32 supports @st.fragment; older falls back to identity decorator.
try:
    _fragment = st.fragment
except AttributeError:
    def _fragment(f):
        return f


def render_sidebar() -> SimulationInputs:
    st.sidebar.markdown("### Initial balances (real $)")
    cash = st.sidebar.number_input("Cash", value=25_000, step=1_000)
    taxable = st.sidebar.number_input("Taxable brokerage", value=834_843, step=10_000)
    basis = st.sidebar.number_input("Taxable cost basis", value=570_659, step=10_000)
    traditional = st.sidebar.number_input("Traditional 401k / IRA", value=837_547, step=10_000)
    roth = st.sidebar.number_input("Roth IRA", value=327_610, step=10_000)
    roth_contrib = st.sidebar.number_input("...of which is direct contributions", value=0, step=1_000,
                                            help="Roth contributions can be withdrawn anytime tax/penalty free.")
    hsa = st.sidebar.number_input("HSA", value=26_000, step=1_000)

    st.sidebar.markdown("### Spending")
    target = st.sidebar.number_input("Target net annual spend (real $)", value=80_000, step=1_000)

    st.sidebar.markdown("### Real returns by asset class")
    stock_return = st.sidebar.slider("Stocks", min_value=0.0, max_value=0.12, value=0.07, step=0.005, format="%.3f",
                                     help="Long-run real (after-inflation) total return on equities.")
    bond_return = st.sidebar.slider("Bonds", min_value=-0.02, max_value=0.06, value=0.02, step=0.005, format="%.3f",
                                    help="Long-run real total return on intermediate bonds.")
    cash_return = st.sidebar.slider("Cash", min_value=-0.02, max_value=0.04, value=0.0, step=0.005, format="%.3f",
                                    help="Real return on cash (often near 0 after inflation).")
    stock_alloc = st.sidebar.slider("Stock allocation (all investment accounts)",
                                    min_value=0.0, max_value=1.0, value=0.85, step=0.05,
                                    help="Fraction of each tax-advantaged + taxable account held in stocks vs bonds. Cash sits separately.")

    st.sidebar.markdown("### Horizon")
    start_age = st.sidebar.number_input("Retirement age", value=35, step=1)
    horizon = st.sidebar.number_input("Years to simulate", value=60, step=1)

    st.sidebar.markdown("### Social Security")
    ss_benefit = st.sidebar.number_input("Annual SS benefit (real $)", value=0, step=1_000,
        help="Estimated annual benefit in today's dollars. Use ssa.gov estimator. Set 0 to disable.")
    ss_claim = st.sidebar.number_input("SS claim age", value=67, min_value=62, max_value=70, step=1,
        help="62=earliest, 67=full retirement age, 70=max delay.")

    st.sidebar.markdown("### ACA assumption")
    aca_mode = st.sidebar.radio(
        "Subsidy schedule",
        options=["cap", "cliff"],
        format_func=lambda m: {"cap": "8.5% cap (IRA-extended)", "cliff": "Cliff at 400% FPL"}[m],
        index=0,
        help=(
            "IRA-extended (cap): premium contribution capped at 8.5% of MAGI even above "
            "400% FPL — current law through 2025. Cliff: subsidies vanish entirely above "
            "400% FPL. Use 'cliff' for conservative planning if you expect the IRA "
            "expansion to lapse."
        ),
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

    scenarios = {}
    for name in chosen:
        inputs = SimulationInputs(**{
            **base.__dict__,
            "strategy": name,
            "custom_conversion": custom_amount if name == "custom" else None,
        })
        scenarios[name] = cached_simulate(inputs)

    # Summary table
    rows = []
    for name, results in scenarios.items():
        s = summarize(results)
        rows.append({
            "Strategy": STRATEGY_DISPLAY.get(name, name),
            "Ending (real $)": f"${s['ending_total']:,.0f}",
            "Years funded": s["years_funded"],
            "Fed tax": f"${s['total_federal_tax']:,.0f}",
            "Healthcare OOP": f"${s['total_healthcare_oop']:,.0f}",
            "Penalty": f"${s['total_penalty']:,.0f}",
            "Conversions total": f"${s['total_conversions']:,.0f}",
            "Shortfall": f"${s['total_shortfall']:,.0f}",
        })
    st.markdown("##### Lifetime summary")
    st.dataframe(rows, hide_index=True, use_container_width=True)

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
        "- WA state has no income tax; the 7% capital gains tax above $262k of LTCG is **not** modeled (irrelevant at this draw rate).\n"
        "- ACA premium model uses the IRA-expanded sliding-scale schedule (or 400% FPL cliff) on a benchmark $8k/yr unsubsidized premium."
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
"""
    )


@_fragment
def monte_carlo_view(base: SimulationInputs) -> None:
    st.markdown("#### Monte Carlo simulation")
    st.caption(
        "Stochastic returns (lognormal). Means use the per-asset returns from the sidebar; "
        "volatility (σ) is configurable here. Future: plug in historical-data playback "
        "(Shiller resampling) or richer regime models via the `ReturnsModel` protocol."
    )

    cols = st.columns(4)
    with cols[0]:
        n_runs = st.number_input("Paths", value=1000, min_value=100, max_value=5000, step=100,
                                 help="More paths = tighter percentile estimates, longer wait.")
    with cols[1]:
        seed = st.number_input("Random seed", value=42, step=1,
                               help="Change to resample. Same seed reproduces same paths.")
    with cols[2]:
        sigma_s = st.slider("Stock σ", 0.0, 0.30, 0.18, 0.01,
                            help="Stdev of log-returns. ~0.18 ≈ historical US large-cap real volatility.")
    with cols[3]:
        sigma_b = st.slider("Bond σ", 0.0, 0.15, 0.07, 0.005,
                            help="Stdev of log-returns. ~0.07 ≈ historical intermediate-bond real volatility.")

    cols2 = st.columns(2)
    with cols2[0]:
        sigma_c = st.slider("Cash σ", 0.0, 0.05, 0.01, 0.005,
                            help="Real cash volatility (mostly inflation surprise).")
    with cols2[1]:
        rho = st.slider("Stock/bond correlation", -0.5, 0.6, 0.05, 0.05,
                        help="Annual real-return correlation. Historical US: -0.05 to +0.20.")

    mc = cached_mc(
        inputs=base,
        n_runs=int(n_runs),
        seed=int(seed),
        sigma_s=float(sigma_s),
        sigma_b=float(sigma_b),
        sigma_c=float(sigma_c),
        rho=float(rho),
    )

    kpis = charts.mc_success_kpis(mc)
    kpi_cols = st.columns(len(kpis))
    for col, (label, value) in zip(kpi_cols, kpis.items()):
        col.metric(label, value)

    if mc.success_rate < 0.80:
        st.warning(
            f"Success rate {mc.success_rate:.0%} is below 80% — sequence-of-returns risk is "
            f"meaningful for this plan. Consider lowering target spend or extending the bridge."
        )
    elif mc.success_rate >= 0.95:
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
