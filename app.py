"""Streamlit GUI for the Early Retirement Withdrawal Planner.

Run: streamlit run app.py
"""

from __future__ import annotations

import json as _json
import dataclasses as _dc

import pandas as pd
import streamlit as st

import charts
from planner.montecarlo import run_monte_carlo
from planner.returns import HistoricalPlayback, LognormalReturns
from planner.simulate import SimulationInputs, simulate, summarize
from planner.state_tax import STATE_PRESETS
from planner.strategy import STRATEGY_DESCRIPTIONS, STRATEGY_PRESETS
from planner.streams import ExpenseStream, IncomeStream


st.set_page_config(page_title="Withdrawal Planner", layout="wide", initial_sidebar_state="expanded")


STRATEGY_DISPLAY = {
    "bridge_optimal": "Bridge optimal",
    "bridge_guarded": "Bridge guarded",
    "bridge_responsive": "Bridge responsive",
    "minimal_convert": "Minimal convert",
    "aggressive_convert": "Aggressive convert",
    "custom": "Custom",
}

# Defaults for sidebar inputs. Loader prefers scenarios/default.local.json (gitignored —
# put your personal scenario here) and falls back to scenarios/default.json (committed,
# generic median-FIRE persona). Editing either JSON is the supported way to change startup
# defaults; the in-app sidebar download/upload buttons let you save/swap others ad hoc.
_DEFAULT_SCENARIO_FILES = ["scenarios/default.local.json", "scenarios/default.json"]
_SIDEBAR_FALLBACK_DEFAULTS = {
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
    "filing_status": "single",
    "income_streams": [],
    "expense_streams": [],
    "state_code": "WA",
    "state_ordinary_rate": 0.0,
    "state_ltcg_rate": 0.0,
    "state_ltcg_threshold": 0.0,
    "spend_mode": "fixed",
    "spend_rate": 0.035,
    "current_age": 35,
    "annual_savings": 0,
    "savings_allocation_cash_pct": 0,
    "savings_allocation_taxable_pct": 0,
    "savings_allocation_traditional_pct": 0,
    "savings_allocation_roth_pct": 0,
    "savings_allocation_hsa_pct": 0,
    "employer_match_pct": 0.0,
    "employer_match_cap_pct": 0.0,
    "accumulation_wage_income": 0,
    "retirement_mode": "fixed",
    "retirement_target_nw": 0,
    "retirement_age_floor": None,
}


def _load_default_scenario() -> dict:
    import os
    path = next((p for p in _DEFAULT_SCENARIO_FILES if os.path.exists(p)), None)
    if path is None:
        return dict(_SIDEBAR_FALLBACK_DEFAULTS)
    with open(path) as f:
        loaded = _json.load(f)
    # Strip non-sidebar fields (strategy / custom_conversion are per-tab choices).
    for k in ("strategy", "custom_conversion"):
        loaded.pop(k, None)
    # Coerce numeric types stored as floats back to ints where the widget expects ints.
    for k in ("initial_cash", "initial_taxable", "taxable_basis", "initial_traditional",
              "initial_roth", "roth_contributions", "initial_hsa", "target_spend",
              "start_age", "horizon_years", "ss_annual_benefit", "ss_claim_age",
              "current_age", "annual_savings", "accumulation_wage_income"):
        if k in loaded:
            loaded[k] = int(loaded[k])
    # retirement_age_floor is nullable int — keep None as-is, coerce numeric to int.
    if "retirement_age_floor" in loaded and loaded["retirement_age_floor"] is not None:
        loaded["retirement_age_floor"] = int(loaded["retirement_age_floor"])
    # Convert savings_allocation list-of-pairs to per-account pct keys for sidebar widgets.
    if "savings_allocation" in loaded:
        alloc_map = {name: frac for name, frac in (loaded.pop("savings_allocation") or [])}
        loaded["savings_allocation_cash_pct"] = int(round(alloc_map.get("cash", 0.0) * 100))
        loaded["savings_allocation_taxable_pct"] = int(round(alloc_map.get("taxable", 0.0) * 100))
        loaded["savings_allocation_traditional_pct"] = int(round(alloc_map.get("traditional", 0.0) * 100))
        loaded["savings_allocation_roth_pct"] = int(round(alloc_map.get("roth", 0.0) * 100))
        loaded["savings_allocation_hsa_pct"] = int(round(alloc_map.get("hsa", 0.0) * 100))
    return {**_SIDEBAR_FALLBACK_DEFAULTS, **loaded}


_SIDEBAR_DEFAULTS = _load_default_scenario()


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


_INCOME_COLS = ["name", "annual_amount", "start_age", "end_age", "taxable"]
_EXPENSE_COLS = ["name", "annual_amount", "start_age", "end_age"]


def _rows_to_income_streams(rows) -> tuple:
    out = []
    for r in rows or []:
        name = r.get("name")
        amt = r.get("annual_amount")
        if not name or amt is None or (isinstance(amt, float) and pd.isna(amt)):
            continue
        out.append(IncomeStream(
            name=str(name),
            annual_amount=float(amt),
            start_age=int(r.get("start_age") or 0),
            end_age=int(r.get("end_age") or 120),
            taxable=bool(r.get("taxable", True)),
        ))
    return tuple(out)


def _rows_to_expense_streams(rows) -> tuple:
    out = []
    for r in rows or []:
        name = r.get("name")
        amt = r.get("annual_amount")
        if not name or amt is None or (isinstance(amt, float) and pd.isna(amt)):
            continue
        out.append(ExpenseStream(
            name=str(name),
            annual_amount=float(amt),
            start_age=int(r.get("start_age") or 0),
            end_age=int(r.get("end_age") or 120),
        ))
    return tuple(out)


def _pick_mc_path(mc, year_label_fn=None, key_prefix: str = "") -> int:
    """Render a Worst/Median/Best/By-index radio. Returns the chosen path_index.

    `mc` is the MCSummary; `year_label_fn(idx)` returns extra caption text (e.g. start year);
    `key_prefix` namespaces the widget keys so two pickers can coexist on the same page.
    """
    options = ["Worst (most shortfall)", "Median ending", "Best ending", "By index"]
    choice = st.radio("Pick path", options, horizontal=True, key=f"{key_prefix}path_choice")
    if choice.startswith("Worst"):
        idx = max(range(mc.n_runs), key=lambda i: mc.paths[i].total_shortfall)
    elif choice.startswith("Median"):
        sorted_pi = sorted(mc.paths, key=lambda p: p.ending_total)
        idx = sorted_pi[len(sorted_pi) // 2].path_index
    elif choice.startswith("Best"):
        idx = max(range(mc.n_runs), key=lambda i: mc.paths[i].ending_total)
    else:
        idx = st.number_input(
            "Path index", min_value=0, max_value=mc.n_runs - 1, value=0, step=1,
            key=f"{key_prefix}path_idx",
        )
        idx = int(idx)
    extra = year_label_fn(idx) if year_label_fn else ""
    p = mc.paths[idx]
    st.caption(
        f"Path {idx}{extra}  ·  ending ${p.ending_total:,.0f}  ·  "
        f"shortfall ${p.total_shortfall:,.0f}  ·  years funded {p.years_funded}"
    )
    return idx


def render_sidebar() -> SimulationInputs:
    # Initialize session state defaults so widgets can use key= only (no value=)
    for k, v in _SIDEBAR_DEFAULTS.items():
        st.session_state.setdefault(f"scn_{k}", v)
    st.session_state.setdefault("scn_optimization_target", "depletion")

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
    spend_mode = st.sidebar.radio(
        "Mode",
        options=["fixed", "swr"],
        format_func=lambda m: {"fixed": "Fixed annual amount", "swr": "% of starting portfolio (SWR)"}[m],
        key="scn_spend_mode",
        horizontal=False,
        help="Fixed: enter a real-$ annual spend. SWR: enter a % and the engine resolves to dollars × starting portfolio.",
    )
    if spend_mode == "fixed":
        target = st.sidebar.number_input(
            "Target net annual spend (real $)", step=1_000, key="scn_target_spend",
        )
        spend_rate = st.session_state.get("scn_spend_rate", 0.035)
    else:
        spend_rate = st.sidebar.slider(
            "SWR (% of starting portfolio)", min_value=0.020, max_value=0.080, step=0.001,
            format="%.3f",
            key="scn_spend_rate",
            help="Safe Withdrawal Rate as a fraction. 0.04 = 4% rule, 0.035 = conservative, 0.030 = lean-FIRE.",
        )
        # Live $ estimate based on the balances entered above.
        _starting_estimate = float(cash) + float(taxable) + float(traditional) + float(roth) + float(hsa)
        st.sidebar.caption(f"≈ ${_starting_estimate * spend_rate:,.0f}/yr at current portfolio")
        target = st.session_state.get("scn_target_spend", 80_000)

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
    start_age = st.sidebar.number_input(
        "Retirement age (ceiling in target-NW mode)", step=1, key="scn_start_age",
        help="Fixed mode: retirement begins at this age. Target-NW mode: retirement begins when net worth hits the trigger, using this as the upper-bound ceiling.",
    )
    horizon = st.sidebar.number_input("Years to simulate", step=1, key="scn_horizon_years")

    # Accumulation phase inputs
    with st.sidebar.expander("Accumulation", expanded=False):
        current_age = st.number_input(
            "Current age", min_value=18, max_value=80,
            key="scn_current_age",
            help="Your age today. Set equal to retirement age to skip the accumulation phase.",
        )
        annual_savings = st.number_input(
            "Annual savings (real $)", min_value=0, step=1_000,
            key="scn_annual_savings",
        )
        st.caption("Allocate savings across accounts (%):")
        _alloc_cols = st.columns(5)
        with _alloc_cols[0]:
            alloc_cash_pct = st.number_input("Cash", min_value=0, max_value=100, step=1,
                                             key="scn_savings_allocation_cash_pct")
        with _alloc_cols[1]:
            alloc_taxable_pct = st.number_input("Taxable", min_value=0, max_value=100, step=1,
                                                key="scn_savings_allocation_taxable_pct")
        with _alloc_cols[2]:
            alloc_trad_pct = st.number_input("Trad", min_value=0, max_value=100, step=1,
                                             key="scn_savings_allocation_traditional_pct")
        with _alloc_cols[3]:
            alloc_roth_pct = st.number_input("Roth", min_value=0, max_value=100, step=1,
                                             key="scn_savings_allocation_roth_pct")
        with _alloc_cols[4]:
            alloc_hsa_pct = st.number_input("HSA", min_value=0, max_value=100, step=1,
                                            key="scn_savings_allocation_hsa_pct")
        _alloc_total = alloc_cash_pct + alloc_taxable_pct + alloc_trad_pct + alloc_roth_pct + alloc_hsa_pct
        if _alloc_total > 100:
            st.caption(f"Allocation sum: {_alloc_total}% ⚠️ exceeds 100 (engine normalizes excess to cash)")
        else:
            st.caption(f"Allocation sum: {_alloc_total}%")
        employer_match_pct_ui = st.number_input(
            "Employer match (% of wages)", min_value=0.0, max_value=50.0, step=0.1,
            value=float(st.session_state.get("scn_employer_match_pct", 0.0)) * 100,
            key="scn_employer_match_pct_ui",
            help="Employer contribution as a percent of your wage income.",
        )
        employer_match_cap_pct_ui = st.number_input(
            "Employer match cap (% of wages)", min_value=0.0, max_value=50.0, step=0.1,
            value=float(st.session_state.get("scn_employer_match_cap_pct", 0.0)) * 100,
            key="scn_employer_match_cap_pct_ui",
            help="Max wages matched. 0 = uncapped.",
        )
        st.caption("0 = uncapped match")
        accumulation_wage_income = st.number_input(
            "Accumulation wage income (real $)", min_value=0, step=1_000,
            key="scn_accumulation_wage_income",
        )

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

    # Scheduled income & expense streams
    with st.sidebar.expander("Scheduled streams (income / expense)", expanded=False):
        st.caption(
            "SS-like recurring cash flows. Income reduces required gross withdrawal; "
            "taxable income also lifts ordinary-income tax. Expense raises required "
            "gross need, no tax effect. Age window inclusive."
        )

        st.markdown("**Income streams** (rentals, pensions, planned sales, etc.)")
        inc_seed = st.session_state.get("scn_income_streams", [])
        inc_df = pd.DataFrame(inc_seed, columns=_INCOME_COLS)
        inc_edited = st.data_editor(
            inc_df,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "annual_amount": st.column_config.NumberColumn("Annual $ (real)", format="$%.0f", step=1_000),
                "start_age": st.column_config.NumberColumn("Start age", min_value=0, max_value=120, step=1),
                "end_age": st.column_config.NumberColumn("End age", min_value=0, max_value=120, step=1),
                "taxable": st.column_config.CheckboxColumn("Taxable?", default=True,
                    help="Adds to federal ordinary income (e.g. rental, pension). Uncheck for non-taxable (gift, muni interest)."),
            },
            key="income_editor",
        )
        st.session_state["scn_income_streams"] = inc_edited.to_dict("records")

        st.markdown("**Expense streams** (property tax/maintenance, tuition, LTC, etc.)")
        exp_seed = st.session_state.get("scn_expense_streams", [])
        exp_df = pd.DataFrame(exp_seed, columns=_EXPENSE_COLS)
        exp_edited = st.data_editor(
            exp_df,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "annual_amount": st.column_config.NumberColumn("Annual $ (real)", format="$%.0f", step=1_000),
                "start_age": st.column_config.NumberColumn("Start age", min_value=0, max_value=120, step=1),
                "end_age": st.column_config.NumberColumn("End age", min_value=0, max_value=120, step=1),
            },
            key="expense_editor",
        )
        st.session_state["scn_expense_streams"] = exp_edited.to_dict("records")

    income_streams = _rows_to_income_streams(st.session_state.get("scn_income_streams", []))
    expense_streams = _rows_to_expense_streams(st.session_state.get("scn_expense_streams", []))

    st.sidebar.markdown("### State & ACA")
    filing_status = st.sidebar.radio(
        "Filing status",
        options=["single", "mfj"],
        format_func=lambda s: {"single": "Single", "mfj": "Married filing jointly"}[s],
        key="scn_filing_status",
        help=(
            "MFJ uses 2026 MFJ federal brackets, doubled standard deduction (~$31.5k), "
            "household-of-2 FPL for ACA, MFJ Social Security thresholds (32k/44k), and "
            "two Medicare premiums post-65. SS benefit field is interpreted as combined "
            "household amount; for split-claim spouses, add a second SS as an income stream."
        ),
    )
    state_code = st.sidebar.selectbox(
        "State",
        options=list(STATE_PRESETS.keys()),
        format_func=lambda c: STATE_PRESETS[c].name,
        key="scn_state_code",
        help="Pluggable state-tax model. Pre-configured: 9 no-tax states + WA's 7%/$262k cap-gains rule. CUSTOM: enter your own flat rates.",
    )
    if state_code == "CUSTOM":
        st.sidebar.caption(
            "Flat marginal-rate approximation. Ignores brackets, SS exemption, "
            "and pension exclusions — for ballpark planning only."
        )
        state_ord_rate = st.sidebar.number_input(
            "State ordinary-income rate", min_value=0.0, max_value=0.15, step=0.005,
            format="%.3f", key="scn_state_ordinary_rate",
            help="Flat marginal rate on ordinary income. Use top-bracket rate as conservative proxy.",
        )
        state_cg_rate = st.sidebar.number_input(
            "State LTCG rate", min_value=0.0, max_value=0.15, step=0.005,
            format="%.3f", key="scn_state_ltcg_rate",
            help="Flat rate on long-term capital gains. Many states tax LTCG as ordinary income — use the same rate.",
        )
        state_cg_threshold = st.sidebar.number_input(
            "State LTCG threshold", min_value=0, step=10_000,
            key="scn_state_ltcg_threshold",
            help="Annual LTCG below this amount is exempt. Set 0 for none.",
        )
    else:
        state_ord_rate = float(st.session_state.get("scn_state_ordinary_rate", 0.0))
        state_cg_rate = float(st.session_state.get("scn_state_ltcg_rate", 0.0))
        state_cg_threshold = float(st.session_state.get("scn_state_ltcg_threshold", 0.0))
        if STATE_PRESETS[state_code].note:
            st.sidebar.caption(STATE_PRESETS[state_code].note)

    aca_mode = st.sidebar.radio(
        "ACA subsidy schedule",
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

    st.sidebar.markdown("### Optimization target")
    st.sidebar.radio(
        "Optimization target",
        options=["depletion", "preservation"],
        format_func=lambda s: {"depletion": "Depletion (success rate)",
                                "preservation": "Preservation (real value sustained)"}[s],
        key="scn_optimization_target",
        help="Depletion: % of paths that don't run out of money. "
             "Preservation: % of paths whose ending real balance ≥ starting. "
             "Use preservation for over-capitalized portfolios where success rate is "
             "saturated near 100%.",
    )

    # Retirement trigger
    with st.sidebar.expander("Retirement trigger", expanded=False):
        retirement_mode = st.radio(
            "Mode",
            options=["fixed", "target_nw"],
            format_func=lambda m: {"fixed": "Fixed age", "target_nw": "Target net worth"}[m],
            key="scn_retirement_mode",
            help="Fixed: retire at the retirement age above. Target NW: retire once net worth hits the trigger (up to the retirement age ceiling).",
        )
        if retirement_mode == "target_nw":
            retirement_target_nw = st.number_input(
                "Target net worth (real $)", min_value=0, step=10_000,
                key="scn_retirement_target_nw",
                help="Retire when real portfolio value reaches this amount (subject to floor and ceiling).",
            )

            def _set_target_from_spend() -> None:
                spend = st.session_state.get("scn_target_spend", 0)
                st.session_state["scn_retirement_target_nw"] = int(float(spend) * 25)

            st.button(
                "Set target = spend × 25",
                on_click=_set_target_from_spend,
                help="Standard FI heuristic: 25× annual spend covers ~4% SWR indefinitely.",
            )
            use_floor = st.checkbox(
                "Set floor age",
                value=st.session_state.get("scn_retirement_age_floor") is not None,
                help="Lock in a minimum age before which retirement cannot trigger, even if net worth is hit.",
            )
            if use_floor:
                retirement_age_floor = int(st.number_input(
                    "Floor age (earliest retirement)", min_value=18, max_value=int(start_age),
                    key="scn_retirement_age_floor_input",
                    value=st.session_state.get("scn_retirement_age_floor") or int(current_age),
                ))
            else:
                retirement_age_floor = None
        else:
            retirement_target_nw = float(st.session_state.get("scn_retirement_target_nw", 0))
            retirement_age_floor = None

    # Build savings_allocation tuple from the five pct inputs.
    _retirement_age = int(start_age)
    _current_age = int(current_age)
    _savings_alloc = tuple(
        (name, pct / 100)
        for name, pct in [
            ("cash", alloc_cash_pct),
            ("taxable", alloc_taxable_pct),
            ("traditional", alloc_trad_pct),
            ("roth", alloc_roth_pct),
            ("hsa", alloc_hsa_pct),
        ]
        if pct > 0
    )

    # G. Download button for scenario save
    built_inputs = SimulationInputs(
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
        start_age=_retirement_age,
        horizon_years=int(horizon),
        aca_mode=aca_mode,
        filing_status=filing_status,
        ss_annual_benefit=float(ss_benefit),
        ss_claim_age=int(ss_claim),
        income_streams=income_streams,
        expense_streams=expense_streams,
        state_code=state_code,
        state_ordinary_rate=state_ord_rate,
        state_ltcg_rate=state_cg_rate,
        state_ltcg_threshold=state_cg_threshold,
        spend_mode=spend_mode,
        spend_rate=float(spend_rate),
        current_age=_current_age,
        retirement_age=_retirement_age,
        annual_savings=float(annual_savings),
        savings_allocation=_savings_alloc,
        employer_match_pct=employer_match_pct_ui / 100,
        employer_match_cap_pct=employer_match_cap_pct_ui / 100,
        accumulation_wage_income=float(accumulation_wage_income),
        retirement_mode=retirement_mode,
        retirement_target_nw=float(retirement_target_nw),
        retirement_age_floor=retirement_age_floor,
    )
    # Build a JSON-serializable dict; savings_allocation becomes a list of [name, fraction] pairs.
    inputs_dict = {k: v for k, v in _dc.asdict(built_inputs).items() if k != "params"}
    inputs_dict["savings_allocation"] = [
        [name, frac] for name, frac in built_inputs.savings_allocation if frac > 0
    ]
    st.sidebar.download_button(
        "Save scenario JSON",
        data=_json.dumps(inputs_dict, indent=2),
        file_name="scenario.json",
        mime="application/json",
    )

    return built_inputs


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


def _scenario_heading(base: SimulationInputs) -> None:
    """One-line context header: starting NW + resolved SWR. Reused by single + Strategies tabs."""
    starting_nw = (base.initial_cash + base.initial_taxable + base.initial_traditional
                   + base.initial_roth + base.initial_hsa)
    if base.spend_mode == "swr":
        eff_spend = starting_nw * base.spend_rate
    else:
        eff_spend = base.target_spend
    swr = (eff_spend / starting_nw) if starting_nw > 0 else 0.0
    st.markdown(
        f"### Scenario: Starting Net Worth ${starting_nw:,.0f}  ·  "
        f"SWR {swr:.2%}  ·  Spend ${eff_spend:,.0f}/yr"
    )


@_fragment
def single_scenario_view(base: SimulationInputs) -> None:
    _scenario_heading(base)
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

    if base.retirement_mode == "target_nw":
        actual_ret_age = summary.get("actual_retirement_age")
        if actual_ret_age is not None:
            st.info(f"Retirement triggered at age **{actual_ret_age}** (target NW reached).")
        else:
            st.info("Target NW not reached within horizon — retirement age ceiling applied.")

    st.plotly_chart(charts.balance_stack(results), use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(charts.cashflow_bars(results), use_container_width=True)
    with cols[1]:
        st.plotly_chart(charts.tax_breakdown(results), use_container_width=True)

    st.plotly_chart(charts.conversions_bars(results), use_container_width=True)

    if any(r.plan.scheduled_income or r.plan.scheduled_expense for r in results):
        st.plotly_chart(charts.scheduled_streams_bars(results), use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(charts.withdrawal_rate(results), use_container_width=True)
    with cols[1]:
        st.plotly_chart(charts.ladder_status(results), use_container_width=True)

    with st.expander("Per-year detail (table)"):
        df = charts.per_year_table(results)
        st.dataframe(df, hide_index=True, use_container_width=True)


@_fragment
def strategies_view(base: SimulationInputs) -> None:
    _scenario_heading(base)

    # Strategy multiselect
    chosen = st.multiselect(
        "Strategies",
        list(STRATEGY_PRESETS),
        default=["bridge_optimal", "bridge_guarded", "minimal_convert", "aggressive_convert"],
    )
    if "custom" in chosen:
        st.number_input("Custom annual conversion", step=1_000, key="custom_conversion")
    custom_amount = st.session_state["custom_conversion"]

    if len(chosen) < 1:
        st.info("Select at least 1 strategy.")
        return

    # Returns model radio
    model_choice = st.radio(
        "Returns model",
        options=["deterministic", "lognormal", "historical"],
        format_func=lambda m: {
            "deterministic": "Deterministic (constant returns from sidebar)",
            "lognormal": "Lognormal MC (synthetic σ)",
            "historical": "Historical (Shiller annual real returns)",
        }[m],
        horizontal=True,
        key="strat_model_choice",
    )

    # Conditional model inputs
    n_runs = 1000
    seed = 42
    sigma_s, sigma_b, sigma_c, rho = 0.18, 0.07, 0.01, 0.05
    mc_results: dict = {}

    if model_choice == "lognormal":
        cols = st.columns(2)
        with cols[0]:
            n_runs = int(st.number_input("Paths", value=1000, min_value=100, max_value=5000, step=100))
        with cols[1]:
            seed = int(st.number_input("Random seed", value=42, step=1))
        with st.expander("Advanced: volatility parameters", expanded=False):
            adv_cols = st.columns(2)
            with adv_cols[0]:
                sigma_s = st.slider("Stock σ", 0.0, 0.30, 0.18, 0.01)
            with adv_cols[1]:
                sigma_b = st.slider("Bond σ", 0.0, 0.15, 0.07, 0.005)
            adv_cols2 = st.columns(2)
            with adv_cols2[0]:
                sigma_c = st.slider("Cash σ", 0.0, 0.05, 0.01, 0.005)
            with adv_cols2[1]:
                rho = st.slider("Stock/bond correlation", -0.5, 0.6, 0.05, 0.05)

    elif model_choice == "historical":
        preview = HistoricalPlayback(horizon_years=base.horizon_years)
        if preview.n_paths == 0:
            st.error(
                f"Horizon of {base.horizon_years} years exceeds the historical dataset "
                f"({preview.coverage[0]}–{preview.coverage[1]}). Switch to Lognormal or reduce horizon."
            )
            return
        st.caption(
            f"Replays {preview.n_paths} sequences with start years "
            f"{preview.start_years[0]}–{preview.start_years[-1]} "
            f"(data: {preview.coverage[0]}–{preview.coverage[1]}, real returns from Shiller's S&P + 10y bond series; cash held at 0% real)."
        )

    # Build per-strategy SimulationInputs once.
    strategy_inputs = {
        name: SimulationInputs(**{
            **base.__dict__,
            "strategy": name,
            "custom_conversion": custom_amount if name == "custom" else None,
        })
        for name in chosen
    }

    # Compute MC results when requested
    if model_choice == "lognormal":
        for name, inp in strategy_inputs.items():
            mc_results[name] = cached_mc_lognormal(
                inputs=inp, n_runs=n_runs, seed=seed,
                sigma_s=sigma_s, sigma_b=sigma_b, sigma_c=sigma_c, rho=rho,
            )
    elif model_choice == "historical":
        for name, inp in strategy_inputs.items():
            mc_results[name] = cached_mc_historical(inputs=inp, start_year_floor=1928)

    mc_enabled = model_choice in ("lognormal", "historical")

    # `scenarios` drives the summary table, comparison charts, and per-strategy
    # account-composition stacks. Under MC, use the median-by-ending-balance
    # path so all three views agree on a single representative trajectory.
    # In deterministic mode, use the constant-returns single run.
    scenarios = {}
    for name, inp in strategy_inputs.items():
        if mc_enabled:
            scenarios[name] = mc_results[name].median_path
        else:
            scenarios[name] = cached_simulate(inp)
    optimization_target = st.session_state.get("scn_optimization_target", "depletion")

    # Lifetime summary table
    rows = []
    for name, results in scenarios.items():
        s = summarize(results)
        row = {
            "Strategy": STRATEGY_DISPLAY.get(name, name),
            "Ending (real $)": f"${s['ending_total']:,.0f}",
            "Years funded": s["years_funded"],
            "Fed tax": f"${s['total_federal_tax']:,.0f}",
            "State tax": f"${s.get('total_state_tax', 0):,.0f}",
            "Healthcare OOP": f"${s['total_healthcare_oop']:,.0f}",
            "Penalty": f"${s['total_penalty']:,.0f}",
            "Conversions total": f"${s['total_conversions']:,.0f}",
            "Shortfall": f"${s['total_shortfall']:,.0f}",
        }
        if mc_enabled:
            mc = mc_results.get(name)
            if optimization_target == "preservation":
                row["Preservation rate"] = f"{mc.preservation_rate:.0%}" if mc else "—"
            else:
                row["Success rate"] = f"{mc.success_rate:.0%}" if mc else "—"
            any_target_nw = any(
                strategy_inputs[n].retirement_mode == "target_nw" for n in chosen
            )
            if any_target_nw:
                inp = strategy_inputs.get(name)
                if inp and inp.retirement_mode == "target_nw" and mc:
                    row["Median retire age"] = str(mc.median_retirement_age) if mc.median_retirement_age is not None else "—"
                    row["Target hit %"] = f"{mc.target_hit_rate:.0%}"
                else:
                    row["Median retire age"] = "—"
                    row["Target hit %"] = "—"
        rows.append(row)
    st.markdown("##### Lifetime summary")
    if mc_enabled:
        st.caption(
            "Per-strategy values and trajectories below reflect the **median Monte Carlo path** "
            "(the path whose ending balance is the median across runs), not a deterministic "
            "constant-returns run. Switch to *Deterministic* mode for the constant-returns view."
        )
        if optimization_target == "preservation":
            st.markdown("*Optimization target: **Preservation** (% of paths ending ≥ starting real balance)*")
        else:
            st.markdown("*Optimization target: **Depletion** (% of paths that don't run out of money)*")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    # Comparison charts (suppressed when exactly 1 strategy)
    if len(chosen) > 1:
        st.plotly_chart(charts.compare_balance_trajectory(scenarios), use_container_width=True)
        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(charts.compare_ending_balance(scenarios), use_container_width=True)
        with cols[1]:
            st.plotly_chart(charts.compare_cumulative_costs(scenarios), use_container_width=True)
        with cols[2]:
            st.plotly_chart(charts.compare_shortfall(scenarios), use_container_width=True)

    # Per-strategy account composition (always)
    st.markdown("##### Account composition over time, per strategy")
    for name, results in scenarios.items():
        fig = charts.balance_stack(results)
        fig.update_layout(title=dict(text=f"{STRATEGY_DISPLAY.get(name, name)}: account balances"))
        st.plotly_chart(fig, use_container_width=True)

    # MC-only outputs
    if mc_enabled and mc_results:
        # Per-strategy metric callouts
        for name, mc in mc_results.items():
            display = STRATEGY_DISPLAY.get(name, name)
            if optimization_target == "preservation":
                rate = mc.preservation_rate
                label = "Preservation rate"
                low_msg = (f"**{display}**: {label} {rate:.0%} is below 80% — "
                           f"most paths fail to sustain real portfolio value. Consider lower spend or a more conservative strategy.")
                mid_msg = f"**{display}**: {label} {rate:.0%} — meaningful but not robust to severe sequences."
                high_msg = f"**{display}**: {label} {rate:.0%} — robust to historical-style volatility."
            else:
                rate = mc.success_rate
                label = "Success rate"
                low_msg = (f"**{display}**: {label} {rate:.0%} is below 80% — "
                           f"sequence-of-returns risk is meaningful. Consider lowering spend or extending the bridge.")
                mid_msg = f"**{display}**: {label} {rate:.0%} — meaningful but not robust to severe sequences."
                high_msg = f"**{display}**: {label} {rate:.0%} — robust to historical-style volatility."
            if rate < 0.80:
                st.warning(low_msg)
            elif rate < 0.95:
                st.info(mid_msg)
            else:
                st.success(high_msg)

        # Per-strategy fan charts
        st.markdown("##### Monte Carlo fan charts")
        for name, mc in mc_results.items():
            display = STRATEGY_DISPLAY.get(name, name)
            fig = charts.mc_fan_chart(mc)
            fig.update_layout(title=dict(text=f"{display}: MC fan chart"))
            st.plotly_chart(fig, use_container_width=True)

        # Retirement age histograms (target_nw mode only)
        target_nw_strategies = [n for n in chosen if strategy_inputs[n].retirement_mode == "target_nw"]
        if target_nw_strategies:
            st.markdown("##### Retirement age distribution (target net worth mode)")
            for name in target_nw_strategies:
                mc = mc_results[name]
                display = STRATEGY_DISPLAY.get(name, name)
                fig = charts.retirement_age_histogram(mc)
                fig.update_layout(title=dict(text=f"{display}: retirement age distribution"))
                st.plotly_chart(fig, use_container_width=True)

        # Path inspector (uses first chosen strategy as reference)
        st.markdown("##### Inspect a single Monte Carlo path under each strategy")
        ref_name = chosen[0]
        ref_mc = mc_results[ref_name]

        if model_choice == "lognormal":
            path_model = LognormalReturns(
                mu_stocks=base.stock_return, sigma_stocks=sigma_s,
                mu_bonds=base.bond_return, sigma_bonds=sigma_b,
                mu_cash=base.cash_return, sigma_cash=sigma_c,
                seed=seed, stock_bond_correlation=rho,
            )
            year_label_fn = None
        else:
            path_model = HistoricalPlayback(horizon_years=base.horizon_years)
            year_label_fn = lambda i: f" (start year {path_model.start_years[i]})"

        idx = _pick_mc_path(ref_mc, year_label_fn, key_prefix="strat_")
        for name in chosen:
            inp_strat = SimulationInputs(**{
                **base.__dict__,
                "strategy": name,
                "custom_conversion": custom_amount if name == "custom" else None,
            })
            single = simulate(inp_strat, returns_model=path_model, path_index=idx)
            fig = charts.balance_stack(single)
            short_total = sum(r.plan.shortfall for r in single)
            display = STRATEGY_DISPLAY.get(name, name)
            fig.update_layout(
                title=dict(text=f"{display}: account balances on this path"
                                + (f" — shortfall ${short_total:,.0f}" if short_total > 0 else ""))
            )
            st.plotly_chart(fig, use_container_width=True)


def inputs_summary_view(base: SimulationInputs) -> None:
    st.markdown("##### Echoed parameters")
    starting_nw = (base.initial_cash + base.initial_taxable + base.initial_traditional
                   + base.initial_roth + base.initial_hsa)
    if base.spend_mode == "swr":
        spend_label = f"{base.spend_rate:.2%} of starting (≈ ${starting_nw * base.spend_rate:,.0f})"
    else:
        spend_label = f"${base.target_spend:,.0f} (fixed)"
    state_label = STATE_PRESETS.get(base.state_code, STATE_PRESETS["NONE"]).name
    if base.state_code == "CUSTOM":
        state_label = (f"Custom — ord {base.state_ordinary_rate:.2%}, "
                       f"LTCG {base.state_ltcg_rate:.2%} above ${base.state_ltcg_threshold:,.0f}")
    items = [
        ("Cash", f"${base.initial_cash:,.0f}"),
        ("Taxable", f"${base.initial_taxable:,.0f}"),
        ("Taxable basis", f"${base.taxable_basis:,.0f}"),
        ("Traditional", f"${base.initial_traditional:,.0f}"),
        ("Roth", f"${base.initial_roth:,.0f}"),
        ("Roth contributions", f"${base.roth_contributions:,.0f}"),
        ("HSA", f"${base.initial_hsa:,.0f}"),
        ("Total starting", f"${starting_nw:,.0f}"),
        ("Spending", spend_label),
        ("Stock return", f"{base.stock_return:.2%}"),
        ("Bond return", f"{base.bond_return:.2%}"),
        ("Cash return", f"{base.cash_return:.2%}"),
        ("Stock allocation", f"{base.stock_allocation:.0%}"),
        ("Start age", base.start_age),
        ("Horizon", f"{base.horizon_years} years"),
        ("SS benefit", f"${base.ss_annual_benefit:,.0f}"),
        ("SS claim age", base.ss_claim_age),
        ("State", state_label),
        ("ACA mode", base.aca_mode),
        ("Filing status", "Married filing jointly" if base.filing_status == "mfj" else "Single"),
        ("Tax params", base.params.label),
    ]
    for label, value in items:
        st.markdown(f"- **{label}:** {value}")

    if base.income_streams:
        st.markdown("##### Income streams")
        st.dataframe(
            [{"Name": s.name, "Annual $": f"${s.annual_amount:,.0f}",
              "Start age": s.start_age, "End age": s.end_age,
              "Taxable?": "yes" if s.taxable else "no"} for s in base.income_streams],
            hide_index=True, use_container_width=True,
        )
    if base.expense_streams:
        st.markdown("##### Expense streams")
        st.dataframe(
            [{"Name": s.name, "Annual $": f"${s.annual_amount:,.0f}",
              "Start age": s.start_age, "End age": s.end_age} for s in base.expense_streams],
            hide_index=True, use_container_width=True,
        )

    st.markdown("##### Notes & assumptions baked in")
    st.markdown(
        "- All values are **real** (inflation-adjusted) dollars; growth rate is real return.\n"
        "- Single filer. Federal brackets are 2026 estimates (2025 actuals).\n"
        "- Growth is applied at the **start** of each year, then withdrawal/conversion happens.\n"
        "- 10% early-withdrawal penalty applies to Traditional withdrawals before age 60 (proxy for 59.5).\n"
        "- HSA non-medical withdrawals are modeled as taxable income; the additional 20% pre-65 penalty is **not** modeled (HSA is last-resort here, so impact is small in well-formed plans).\n"
        "- State tax: pluggable. 9 no-tax states + WA's 7% LTCG above $262k pre-configured; CUSTOM lets you enter flat marginal rates. Per-state quirks (SS exemptions, pension exclusions, brackets) are NOT modeled — see How-it-works for details.\n"
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

### Accumulation phase

When `current_age < retirement_age`, the engine runs an **accumulation branch** for each
intervening year before switching to the full retirement loop. If the two ages are equal
(the default), no accumulation years run and behavior is identical to the original
retirement-only model.

**Per-year accumulation order:**
1. Real growth is applied to all accounts first.
2. `annual_savings + active income streams + employer match` forms the **contribution pool**. Employer match is capped at `employer_match_cap_pct` of wages when non-zero. Pool minus active expense streams is allocated across accounts per `savings_allocation`. **Employer match always lands in Traditional.**
3. If expense streams exceed the pool, the shortfall draws from **cash first**, then **taxable brokerage** (with LTCG gross-up). Tax-advantaged depletion (Roth, Traditional with penalty) is intentionally out of scope — if cash and taxable are exhausted, the year is flagged as a shortfall and the user should revisit their inputs. See `ROADMAP.md` for planned improvements.

**Simple-tax contract:** savings are entered as **net dollars** — the user has already
accounted for income tax and payroll in `annual_savings`. The engine does not model FICA
or income tax on wages during accumulation. Contributions flow directly into accounts
without a tax deduction pass.

**Why the model still asks for `accumulation_wage_income`:** when an expense shock forces
a taxable sale, the realized LTCG stacks on top of wages for federal bracket purposes.
Without wages, LTCG would always fall in the 0% bracket and the simulation would silently
undertax shocks. State LTCG uses the existing flat-rate model unchanged.

**Concrete example:** a $200k house down-payment at age 40, against $50k cash + $300k
taxable (50% gain ratio) at $150k wages, will sell ≈ $162k of taxable, realize ~$81k
LTCG on top of wages, and pay ~$12k federal LTCG tax to net the required amount.

---

### Dynamic retirement trigger (target net worth mode)

Standard fixed-age Monte Carlo is pessimistic for early-retirement scenarios: a bad
sequence before the assumed retirement date depletes the portfolio and locks in a weakened
starting balance, when in reality most people would simply delay retirement by a year or
two until the portfolio recovers. The **target net worth** trigger corrects for this by
letting the engine decide when retirement begins based on whether the portfolio has reached
the FI goal, rather than always retiring at the same fixed age regardless of market
conditions. The classic FI target is 25× annual spend (the 4% rule heuristic), which you
can populate with the "Set target = spend × 25" button.

Three parameters control the trigger: `retirement_target_nw` is the real-dollar net worth
threshold; `retirement_age_floor` is the earliest age at which the trigger can fire (useful
when there are accumulation commitments before a certain age); and `retirement_age` (the
"Retirement age" field in the Horizon section) acts as a ceiling — if the target is never
hit, retirement begins at that ceiling age regardless. At the end of each accumulation year
the engine checks whether the portfolio equals or exceeds the target and the floor age has
been reached; if so, the retirement phase begins the following year. In Monte Carlo runs,
each path triggers independently, producing a distribution of actual retirement ages across
paths; the histogram of those ages (shown in the Monte Carlo tab under this mode) reveals
how much sequence-of-returns risk compresses or stretches the accumulation window.

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
ladder gap from years 16–24 that forces some pre-60 penalty Trad pulls. Worse, in
deep historical drawdowns (1929, 1937 cohorts) the bracket-sized conversion drains the
*Traditional* balance itself before age 60 — so when the Roth ladder runs out at the
ladder-gap ages, there's nothing left to draw on at all.

#### `bridge_guarded` — like bridge_optimal, but cap by Trad reserve
Same bracket-driven target as bridge_optimal, but additionally capped at
`traditional_balance / years_to_60` in the pre-60 phase. The cap throttles automatically:
in good sequences Trad grows faster than the cap shrinks, so the bracket target binds
and behavior matches bridge_optimal. In bad sequences (Trad shrinking faster than time
remaining), the cap binds and conversion sizes drop, preserving the post-ladder backstop.
Post-60 the cap relaxes (no penalty risk). Best choice when robustness across
sequence-of-returns scenarios matters more than maximum Roth stacking.

#### `bridge_responsive` — scale by drawdown from peak
Blends between **`minimal_convert`** (no drawdown) and **`bridge_optimal`** (≥20% drawdown
from peak). Each year:

```
target = minimal_target + blend × (bracket_target − minimal_target)
blend  = min(drawdown_from_peak / 0.20, 1.0)
```

**The rationale:** a converted dollar buys more future-recovery shares when prices are
depressed than when the portfolio is at a peak. A 30% drawdown means the same tax bill
converts ~43% more shares. Conversely, converting at an all-time high pre-pays tax on
dollars that may subsequently fall — the same failure mode that makes `bridge_optimal`
costly in historical crash sequences.

Unlike `bridge_guarded`, there is no explicit Trad-reserve cap. Instead, the drawdown signal
naturally suppresses conversion in flat or rising markets and amplifies it when prices are
already down. The two strategies converge in bad sequences and diverge in good ones:
`bridge_guarded` caps by Trad balance mechanics; `bridge_responsive` caps by absence of
drawdown signal.

**Best for:** over-capitalized portfolios where the goal is preserving real net worth
rather than surviving a funding gap. If you don't need to spend down the portfolio,
converting aggressively at peak prices is a tax-efficiency leak with no upside.

#### `minimal_convert` — fill the standard deduction only
Convert exactly `standard_deduction` (~$15.7k) every year. This is **always free**: the
deduction wipes out the ordinary tax on it, and it's small enough not to push LTCG into
the 15% bracket. Lowest tax + ACA over the bridge years, but creates a starved ladder —
$80k spend can't be funded from $15.7k rungs, so you eat penalty Trad pulls in the gap.
Surprisingly robust on bad sequences because the under-conversion *preserves* Trad as a
late-bridge fallback — the same insurance bridge_guarded gets explicitly.

#### `aggressive_convert` — fill the top of the 12% bracket
Convert `standard_deduction + 48,475` (~$64.2k) every year. Ordinary tax rate stays at
12%, but every dollar above the 0% LTCG ceiling pushes that much LTCG up into the 15%
bracket. **Most tax-expensive in early years**, but the ladder is fully fed (each $64k
rung covers a full year of $80k spend with margin), so **zero pre-60 penalty** in the
deterministic case. Same Trad-drain failure mode as bridge_optimal under deep historical
drawdowns.

#### `custom` — your number
Set a fixed annual conversion amount. Useful for sensitivity testing or matching an
external plan.

---

### Reading the comparison view

The interesting comparison is **(federal tax + ACA premium + penalty)** vs **ending balance**
vs **success rate under stochastic / historical returns**:

- `minimal_convert` minimizes the visible cost (tax + ACA) but pays it back as penalty in
  the ladder gap. Quietly robust because under-conversion preserves the Trad backstop.
- `aggressive_convert` eliminates penalty entirely under deterministic returns at the cost
  of higher early-year tax — but its blind bracket-filling drains Trad in bad sequences.
- `bridge_optimal` aims for the deterministic sweet spot; same Trad-drain risk as aggressive.
- `bridge_guarded` accepts a small reduction in deterministic upside in exchange for an
  explicit reserve cap that prevents the Trad-drain failure mode. Often the dominant choice
  on robustness-weighted portfolios.
- `bridge_responsive` converts little during good runs and aggressively during drawdowns.
  Often matches `bridge_optimal` on preservation rate with slightly higher median ending
  balance; less protective than `bridge_guarded` in worst-case sequences. Best for
  over-capitalized portfolios where the optimization target is preservation, not depletion.

There is no single "right" answer. Pick `aggressive` for terminal wealth in good times,
`bridge_guarded` or `minimal` for survival in bad sequences, `bridge_responsive` if you are
over-capitalized and targeting real-value preservation, `bridge_optimal` if you believe
returns will be close to the long-run mean.

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

---

### State income & capital-gains tax

State tax is modeled with a flat-rate approximation: one rate on ordinary income, one
rate on long-term capital gains (with optional threshold). Pre-configured presets cover
the 9 no-income-tax states (TX, FL, NV, AK, NH, SD, TN, WY, NONE) plus Washington's
specific 7% LTCG rule above $262k. For any other state, choose **CUSTOM** and enter
your top marginal rates.

This is a deliberately rough model. It **does not** capture per-state bracket structures,
SS taxation rules (most states exempt SS, some don't), pension or retirement-income
exclusions (PA exempts most, IL exempts pensions, etc.), or itemized deductions. For a
high-tax state like CA or NY, entering the top marginal rate as the flat rate is a
*conservative* approximation — your actual state tax will usually be lower because
withdrawal-era ordinary income often falls below top brackets and SS/pension exclusions
apply. For state-specific filing prep, consult a tax pro.

---

### Filing status (single vs MFJ)

The sidebar **Filing status** radio switches between two tax parameter sets.

**MFJ differences from single:**
- **Federal brackets:** wider ordinary-income and LTCG brackets; standard deduction ~$31,500
  (~2× single's ~$15,750). More ordinary income fits in the 0% and 12% brackets.
- **ACA:** household-of-2 FPL (~$21,150 vs ~$15,060 for one person); benchmark premium
  is set to ~$16,000 (covering both spouses). The subsidy curve shifts right, so the same
  MAGI buys more subsidy, but the higher benchmark means ACA costs are larger in absolute
  dollars above the cap.
- **Social Security taxation:** provisional-income thresholds are $32,000 / $44,000
  (vs $25,000 / $34,000 single). More room for SS to remain partially non-taxable.
- **Medicare (post-65):** both spouses are on Medicare — cost is doubled. Each spouse
  pays their own Part B + Part D + IRMAA per-person surcharge. IRMAA tiers are
  approximately 2× the single thresholds, so a given MAGI lands in a similar tier but
  the dollar cost is two premiums.

**Model caveats:**
- The engine treats the household as a unit. The `ss_annual_benefit` field is interpreted
  as the combined household SS amount. If spouses claim at different ages or have very
  different individual benefits, model spouse 2 as a separate income stream for accurate
  timing.
- Washington's $262k LTCG threshold applies per filing unit and does not change under MFJ
  — a couple filing jointly shares the same $262k exemption as a single filer.

---

### Spending: fixed dollars or SWR percent

Spending can be entered two ways:

- **Fixed annual amount:** a real-dollar number that stays constant in real terms across
  all years.
- **% of starting portfolio (SWR):** a Safe Withdrawal Rate as a fraction. The engine
  resolves it to dollars at simulation start as `starting_total × spend_rate` and uses
  that constant amount for every year. 0.04 corresponds to the 4% rule; 0.035 is more
  conservative; 0.030 lean-FIRE territory.

Both modes produce an identical year-1 dollar amount when the rate matches; the
difference is intent. SWR mode is useful for sensitivity testing across portfolio sizes
without manually recomputing the dollar figure each time.
"""
    )




# --- main ----------------------------------------------------------------

st.title("Early Retirement Withdrawal Planner")
st.caption("Tax + penalty optimization for early retirees. Real dollars throughout.")

if "custom_conversion" not in st.session_state:
    st.session_state["custom_conversion"] = 40_000

base = render_sidebar()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single scenario", "Strategies", "How it works", "Inputs & assumptions"]
)

with tab1:
    single_scenario_view(base)
with tab2:
    strategies_view(base)
with tab3:
    how_it_works_view()
with tab4:
    inputs_summary_view(base)
