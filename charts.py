"""Plotly chart builders. All values are real dollars unless noted."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from planner.simulate import YearResult


PALETTE = {
    "cash": "#9CA3AF",
    "taxable": "#3B82F6",
    "traditional": "#EF4444",
    "roth": "#10B981",
    "hsa": "#A855F7",
    "tax_ord": "#DC2626",
    "tax_ltcg": "#F59E0B",
    "aca": "#0EA5E9",
    "medicare": "#0369A1",
    "penalty": "#7C2D12",
    "spend": "#1F2937",
    "conversion": "#10B981",
}

STRATEGY_DISPLAY = {
    "bridge_optimal": "Bridge optimal",
    "bridge_guarded": "Bridge guarded",
    "minimal_convert": "Minimal convert",
    "aggressive_convert": "Aggressive convert",
    "custom": "Custom",
}

STRATEGY_COLORS = {
    "bridge_optimal": "#3B82F6",      # blue
    "bridge_guarded": "#0EA5E9",      # cyan
    "minimal_convert": "#10B981",     # green
    "aggressive_convert": "#F59E0B",  # amber
    "custom": "#A855F7",              # purple
}
_STRATEGY_FALLBACK = "#6B7280"


def _layout(title: str, height: int = 420, **kwargs) -> dict:
    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", color="#111827"),
        margin=dict(l=60, r=20, t=60, b=50),
        height=height,
        xaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=False, **kwargs.get("xaxis", {})),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=False, **kwargs.get("yaxis", {})),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )


def _ages(results: List[YearResult]) -> List[int]:
    return [r.age for r in results]


def balance_stack(results: List[YearResult]) -> go.Figure:
    """Stacked area: account balances over time (real dollars)."""
    ages = _ages(results)
    series = {
        "Cash": ([r.snapshot["cash"] for r in results], PALETTE["cash"]),
        "Taxable": ([r.snapshot["taxable"] for r in results], PALETTE["taxable"]),
        "Traditional 401k": ([r.snapshot["traditional"] for r in results], PALETTE["traditional"]),
        "Roth": ([r.snapshot["roth"] for r in results], PALETTE["roth"]),
        "HSA": ([r.snapshot["hsa"] for r in results], PALETTE["hsa"]),
    }
    fig = go.Figure()
    for name, (values, color) in series.items():
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=values,
                name=name,
                stackgroup="balances",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate="%{y:$,.0f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        **_layout(
            "Account balances over time (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Balance", tickformat="$,.0f"),
        )
    )
    return fig


def cashflow_bars(results: List[YearResult]) -> go.Figure:
    """Per-year stacked bars: spending funded, taxes, ACA OOP, penalty."""
    ages = _ages(results)
    spending = [r.target_net for r in results]
    fed_tax = [r.plan.federal_tax for r in results]
    aca_pre65 = [r.plan.healthcare_oop if r.age < 65 else 0 for r in results]
    medicare_65plus = [r.plan.healthcare_oop if r.age >= 65 else 0 for r in results]
    penalty = [r.plan.penalty for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ages, y=spending, name="Net spending", marker_color=PALETTE["spend"], hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=fed_tax, name="Federal tax", marker_color=PALETTE["tax_ord"], hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=aca_pre65, name="ACA premium (pre-65)", marker_color=PALETTE["aca"], hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=medicare_65plus, name="Medicare + IRMAA (65+)", marker_color=PALETTE["medicare"], hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=penalty, name="Early-withdrawal penalty", marker_color=PALETTE["penalty"], hovertemplate="%{y:$,.0f}"))
    fig.update_layout(
        **_layout(
            "Annual cash outflows (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Cash outflow", tickformat="$,.0f"),
        ),
        barmode="stack",
    )
    return fig


def conversions_bars(results: List[YearResult]) -> go.Figure:
    """Annual Roth conversion amounts (kept on its own panel to avoid dual-axis confusion)."""
    ages = _ages(results)
    conv = [r.plan.conversion for r in results]
    fig = go.Figure(
        go.Bar(x=ages, y=conv, marker_color=PALETTE["conversion"], hovertemplate="%{y:$,.0f}")
    )
    fig.update_layout(
        **_layout(
            "Annual Roth conversions (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Conversion amount", tickformat="$,.0f"),
        )
    )
    return fig


def tax_breakdown(results: List[YearResult]) -> go.Figure:
    """Stacked bars: federal ordinary, federal LTCG (combined into federal here), ACA OOP, penalty."""
    ages = _ages(results)
    fed_tax = [r.plan.federal_tax for r in results]
    aca_pre65 = [r.plan.healthcare_oop if r.age < 65 else 0 for r in results]
    medicare_65plus = [r.plan.healthcare_oop if r.age >= 65 else 0 for r in results]
    penalty = [r.plan.penalty for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ages, y=fed_tax, name="Federal income tax", marker_color=PALETTE["tax_ord"]))
    fig.add_trace(go.Bar(x=ages, y=aca_pre65, name="ACA premium (pre-65)", marker_color=PALETTE["aca"]))
    fig.add_trace(go.Bar(x=ages, y=medicare_65plus, name="Medicare + IRMAA (65+)", marker_color=PALETTE["medicare"]))
    fig.add_trace(go.Bar(x=ages, y=penalty, name="Early-withdrawal penalty", marker_color=PALETTE["penalty"]))
    fig.update_layout(
        **_layout(
            "Tax + healthcare cost breakdown (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Cost", tickformat="$,.0f"),
        ),
        barmode="stack",
    )
    return fig


def withdrawal_rate(results: List[YearResult]) -> go.Figure:
    """Line: gross withdrawal / beginning-of-year portfolio, with safe-rate band."""
    ages = _ages(results)
    rates = [r.withdrawal_rate * 100 for r in results]
    fig = go.Figure()
    # Safe-rate band drawn as two scatters with fill='tonexty' (cheaper than add_hrect).
    band_low = [3.25] * len(ages)
    band_high = [4.0] * len(ages)
    fig.add_trace(go.Scatter(x=ages, y=band_low, mode="lines", line=dict(color="#10B981", width=1, dash="dot"),
                             name="60-yr safe rate ~3.25%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=band_high, mode="lines", line=dict(color="#10B981", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(16,185,129,0.12)",
                             name="4% rule", hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=rates,
            mode="lines",
            name="Withdrawal rate",
            line=dict(color="#1F2937", width=2),
            hovertemplate="%{y:.2f}%",
        )
    )
    fig.update_layout(
        **_layout(
            "Withdrawal rate vs. safe-rate band",
            xaxis=dict(title="Age"),
            yaxis=dict(title="% of beginning-of-year portfolio", ticksuffix="%"),
        )
    )
    return fig


def ladder_status(results: List[YearResult]) -> go.Figure:
    """Roth-ladder rung balance over time (total unwithdrawn rungs)."""
    ages = _ages(results)
    rung_total = [r.snapshot["roth_seasoned"] for r in results]
    earnings = [r.snapshot["roth_earnings"] for r in results]
    contribs = [r.snapshot["roth_contributions"] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ages, y=contribs, name="Contributions", marker_color="#A7F3D0",
                         hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=rung_total, name="Conversion rungs", marker_color="#10B981",
                         hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Bar(x=ages, y=earnings, name="Earnings (locked pre-59.5)", marker_color="#065F46",
                         hovertemplate="%{y:$,.0f}"))
    fig.update_layout(
        **_layout(
            "Roth IRA composition over time (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Balance", tickformat="$,.0f"),
        ),
        barmode="stack",
    )
    return fig


# --- comparison charts ------------------------------------------------------


def compare_balance_trajectory(scenarios: Dict[str, List[YearResult]]) -> go.Figure:
    """Multi-line: total portfolio over time per strategy."""
    fig = go.Figure()
    for name, results in scenarios.items():
        color = STRATEGY_COLORS.get(name, _STRATEGY_FALLBACK)
        display_name = STRATEGY_DISPLAY.get(name, name)
        fig.add_trace(
            go.Scatter(
                x=_ages(results),
                y=[r.ending_total for r in results],
                mode="lines",
                name=display_name,
                line=dict(color=color, width=2),
                hovertemplate="%{y:$,.0f}<extra>" + display_name + "</extra>",
            )
        )
    fig.update_layout(
        **_layout(
            "Total portfolio over time by strategy (real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Ending balance", tickformat="$,.0f"),
        )
    )
    return fig


def compare_ending_balance(scenarios: Dict[str, List[YearResult]]) -> go.Figure:
    names = list(scenarios.keys())
    endings = [scenarios[n][-1].ending_total if scenarios[n] else 0 for n in names]
    colors = [STRATEGY_COLORS.get(n, _STRATEGY_FALLBACK) for n in names]
    fig = go.Figure(go.Bar(x=names, y=endings, marker_color=colors, hovertemplate="%{y:$,.0f}"))
    fig.update_layout(
        **_layout(
            "Ending balance by strategy (real $)",
            xaxis=dict(title="Strategy"),
            yaxis=dict(title="Ending balance", tickformat="$,.0f"),
        )
    )
    return fig


def compare_cumulative_costs(scenarios: Dict[str, List[YearResult]]) -> go.Figure:
    """Grouped bar: lifetime federal tax + ACA + penalty per strategy."""
    names = list(scenarios.keys())
    fed = [sum(r.plan.federal_tax for r in scenarios[n]) for n in names]
    aca = [sum(r.plan.healthcare_oop for r in scenarios[n]) for n in names]
    pen = [sum(r.plan.penalty for r in scenarios[n]) for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=fed, name="Federal tax", marker_color=PALETTE["tax_ord"]))
    fig.add_trace(go.Bar(x=names, y=aca, name="Healthcare OOP", marker_color=PALETTE["aca"]))
    fig.add_trace(go.Bar(x=names, y=pen, name="Early-withdrawal penalty", marker_color=PALETTE["penalty"]))
    fig.update_layout(
        **_layout(
            "Lifetime tax + healthcare cost by strategy",
            xaxis=dict(title="Strategy"),
            yaxis=dict(title="Total cost", tickformat="$,.0f"),
        ),
        barmode="stack",
    )
    return fig


def compare_shortfall(scenarios: Dict[str, List[YearResult]]) -> go.Figure:
    """Lifetime shortfall (years where target spending couldn't be met) per strategy."""
    names = list(scenarios.keys())
    short = [sum(r.plan.shortfall for r in scenarios[n]) for n in names]
    fig = go.Figure(go.Bar(x=names, y=short, marker_color="#DC2626", hovertemplate="%{y:$,.0f}"))
    fig.update_layout(
        **_layout(
            "Lifetime shortfall by strategy (real $)",
            xaxis=dict(title="Strategy"),
            yaxis=dict(title="Total shortfall", tickformat="$,.0f"),
        )
    )
    return fig


def per_year_table(results: List[YearResult]) -> pd.DataFrame:
    """Flat table for display."""
    rows = []
    for r in results:
        s = r.snapshot
        p = r.plan
        rows.append({
            "Year": r.year,
            "Age": r.age,
            "Total $": round(r.ending_total),
            "Cash": round(s["cash"]),
            "Taxable": round(s["taxable"]),
            "Trad 401k": round(s["traditional"]),
            "Roth": round(s["roth"]),
            "HSA": round(s["hsa"]),
            "Sale": round(p.withdrawals.taxable),
            "Conv": round(p.conversion),
            "Roth wd (seas)": round(p.withdrawals.roth_seasoned),
            "Roth wd (post-60)": round(p.withdrawals.roth_post60),
            "Trad wd": round(p.withdrawals.traditional),
            "LTCG": round(p.ltcg),
            "MAGI": round(p.magi),
            "Fed tax": round(p.federal_tax),
            "Healthcare OOP": round(p.healthcare_oop),
            "Penalty": round(p.penalty),
            "Withdraw %": round(r.withdrawal_rate * 100, 2),
            "Shortfall": round(p.shortfall),
        })
    return pd.DataFrame(rows)


# --- Monte Carlo charts -----------------------------------------------------


def mc_fan_chart(mc) -> go.Figure:
    """Fan chart: P5/P25/P50/P75/P95 portfolio trajectories from Monte Carlo."""
    fig = go.Figure()
    # Outer band P5–P95
    fig.add_trace(go.Scatter(x=mc.age, y=mc.p5_balance, mode="lines",
                             line=dict(color="#3B82F6", width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=mc.age, y=mc.p95_balance, mode="lines",
                             line=dict(color="#3B82F6", width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.10)",
                             name="5th–95th percentile", hoverinfo="skip"))
    # Inner band P25–P75
    fig.add_trace(go.Scatter(x=mc.age, y=mc.p25_balance, mode="lines",
                             line=dict(color="#3B82F6", width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=mc.age, y=mc.p75_balance, mode="lines",
                             line=dict(color="#3B82F6", width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.25)",
                             name="25th–75th percentile", hoverinfo="skip"))
    # Median line
    fig.add_trace(go.Scatter(x=mc.age, y=mc.p50_balance, mode="lines",
                             line=dict(color="#1F2937", width=2), name="Median",
                             hovertemplate="%{y:$,.0f}"))
    fig.update_layout(
        **_layout(
            f"Portfolio fan chart — Monte Carlo ({mc.n_runs} paths, real $)",
            xaxis=dict(title="Age"),
            yaxis=dict(type="log", title="Portfolio balance (log scale, real $)", tickformat="$,.0f"),
        )
    )
    return fig


def mc_success_kpis(mc) -> Dict[str, str]:
    """Format MC summary for KPI display."""
    out = {
        "Success rate": f"{mc.success_rate * 100:.1f}%",
        "Median ending (real $)": f"${mc.median_ending:,.0f}",
        "5th-pctile ending": f"${mc.p5_ending:,.0f}",
        "95th-pctile ending": f"${mc.p95_ending:,.0f}",
    }
    if mc.median_depletion_age is not None:
        out["Median depletion age"] = f"{mc.median_depletion_age:.0f}"
    return out
