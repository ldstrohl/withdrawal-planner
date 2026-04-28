"""State income / capital-gains tax — flat-rate approximation.

Each `StateTaxParams` represents one state with two flat rates: ordinary income
and long-term capital gains (with optional threshold). Most no-income-tax states
use the defaults (zero everywhere). WA is special — no wage tax but a 7% LTCG
rate above $262k. Other states use a "CUSTOM" option where the user enters their
own marginal rate; this is a deliberate approximation.

The flat-rate model intentionally ignores per-state quirks (bracket structures,
SS/pension exclusions, retirement-income carve-outs). Use it for cross-state
ballpark comparison, not precise state filing prep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StateTaxParams:
    name: str                         # human-readable label for UI
    code: str                         # 2-letter state code or "NONE"/"CUSTOM"
    ordinary_rate: float = 0.0        # flat marginal rate on ordinary income
    ltcg_rate: float = 0.0            # flat rate on LTCG above threshold
    ltcg_threshold: float = 0.0
    note: str = ""                    # optional caveat shown in UI


STATE_PRESETS: Dict[str, StateTaxParams] = {
    "NONE": StateTaxParams("None / not residing in a US state", "NONE"),
    "AK":   StateTaxParams("Alaska — no income tax", "AK"),
    "FL":   StateTaxParams("Florida — no income tax", "FL"),
    "NV":   StateTaxParams("Nevada — no income tax", "NV"),
    "NH":   StateTaxParams("New Hampshire — wages exempt", "NH",
                           note="Interest/dividends were taxed historically; phased out as of 2025."),
    "SD":   StateTaxParams("South Dakota — no income tax", "SD"),
    "TN":   StateTaxParams("Tennessee — no income tax", "TN"),
    "TX":   StateTaxParams("Texas — no income tax", "TX"),
    "WA":   StateTaxParams("Washington — 7% LTCG above $262k", "WA",
                           ltcg_rate=0.07, ltcg_threshold=262_000.0,
                           note="No wage tax. 7% capital-gains tax on LTCG above $262k threshold (2024 indexed)."),
    "WY":   StateTaxParams("Wyoming — no income tax", "WY"),
    "CUSTOM": StateTaxParams("Custom — enter rates below", "CUSTOM",
                             note="Flat marginal-rate approximation. Ignores brackets, SS exemption, pension exclusions."),
}


def state_tax(ordinary_income: float, ltcg: float, params: StateTaxParams) -> float:
    """Flat marginal-rate state tax on ordinary income + LTCG above threshold."""
    ord_tax = max(ordinary_income, 0.0) * params.ordinary_rate
    cg_excess = max(ltcg - params.ltcg_threshold, 0.0)
    cg_tax = cg_excess * params.ltcg_rate
    return ord_tax + cg_tax


def resolve_state_params(
    code: str,
    custom_ordinary_rate: float = 0.0,
    custom_ltcg_rate: float = 0.0,
    custom_ltcg_threshold: float = 0.0,
) -> StateTaxParams:
    """Look up a preset by code; for CUSTOM, build from the supplied rates."""
    if code == "CUSTOM":
        return StateTaxParams(
            name="Custom",
            code="CUSTOM",
            ordinary_rate=custom_ordinary_rate,
            ltcg_rate=custom_ltcg_rate,
            ltcg_threshold=custom_ltcg_threshold,
        )
    return STATE_PRESETS.get(code, STATE_PRESETS["NONE"])
