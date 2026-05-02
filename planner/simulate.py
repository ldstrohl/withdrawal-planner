"""Year-by-year simulation orchestrator.

Order of operations within a year (real-dollar terms):
  1. Apply real growth to all accounts.
  2. Run strategy planner -> withdrawals + conversion decision.
  3. Apply withdrawals to portfolio.
  4. Apply conversion (Trad balance -> new Roth ladder rung).
  5. Record outcomes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .accounts import Cash, HSA, Portfolio, RothIRA, Taxable, TraditionalIRA
from .returns import ConstantReturns, ReturnsModel
from .state_tax import StateTaxParams, resolve_state_params, state_tax
from .strategy import PlanResult, Withdrawals, plan_year
from .streams import ExpenseStream, IncomeStream, active_expense, active_income
from .tax import TAX_PARAMS_2026, TaxParams, federal_tax, required_min_distribution, tax_params_for


@dataclass
class YearResult:
    year: int  # 0-indexed year of retirement
    age: int
    starting_total: float
    ending_total: float
    plan: PlanResult
    snapshot: dict  # account snapshot at end-of-year
    target_net: float
    withdrawal_rate: float  # gross withdrawals / starting_total


@dataclass(frozen=True)
class SimulationInputs:
    initial_cash: float = 25_000
    initial_taxable: float = 834_843
    taxable_basis: float = 570_659
    initial_traditional: float = 837_547
    initial_roth: float = 327_610
    roth_contributions: float = 0.0  # of the Roth balance, how much is direct contributions
    initial_hsa: float = 26_000
    target_spend: float = 80_000  # real, net
    # Real returns by asset class (used by ConstantReturns when no model is passed).
    stock_return: float = 0.07
    bond_return: float = 0.02
    cash_return: float = 0.0
    # Stock allocation per investment account (0..1). Cash account ignores allocation.
    stock_allocation: float = 0.85
    start_age: int = 35
    horizon_years: int = 60
    strategy: str = "bridge_optimal"
    custom_conversion: Optional[float] = None
    aca_mode: str = "cap"
    filing_status: str = "single"     # "single" | "mfj"
    params: TaxParams = field(default_factory=lambda: TAX_PARAMS_2026)
    ss_annual_benefit: float = 0.0
    ss_claim_age: int = 67
    income_streams: tuple = ()    # tuple of IncomeStream (real $)
    expense_streams: tuple = ()   # tuple of ExpenseStream (real $)
    state_code: str = "WA"        # see planner.state_tax.STATE_PRESETS
    state_ordinary_rate: float = 0.0    # used only when state_code == "CUSTOM"
    state_ltcg_rate: float = 0.0
    state_ltcg_threshold: float = 0.0
    spend_mode: str = "fixed"     # "fixed" (uses target_spend) | "swr" (rate × starting NW)
    spend_rate: float = 0.035     # used only when spend_mode == "swr"
    current_age: Optional[int] = None         # when None, treated as start_age (no accumulation phase)
    retirement_age: Optional[int] = None      # when None, treated as start_age
    annual_savings: float = 0.0
    savings_allocation: tuple = ()            # tuple of (account_name, fraction) pairs
    employer_match_pct: float = 0.0
    employer_match_cap_pct: float = 0.0       # 0 means uncapped match
    accumulation_wage_income: float = 0.0


def build_portfolio(inputs: SimulationInputs) -> Portfolio:
    return Portfolio(
        cash=Cash(balance=inputs.initial_cash),
        taxable=Taxable(balance=inputs.initial_taxable, basis=inputs.taxable_basis),
        traditional=TraditionalIRA(balance=inputs.initial_traditional),
        roth=RothIRA(
            contributions=inputs.roth_contributions,
            earnings=inputs.initial_roth - inputs.roth_contributions,
        ),
        hsa=HSA(balance=inputs.initial_hsa),
        stock_allocation_taxable=inputs.stock_allocation,
        stock_allocation_traditional=inputs.stock_allocation,
        stock_allocation_roth=inputs.stock_allocation,
        stock_allocation_hsa=inputs.stock_allocation,
    )


def _apply_action(portfolio: Portfolio, plan: PlanResult, year: int) -> None:
    w = plan.withdrawals
    portfolio.cash.withdraw(w.cash)
    portfolio.taxable.sell(w.taxable)
    portfolio.roth.withdraw_seasoned(w.roth_seasoned, current_year=year)
    portfolio.roth.withdraw_contributions(w.roth_contributions)
    portfolio.roth.withdraw_any(w.roth_post60)
    portfolio.traditional.withdraw(w.traditional)
    portfolio.hsa.withdraw(w.hsa)

    # Re-credit any RMD surplus back to cash.
    # When rmd_amount > gross_need, w.traditional was force-bumped above what cash-flow
    # required. The excess lands in cash (the retiree receives it, doesn't spend it yet).
    gross_withdrawn = (
        w.cash + w.taxable + w.roth_seasoned + w.roth_contributions
        + w.roth_post60 + w.traditional + w.hsa
    )
    surplus = gross_withdrawn - plan.gross_used
    if surplus > 0:
        portfolio.cash.balance += surplus

    # Apply conversion: pull from Trad, add to Roth ladder rung
    if plan.conversion > 0:
        actually_converted = portfolio.traditional.withdraw(plan.conversion)
        portfolio.roth.add_conversion(year=year, amount=actually_converted)


def _force_taxable_sale(
    taxable: Taxable,
    net_need: float,
    wage_income: float,
    tax_params: TaxParams,
    state_params: StateTaxParams,
) -> Tuple[float, float, float, float, float]:
    """Sell from Taxable, grossing up to cover the LTCG tax it generates.

    Returns (proceeds, ltcg_realized, federal_ltcg_tax, state_ltcg_tax, unmet).

    LTCG stacks on `wage_income` for federal bracket purposes (no other ordinary
    income during accumulation under simple-tax mode). State LTCG uses the
    flat-rate model with no ordinary-income interaction.
    """
    if net_need <= 0 or taxable.balance <= 0:
        return 0.0, 0.0, 0.0, 0.0, max(net_need, 0.0)
    gain_ratio = taxable.gain_ratio
    gross = net_need
    for _ in range(20):
        sale = min(gross, taxable.balance)
        gain = sale * gain_ratio
        f_tax = federal_tax(wage_income, gain, tax_params)["ltcg"]
        s_tax = state_tax(0.0, gain, state_params)
        net_after_tax = sale - f_tax - s_tax
        if abs(net_after_tax - net_need) < 0.5 or sale >= taxable.balance:
            break
        gross = net_need + f_tax + s_tax
    sale = min(gross, taxable.balance)
    proceeds, ltcg = taxable.sell(sale)
    f_tax = federal_tax(wage_income, ltcg, tax_params)["ltcg"]
    s_tax = state_tax(0.0, ltcg, state_params)
    net_after_tax = proceeds - f_tax - s_tax
    unmet = max(net_need - net_after_tax, 0.0)
    return proceeds, ltcg, f_tax, s_tax, unmet


def _run_accumulation_year(
    portfolio: Portfolio,
    age: int,
    year_index: int,
    returns_model: ReturnsModel,
    path_index: int,
    inputs: "SimulationInputs",
    tax_params: TaxParams,
    state_params: StateTaxParams,
    results: List[YearResult],
    effective_spend: float,
) -> None:
    starting_total = portfolio.total
    year_returns = returns_model.get(year_index=year_index, path_index=path_index)
    portfolio.apply_growth(year_returns)

    wages = inputs.accumulation_wage_income
    match_raw = wages * inputs.employer_match_pct
    if inputs.employer_match_cap_pct > 0:
        match = min(match_raw, wages * inputs.employer_match_cap_pct)
    else:
        match = match_raw

    sched_income, _sched_taxable = active_income(inputs.income_streams, age)
    sched_expense = active_expense(inputs.expense_streams, age)
    pool = inputs.annual_savings + sched_income + match

    allocation_dict = dict(inputs.savings_allocation)

    net = pool - sched_expense
    ltcg_realized = 0.0
    ltcg_tax_federal = 0.0
    ltcg_tax_state = 0.0
    unmet = 0.0
    if net >= 0:
        if match > 0:
            portfolio.traditional.deposit(match)
        remainder = max(net - match, 0.0)
        if remainder > 0:
            portfolio.contribute(allocation_dict, remainder)
        contribution = max(net, 0.0)
    else:
        if match > 0:
            portfolio.traditional.deposit(match)
        shortfall_remaining = -net
        cash_take = portfolio.cash.withdraw(shortfall_remaining)
        shortfall_remaining -= cash_take
        if shortfall_remaining > 0:
            _proceeds, ltcg_realized, ltcg_tax_federal, ltcg_tax_state, unmet = _force_taxable_sale(
                portfolio.taxable,
                shortfall_remaining,
                wages,
                tax_params,
                state_params,
            )
        contribution = match

    plan = PlanResult(
        withdrawals=Withdrawals(),
        ltcg=ltcg_realized,
        conversion=0.0,
        ordinary_income=0.0,
        federal_tax=ltcg_tax_federal,
        penalty=0.0,
        healthcare_oop=0.0,
        magi=0.0,
        target_net=0.0,
        funded=0.0,
        shortfall=unmet,
        state_tax=ltcg_tax_state,
        gross_used=0.0,
        scheduled_income=sched_income,
        scheduled_taxable_income=0.0,
        scheduled_expense=sched_expense,
        phase="accumulation",
        contribution=contribution,
    )

    ending_total = portfolio.total
    results.append(
        YearResult(
            year=year_index,
            age=age,
            starting_total=starting_total,
            ending_total=ending_total,
            plan=plan,
            snapshot=portfolio.snapshot(),
            target_net=0.0,
            withdrawal_rate=0.0,
        )
    )


def simulate(
    inputs: SimulationInputs,
    returns_model: Optional[ReturnsModel] = None,
    path_index: int = 0,
) -> List[YearResult]:
    """Run a single deterministic simulation path.

    `returns_model` defaults to ConstantReturns built from `inputs.stock_return / bond_return / cash_return`.
    For Monte Carlo, pass a stochastic model and vary `path_index` per run.
    """
    if returns_model is None:
        returns_model = ConstantReturns(
            stocks=inputs.stock_return,
            bonds=inputs.bond_return,
            cash=inputs.cash_return,
        )

    portfolio = build_portfolio(inputs)
    results: List[YearResult] = []

    # Resolve scenario-wide derived params once.
    state_params = resolve_state_params(
        inputs.state_code,
        custom_ordinary_rate=inputs.state_ordinary_rate,
        custom_ltcg_rate=inputs.state_ltcg_rate,
        custom_ltcg_threshold=inputs.state_ltcg_threshold,
    )
    # filing_status is the source of truth; inputs.params is legacy and overridden here.
    tax_params = tax_params_for(inputs.filing_status)
    if inputs.spend_mode == "swr":
        effective_spend = portfolio.total * inputs.spend_rate
    else:
        effective_spend = inputs.target_spend

    current_age = inputs.current_age if inputs.current_age is not None else inputs.start_age
    retirement_age = inputs.retirement_age if inputs.retirement_age is not None else inputs.start_age

    peak_total = portfolio.total
    for y in range(inputs.horizon_years):
        age = current_age + y

        if age < retirement_age:
            _run_accumulation_year(
                portfolio=portfolio,
                age=age,
                year_index=y,
                returns_model=returns_model,
                path_index=path_index,
                inputs=inputs,
                tax_params=tax_params,
                state_params=state_params,
                results=results,
                effective_spend=effective_spend,
            )
            peak_total = max(peak_total, portfolio.total)
            continue

        # 1. Capture beginning-of-year balance (pre-growth) for SWR-conventional WR denominator.
        starting_total = portfolio.total
        year_returns = returns_model.get(year_index=y, path_index=path_index)
        portfolio.apply_growth(year_returns)
        peak_total = max(peak_total, portfolio.total)
        drawdown_from_peak = max(0.0, 1.0 - portfolio.total / peak_total) if peak_total > 0 else 0.0

        # 2. Plan the year.
        ss = inputs.ss_annual_benefit if age >= inputs.ss_claim_age else 0.0
        rmd = required_min_distribution(portfolio.traditional.balance, age)
        sched_income, sched_taxable = active_income(inputs.income_streams, age)
        sched_expense = active_expense(inputs.expense_streams, age)
        plan = plan_year(
            portfolio=portfolio,
            age=age,
            year=y,
            target_net=effective_spend,
            strategy_name=inputs.strategy,
            params=tax_params,
            aca_mode=inputs.aca_mode,
            custom_conversion=inputs.custom_conversion,
            ss_income=ss,
            rmd_amount=rmd,
            scheduled_income=sched_income,
            scheduled_taxable_income=sched_taxable,
            scheduled_expense=sched_expense,
            state_params=state_params,
            drawdown_from_peak=drawdown_from_peak,
            filing_status=inputs.filing_status,
        )

        # 3+4. Apply.
        _apply_action(portfolio, plan, year=y)

        # 5. Record.
        ending_total = portfolio.total
        gross_withdrawn = (
            plan.withdrawals.cash
            + plan.withdrawals.taxable
            + plan.withdrawals.roth_seasoned
            + plan.withdrawals.roth_contributions
            + plan.withdrawals.roth_post60
            + plan.withdrawals.traditional
            + plan.withdrawals.hsa
        )
        wr = gross_withdrawn / starting_total if starting_total > 0 else 0.0

        results.append(
            YearResult(
                year=y,
                age=age,
                starting_total=starting_total,
                ending_total=ending_total,
                plan=plan,
                snapshot=portfolio.snapshot(),
                target_net=effective_spend,
                withdrawal_rate=wr,
            )
        )

        # Stop if portfolio is depleted.
        if ending_total < 1.0 and plan.shortfall > 0:
            break

    return results


def summarize(results: List[YearResult]) -> dict:
    """Cross-cutting metrics for a single scenario."""
    if not results:
        return {}
    total_tax = sum(r.plan.federal_tax for r in results)
    total_aca = sum(r.plan.healthcare_oop for r in results)
    total_penalty = sum(r.plan.penalty for r in results)
    total_conversions = sum(r.plan.conversion for r in results)
    total_shortfall = sum(r.plan.shortfall for r in results)
    total_state_tax = sum(r.plan.state_tax for r in results)
    total_scheduled_income = sum(r.plan.scheduled_income for r in results)
    total_scheduled_expense = sum(r.plan.scheduled_expense for r in results)
    last = results[-1]
    return {
        "ending_total": last.ending_total,
        "ending_age": last.age,
        "years_funded": len(results),
        "depleted": last.ending_total < 1.0,
        "total_federal_tax": total_tax,
        "total_healthcare_oop": total_aca,
        "total_penalty": total_penalty,
        "total_conversions": total_conversions,
        "total_shortfall": total_shortfall,
        "total_state_tax": total_state_tax,
        "total_scheduled_income": total_scheduled_income,
        "total_scheduled_expense": total_scheduled_expense,
    }
