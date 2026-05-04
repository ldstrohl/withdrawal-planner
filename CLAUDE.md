# CLAUDE.md

Tax- and penalty-optimized early-retirement withdrawal simulator. Streamlit UI on top of a pure-Python year-by-year engine. Single-file UI (`app.py`), engine modules under `planner/`.

## Run / test

```sh
streamlit run app.py        # local UI on http://localhost:8501
pytest                      # one test file: tests/test_tax.py
python scripts/build_historical_csv.py   # refetch Shiller real-returns CSV
```

Deps: see `requirements.txt`. Live deployment: https://fi-withdrawal-planner.streamlit.app/

## Architecture

Year-by-year deterministic engine, optionally wrapped by a Monte Carlo / historical-playback orchestrator. Real (inflation-adjusted) dollars throughout.

```
app.py (Streamlit sidebar + 4 tabs)
  └─ planner.simulate.simulate(inputs, returns_model) -> List[YearResult]
       ├─ portfolio.apply_growth(returns)               # accounts.py
       ├─ planner.strategy.plan_year(...)                # fixed-point tax solver
       │    ├─ _conversion_for_strategy(name, ...)       # 6 presets
       │    └─ _fund_priority(portfolio, age, ...)       # withdrawal order
       ├─ _apply_action(...)                             # mutates portfolio
       └─ records YearResult(year, age, snapshot, plan, ...)
  └─ planner.montecarlo.run_monte_carlo(...)             # N-path orchestration
  └─ charts.py                                           # Plotly + per_year_table
```

**Per-year order of operations** (`simulate.py:151–212`): growth → SS/RMD/streams → `plan_year` (iterative tax fixed-point, ≤40 iters, damped 50/50) → apply withdrawals/conversions → record snapshot.

**Withdrawal priority** (`strategy.py:126–192`): age-gated.
- Pre-60: cash → taxable → seasoned Roth → Roth contributions → Traditional (10% penalty) → HSA
- Post-60: cash → taxable → any Roth → Traditional → HSA

**Strategies** (preset names; `strategy.py:64–120`): `bridge_optimal`, `bridge_guarded`, `bridge_responsive`, `minimal_convert`, `aggressive_convert`, `custom`.

## Data shapes

**`SimulationInputs`** (`simulate.py`): account balances, `target_spend` or `spend_rate`+`spend_mode`, returns, `start_age`, `horizon_years`, `ss_*`, `strategy`, `aca_mode`, `filing_status`, `state_*`, `income_streams`, `expense_streams`.

**`YearResult`** (`simulate.py:25–34`): `year`, `age`, `starting_total`, `ending_total`, `plan: PlanResult`, `snapshot: dict`, `target_net`, `withdrawal_rate`.

**`snapshot` keys** (`accounts.py:222–234`): `cash`, `taxable`, `taxable_basis`, `traditional`, `roth`, `roth_seasoned`, `roth_contributions`, `roth_earnings`, `hsa`, `total`. Charts depend on these key names.

**Scenario JSON** (`scenarios/default.json`): 31 keys covering balances, spend, returns, timing, strategy, tax/ACA, streams. Loader prefers `scenarios/default.local.json` (gitignored) over `scenarios/default.json`.

## Conventions

- **All values are real dollars.** Don't add nominal-dollar handling without converting throughout the pipeline.
- **Two-phase timeline.** `current_age` and `retirement_age` are independent. When `age < retirement_age`, the engine runs an accumulation branch (growth → contributions/expenses, no `plan_year` / strategies / ACA / Medicare / RMDs / SS). When `age >= retirement_age`, the existing retirement loop runs unchanged. If `current_age == retirement_age` (default), the engine reproduces retirement-only behavior bit-for-bit.
- **Retirement trigger.** `retirement_mode` controls when accumulation ends. `"fixed"` (default) — retire at `retirement_age` exactly; bit-for-bit unchanged from legacy. `"target_nw"` — retire the year after end-of-year `portfolio.total >= retirement_target_nw`, gated by `retirement_age_floor` (earliest age allowed; `None` = no floor) and capped by `retirement_age` (ceiling: forces retirement even if target not hit). `summarize()` exposes `actual_retirement_age` (first retirement-phase `age`, or `None` if the horizon never reaches retirement).
- **Age 60 is the bridge boundary**, not 59.5 — the engine uses `age >= 60` as a single cliff for both Trad penalty and Roth-ladder seasoning logic. Don't introduce 59.5 fractional ages.
- **Roth ladder bookkeeping** lives in `RothIRA.rungs` (year_converted + amount); 5-year seasoning is enforced in `withdraw_seasoned()`.
- **Tax engine handles distributions only** — no W-2 wages, no FICA. Adding wage income requires extending `federal_tax` inputs, not retrofitting `IncomeStream(taxable=True)` (which only offsets gross need; it does not flow through brackets correctly for payroll).
- **MFJ doubles brackets** including standard deduction, FPL household-of-2, IRMAA tiers, and SS thresholds (32k/44k). `ss_annual_benefit` is the combined household amount in MFJ.
- **State tax is a flat-rate approximation** — pluggable presets in `state_tax.py` plus CUSTOM. Not for filing prep.

## When editing

- Withdrawal logic → `strategy.py` (touch `_fund_priority` and `plan_year`; the fixed-point solver is sensitive — preserve the damped update and convergence tolerance).
- New tax rules → `tax.py`; federal brackets are 2026 estimates at lines 36–42.
- New account types → extend `accounts.py` *and* update `Portfolio.snapshot()` keys consumers rely on, *and* add chart series in `charts.py`.
- New scenario fields → add to `SimulationInputs`, the sidebar in `app.py:render_sidebar`, the JSON save/load round-trip, and `scenarios/default.json`.
- New strategy preset → add to `STRATEGY_PRESETS` and `_conversion_for_strategy`, then document in the "How it works" tab (`app.py:824+`) and the README `Strategies` section.

## Tests

Single file: `tests/test_tax.py`. Pattern: build `SimulationInputs` → `simulate(...)` → assert on `results[i].plan.*` or `summarize(results).*`. Monte Carlo smoke tests via `run_monte_carlo(...)` checking `success_rate` / `preservation_rate`. Add new test cases here rather than creating new files unless the count justifies splitting.

## Known non-features

- Accumulation runs in **simple-tax mode** only — savings are entered as net dollars; FICA, federal/state income tax on wages, and tax-advantaged depletion ordering for accumulation expense shocks are deferred to `ROADMAP.md`.
- No nominal-dollar mode.
- No HSA 20% pre-65 non-medical penalty (treated as ordinary income only).
- No state-bracket modeling (flat rate approximation only).
- IRMAA uses current-year MAGI rather than the IRS 2-year lookback.
