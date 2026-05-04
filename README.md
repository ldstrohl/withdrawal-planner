# Early Retirement Withdrawal Planner

Tax- and penalty-optimized withdrawal simulator for early retirees. Single or
married-filing-jointly on ACA marketplace healthcare; pluggable state-tax model
(9 no-tax states + WA + CUSTOM). Real-dollar, year-by-year simulation with
deterministic, Monte Carlo (lognormal), and historical (Shiller) returns modes.

## Quickstart

Latest deployment:
https://fi-withdrawal-planner.streamlit.app/ 


Local installation:
```sh
git clone https://github.com/ldstrohl/withdrawal-planner.git
cd withdrawal-planner
pip install --user -r requirements.txt   # add --break-system-packages on PEP-668 systems
streamlit run app.py
```
The app launches a local server accessed in a browser at localhost

The app loads `scenarios/default.json` as the startup scenario. To use your own:
copy or rename your scenario file to `scenarios/default.local.json` (gitignored) and the
loader will prefer it. Otherwise, use the in-app sidebar to import/export JSON.

## Layout

- `planner/tax.py` — federal brackets, LTCG stacking, ACA subsidy, Medicare + IRMAA, RMDs, taxable-SS rule
- `planner/state_tax.py` — pluggable state-tax presets and flat-rate engine
- `planner/accounts.py` — account types and Roth ladder bookkeeping
- `planner/strategy.py` — preset withdrawal/conversion strategies, source priority, year planner with iterative tax fixed-point
- `planner/streams.py` — generic scheduled income / expense streams (rentals, pensions, tuition, LTC, etc.)
- `planner/returns.py` — pluggable returns models (`ConstantReturns`, `LognormalReturns`, `HistoricalPlayback`)
- `planner/montecarlo.py` — multi-path orchestration + percentile aggregation
- `planner/simulate.py` — year-by-year orchestration, real-dollar growth, results recording
- `charts.py` — Plotly chart builders + per-year DataFrame
- `app.py` — Streamlit GUI (sidebar inputs, single scenario, comparison, MC, how-it-works, assumptions)
- `data/historical_real_returns.csv` — annual real returns 1871–2022 (S&P + 10y bond, Shiller)
- `scripts/build_historical_csv.py` — refetches Shiller's `ie_data.xls` and rebuilds the CSV
- `scenarios/default.json` — generic median-FIRE persona startup scenario
- `tests/test_tax.py` — pinned tax-engine, strategy, and streams math

## Accumulation

Model pre-retirement saving and one-time or recurring expense shocks before the
retirement phase begins.

Inputs: `current_age`, `retirement_age`, `annual_savings`, `savings_allocation`,
`accumulation_wage_income`.

Savings are entered as **net dollars** (after income tax and FICA); `accumulation_wage_income`
is used only as the LTCG stacking baseline when a taxable sale is forced by an expense shock —
not for full payroll modeling.

Expense overflow order: contribution pool → cash → taxable brokerage (with LTCG gross-up).
If cash and taxable are exhausted, the year is recorded as a shortfall.
See the "How it works" tab for the detailed per-year logic, and `ROADMAP.md` for deferred
features (full payroll mode, tax-advantaged depletion ordering, contribution limits).

## Strategies

- `bridge_optimal` — convert Trad→Roth up to lesser of (0% LTCG ceiling, 400% FPL ceiling), floor at standard deduction
- `bridge_guarded` — like `bridge_optimal` but caps each year's conversion at `traditional_balance / years_to_60` so bracket-driven conversion can't drain Trad faster than the bridge years remaining (sequence-of-returns insurance)
- `bridge_responsive` — scales between `minimal_convert` (no drawdown) and `bridge_optimal` (≥20% drawdown from peak) by current drawdown. Converts cheaply when prices are depressed; preserves Trad and avoids tax-prepay during good runs. Targets preservation, not depletion.
- `minimal_convert` — fill the standard deduction only (~$15.7k/yr)
- `aggressive_convert` — fill the top of the 12% bracket every year regardless of LTCG impact
- `custom` — user-specified annual conversion amount

## Returns models

- **Constant** — single deterministic 7%/2%/0% real (default for the single-scenario tab)
- **Lognormal** — synthetic, configurable σ + correlation; runs N stochastic paths
- **Historical (rolling start)** — replays each calendar start year 1928 onward through the horizon; n_paths bounded by data coverage and horizon length (shorter horizons yield more sequences — a 45y horizon gives ~51 paths vs ~36 at 60y). Path-inspector lets you pick worst-failing/median/best/by-index and see the underlying account-balance trajectory.

## Spending input

Spend can be entered two ways:
- **Fixed annual amount** — real-dollar number, constant in real terms across all years.
- **% of starting portfolio (SWR)** — engine resolves to dollars at sim start as
  `starting_total × spend_rate` and uses that constant amount throughout. Pick this when
  you want to think in 4%-rule / 3.5% / lean-FIRE terms.

## State tax

Pluggable. Pre-configured presets:
- 9 no-income-tax states: TX, FL, NV, AK, NH, SD, TN, WY, plus NONE
- WA — the existing 7% LTCG above $262k rule
- CUSTOM — flat marginal-rate approximation; you enter ordinary rate, LTCG rate, and
  optional LTCG threshold

The CUSTOM model intentionally ignores per-state brackets, SS exemptions, pension
exclusions, and itemized deductions. Use it as a conservative bound for cross-state
planning, not for filing prep.

## Key assumptions

- Single or MFJ filing status; federal brackets are 2026 estimates (2025 actuals projected forward)
- All values are real (inflation-adjusted) dollars
- Growth applied at start of year, then withdrawal/conversion/RMD
- 10% early-withdrawal penalty on Traditional pre-age-60 (proxy for 59.5)
- HSA non-medical withdrawals modeled as ordinary income only (20% pre-65 penalty omitted)
- State tax: pluggable per scenario (see "State tax" section above)
- Social Security taxed using IRS provisional-income test (0/50/85% inclusion)
- Medicare IRMAA uses current-year MAGI (IRS actually uses MAGI from 2 years prior)
- ACA: IRA-extended sliding-scale schedule (`cap`) or pre-IRA cliff (`cliff`) on $8k benchmark unsubsidized premium
- Historical mode: real cash held at 0% (Shiller doesn't publish T-bill); edit the CSV to override
- MFJ: doubles standard deduction, widens ordinary + LTCG brackets, uses household-of-2 FPL,
  MFJ SS thresholds (32k/44k), and 2 Medicare premiums post-65 with MFJ IRMAA tiers. The
  `ss_annual_benefit` field is interpreted as the combined household amount; model spouse 2
  separately as an income stream if claim ages differ.
