# Early Retirement Withdrawal Planner

Tax- and penalty-optimized withdrawal simulator for early retirees. Tuned for a single
filer in Washington State on ACA marketplace healthcare. Real-dollar, year-by-year
simulation with deterministic, Monte Carlo (lognormal), and historical (Shiller) modes.

## Quickstart

```sh
pip install --user -r requirements.txt   # add --break-system-packages on PEP-668 systems
streamlit run app.py
pytest                                    # tax-engine + strategy + streams tests
```

The app loads `scenarios/default.json` as the startup scenario. To use your own:
copy or rename your scenario file to `scenarios/default.local.json` (gitignored) and the
loader will prefer it. Otherwise, use the in-app sidebar to import/export JSON.

## Layout

- `planner/tax.py` — federal brackets, LTCG stacking, ACA subsidy, Medicare + IRMAA, RMDs, taxable-SS rule, WA cap-gains
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

## Strategies

- `bridge_optimal` — convert Trad→Roth up to lesser of (0% LTCG ceiling, 400% FPL ceiling), floor at standard deduction
- `bridge_guarded` — like `bridge_optimal` but caps each year's conversion at `traditional_balance / years_to_60` so bracket-driven conversion can't drain Trad faster than the bridge years remaining (sequence-of-returns insurance)
- `minimal_convert` — fill the standard deduction only (~$15.7k/yr)
- `aggressive_convert` — fill the top of the 12% bracket every year regardless of LTCG impact
- `custom` — user-specified annual conversion amount

## Returns models

- **Constant** — single deterministic 7%/2%/0% real (default for the single-scenario tab)
- **Lognormal** — synthetic, configurable σ + correlation; runs N stochastic paths
- **Historical (rolling start)** — replays each calendar start year 1928 onward through the horizon; n_paths bounded by data coverage and horizon length. Path-inspector lets you pick worst-failing/median/best/by-index and see the underlying account-balance trajectory.

## Key assumptions

- Single filer; brackets are 2026 estimates (2025 actuals projected forward)
- All values are real (inflation-adjusted) dollars
- Growth applied at start of year, then withdrawal/conversion/RMD
- 10% early-withdrawal penalty on Traditional pre-age-60 (proxy for 59.5)
- HSA non-medical withdrawals modeled as ordinary income only (20% pre-65 penalty omitted)
- WA capital gains tax (7% above $262k LTCG) **is** modeled (`state_tax` field)
- Social Security taxed using IRS provisional-income test (0/50/85% inclusion)
- Medicare IRMAA uses current-year MAGI (IRS actually uses MAGI from 2 years prior)
- ACA: IRA-extended sliding-scale schedule (`cap`) or pre-IRA cliff (`cliff`) on $8k benchmark unsubsidized premium
- Historical mode: real cash held at 0% (Shiller doesn't publish T-bill); edit the CSV to override
