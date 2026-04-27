# Early Retirement Withdrawal Planner

Tax- and penalty-optimized withdrawal simulator for early retirees. Tuned for a single
filer in Washington State on ACA marketplace healthcare. Real-dollar, year-by-year
simulation with side-by-side strategy comparison.

## Quickstart

```sh
pip install --user -r requirements.txt   # add --break-system-packages on PEP-668 systems
streamlit run app.py
pytest                                    # tax-engine tests
```

## Layout

- `planner/tax.py` — federal brackets, LTCG stacking, ACA subsidy, early-withdrawal penalty
- `planner/accounts.py` — account types and Roth ladder bookkeeping
- `planner/strategy.py` — preset withdrawal/conversion strategies, source priority, year planner
- `planner/simulate.py` — year-by-year orchestration, real-dollar growth, results recording
- `charts.py` — Plotly chart builders + per-year DataFrame
- `app.py` — Streamlit GUI (sidebar inputs, single-scenario tab, comparison tab, assumptions tab)
- `tests/test_tax.py` — pinned tax-engine math

## Strategies

- `bridge_optimal` — convert Trad→Roth up to lesser of (0% LTCG ceiling, 400% FPL), floor at standard deduction
- `minimal_convert` — fill the standard deduction only (~$15.7k/yr)
- `aggressive_convert` — fill the top of the 12% bracket every year regardless of LTCG impact
- `custom` — user-specified annual conversion amount

## Key assumptions

- Single filer; brackets are 2026 estimates (2025 actuals projected forward)
- All values are real (inflation-adjusted) dollars
- Growth applied at start of year, then withdrawal/conversion
- 10% early-withdrawal penalty on Traditional pre-age-60 (proxy for 59.5)
- HSA non-medical withdrawals modeled as ordinary income only (20% pre-65 penalty omitted)
- WA capital gains tax (7% above $262k LTCG) not modeled — irrelevant at expected draw
- ACA: IRA-extended sliding-scale schedule (`cap` mode) or pre-IRA cliff (`cliff` mode) on a $8k benchmark unsubsidized premium
