# Roadmap

Tracking deferred enhancements. Items are unchecked until implemented.

## Accumulation phase

- [ ] Per-field `wage_growth` for accumulation years (currently flat real wages)
- [ ] Tax-advantaged depletion ordering for accumulation expense shocks (currently fail after Cash → Taxable)
- [ ] IRS contribution limits as hard caps with catch-up rules (currently soft warnings)
- [ ] Roth IRA MAGI-based contribution phaseout
- [ ] Employer match: vesting schedules and per-paycheck true-up
- [ ] Full payroll-tax mode (FICA, federal/state income tax on wages, AGI-driven side effects)
- [ ] Phase-aware stream semantics (today: income streams deposit during accumulation, offset need during retirement; expense streams reduce contributions during accumulation, increase need during retirement — make the dual-meaning explicit in UI labels)

## Tax engine

- [ ] Nominal-dollar mode (would touch every module)
- [ ] HSA 20% pre-65 non-medical penalty
- [ ] State-bracket modeling (currently flat-rate approximation)
- [ ] IRMAA 2-year MAGI lookback (currently uses current-year MAGI)
- [ ] Wage income post-retirement (partial retirement / consulting)

## UX

- [ ] Phase marker / shaded region on time-series charts
- [ ] Sensitivity analysis (savings rate vs retirement date trade space)

## Build & deploy

- [ ] Unpin Python 3.12 in `runtime.txt` once `planner/simulate.py` is 3.14-compatible. PR #4 pinned because Streamlit Cloud on 3.14 fails `SimulationInputs.__init__() got an unexpected keyword argument 'current_age'` despite the field being declared. Suspect: `from __future__ import annotations` + `@dataclass(frozen=True)` + `Optional[int]` interaction with PEP 649 deferred annotations. Try removing the `__future__` import or switching `Optional[int]` → `int | None` and verify on a 3.14 venv.
