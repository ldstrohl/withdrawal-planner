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
