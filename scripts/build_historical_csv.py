"""Fetch Shiller's monthly stock+bond data and write annual real returns CSV.

Source: Robert J. Shiller, http://www.econ.yale.edu/~shiller/data/ie_data.xls
The xls is the dataset behind "Irrational Exuberance" — monthly S&P composite,
dividends, CPI, long-term yield, and pre-built real total return indices for
stocks and bonds. We compute year-over-year (Jan-to-Jan) real returns from the
level series, giving annual real returns for every full calendar year in the file.

Cash: Shiller doesn't publish T-bill returns. We hold real cash at 0%, matching
the planner default. Users who care can edit the CSV.

Run:  python3 scripts/build_historical_csv.py
Out:  data/historical_real_returns.csv  (columns: year, real_stocks, real_bonds, real_cash)
"""
from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

import xlrd

URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
CACHE = Path("/tmp/shiller_ie_data.xls")
OUT = Path(__file__).parent.parent / "data" / "historical_real_returns.csv"

COL_DATE = 0
COL_REAL_TR_STOCKS = 9         # "Real Total Return Price" — level series
COL_REAL_TR_BONDS = 18         # "Real Total Bond Returns" — level series


def fetch() -> None:
    if CACHE.exists() and CACHE.stat().st_size > 100_000:
        return
    print(f"downloading {URL} -> {CACHE}")
    urllib.request.urlretrieve(URL, CACHE)


def jan_levels() -> dict[int, tuple[float, float]]:
    """Return {year: (real_stock_level, real_bond_level)} for every January in the file."""
    wb = xlrd.open_workbook(CACHE)
    sh = wb.sheet_by_name("Data")
    levels: dict[int, tuple[float, float]] = {}
    for r in range(8, sh.nrows):
        date = sh.cell_value(r, COL_DATE)
        if not isinstance(date, float):
            continue
        # Date is encoded as YYYY.MM where MM in [01..12]; .01 = January.
        year = int(date)
        month = round((date - year) * 100)
        if month != 1:
            continue
        s = sh.cell_value(r, COL_REAL_TR_STOCKS)
        b = sh.cell_value(r, COL_REAL_TR_BONDS)
        if not isinstance(s, float) or not isinstance(b, float):
            continue
        levels[year] = (s, b)
    return levels


def annual_returns(levels: dict[int, tuple[float, float]]) -> list[tuple[int, float, float, float]]:
    """Year Y return = level(Jan Y+1) / level(Jan Y) - 1."""
    rows: list[tuple[int, float, float, float]] = []
    years = sorted(levels)
    for y in years:
        if y + 1 not in levels:
            continue
        s0, b0 = levels[y]
        s1, b1 = levels[y + 1]
        rows.append((y, s1 / s0 - 1.0, b1 / b0 - 1.0, 0.0))
    return rows


def main() -> None:
    fetch()
    levels = jan_levels()
    rows = annual_returns(levels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "real_stocks", "real_bonds", "real_cash"])
        for y, s, b, c in rows:
            w.writerow([y, f"{s:.6f}", f"{b:.6f}", f"{c:.6f}"])
    print(f"wrote {len(rows)} years -> {OUT}  ({rows[0][0]}-{rows[-1][0]})")


if __name__ == "__main__":
    main()
