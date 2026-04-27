"""Scheduled income and expense streams.

A unified abstraction for SS-like cash flows the user can stack: rental net income,
pension, planned downsize sale, kid tuition, long-term-care, etc. Each stream is
active over an inclusive age window [start_age, end_age] and contributes a fixed
real-dollar annual amount.

Income streams optionally feed federal ordinary income (taxable=True). Expense
streams always increase the year's gross spending need; they have no tax effect
(itemized deductions aren't modeled; standard deduction is assumed).

Social Security stays as its own special-cased input because it uses the IRS
provisional-income test for partial taxation; everything else is either fully
taxable or not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class IncomeStream:
    name: str
    annual_amount: float    # real $
    start_age: int
    end_age: int            # inclusive
    taxable: bool = True    # if True, adds to federal ordinary income


@dataclass(frozen=True)
class ExpenseStream:
    name: str
    annual_amount: float    # real $
    start_age: int
    end_age: int            # inclusive


def active_income(streams: Iterable[IncomeStream], age: int) -> Tuple[float, float]:
    """Return (total_income, taxable_portion) of income streams active at this age."""
    total = 0.0
    taxable = 0.0
    for s in streams:
        if s.start_age <= age <= s.end_age:
            total += s.annual_amount
            if s.taxable:
                taxable += s.annual_amount
    return total, taxable


def active_expense(streams: Iterable[ExpenseStream], age: int) -> float:
    """Return sum of expense-stream amounts active at this age."""
    return sum(s.annual_amount for s in streams if s.start_age <= age <= s.end_age)
