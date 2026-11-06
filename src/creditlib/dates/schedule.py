"""Coupon schedules for bonds and standard (post-2009 "Big Bang") CDS."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

from .calendar import WEEKENDS_ONLY, BusinessDayConvention, Calendar, add_months


@dataclass(frozen=True)
class Period:
    """One accrual period: accrues from ``accrual_start`` to ``accrual_end``, paid on ``payment_date``."""

    accrual_start: date
    accrual_end: date
    payment_date: date


def generate_schedule(
    start: date,
    end: date,
    frequency: int,
    calendar: Calendar = WEEKENDS_ONLY,
    payment_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    end_of_month: bool = False,
) -> List[Period]:
    """Regular schedule generated backward from ``end`` (any stub is a short front stub).

    ``frequency`` is the number of payments per year (1, 2, 4 or 12). Accrual
    dates are left unadjusted, as is standard for fixed-rate bonds; payment
    dates are adjusted with ``payment_convention``.
    """
    if frequency not in (1, 2, 4, 12):
        raise ValueError("frequency must be 1, 2, 4 or 12")
    if end <= start:
        raise ValueError("end must be after start")
    step = 12 // frequency
    dates = [end]
    k = 1
    while True:
        d = add_months(end, -step * k, end_of_month)
        if d <= start:
            break
        dates.append(d)
        k += 1
    dates.append(start)
    dates.reverse()
    return [
        Period(a, b, calendar.adjust(b, payment_convention))
        for a, b in zip(dates[:-1], dates[1:])
    ]


# --------------------------------------------------------------------------- #
# Standard CDS dates
# --------------------------------------------------------------------------- #
_IMM_MONTHS = (3, 6, 9, 12)


def previous_imm_date(d: date) -> date:
    """Latest quarterly CDS date (20-Mar/Jun/Sep/Dec) on or before ``d``."""
    month_start = date(d.year, d.month, 1)
    for back in range(0, 4):
        m = add_months(month_start, -back)
        if m.month in _IMM_MONTHS and date(m.year, m.month, 20) <= d:
            return date(m.year, m.month, 20)
    raise AssertionError("unreachable: an IMM date always lies within the last 4 months")


def next_imm_date(d: date) -> date:
    """Earliest quarterly CDS date strictly after ``d``."""
    prev = previous_imm_date(d)
    return add_months(prev, 3)


def cds_standard_maturity(trade_date: date, tenor_years: int) -> date:
    """Maturity of a standard CDS under the semi-annual roll rule (in force since Dec-2015).

    Contracts roll on 20-Mar and 20-Sep. A trade on or after 20-Mar (and before
    20-Sep) matures on 20-Jun, ``tenor_years`` later; a trade on or after
    20-Sep matures on 20-Dec; a trade before 20-Mar still belongs to the
    previous December series.
    """
    y = trade_date.year
    if trade_date < date(y, 3, 20):
        return date(y - 1 + tenor_years, 12, 20)
    if trade_date < date(y, 9, 20):
        return date(y + tenor_years, 6, 20)
    return date(y + tenor_years, 12, 20)


def cds_accrual_start(trade_date: date, calendar: Calendar = WEEKENDS_ONLY) -> date:
    """Accrual start of a standard CDS: the previous IMM date on or before the step-in date (T+1), adjusted Following."""
    step_in = trade_date + timedelta(days=1)
    return calendar.adjust(previous_imm_date(step_in), BusinessDayConvention.FOLLOWING)


def cds_schedule(accrual_start: date, maturity: date, calendar: Calendar = WEEKENDS_ONLY) -> List[Period]:
    """Quarterly premium schedule of a standard CDS (ISDA Standard Model conventions).

    Dates roll backward from ``maturity`` on the 20th; interior dates are adjusted
    Following. The final period accrues through the maturity date inclusive, so
    its ``accrual_end`` is ``maturity + 1 day``; it is paid on maturity adjusted.
    """
    unadjusted = [maturity]
    k = 1
    while True:
        d = add_months(maturity, -3 * k)
        if d <= accrual_start:
            break
        unadjusted.append(d)
        k += 1
    unadjusted.reverse()

    periods: List[Period] = []
    start = accrual_start
    for i, d in enumerate(unadjusted):
        is_last = i == len(unadjusted) - 1
        if is_last:
            end = maturity + timedelta(days=1)
            pay = calendar.adjust(maturity, BusinessDayConvention.FOLLOWING)
        else:
            end = calendar.adjust(d, BusinessDayConvention.FOLLOWING)
            pay = end
        periods.append(Period(start, end, pay))
        start = end
    return periods
