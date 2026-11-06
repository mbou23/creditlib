"""Business-day calendars, roll conventions and tenor arithmetic."""
from __future__ import annotations

import calendar as _cal
import re
from datetime import date, timedelta
from enum import Enum
from typing import Iterable, Optional


class BusinessDayConvention(Enum):
    UNADJUSTED = "Unadjusted"
    FOLLOWING = "Following"
    MODIFIED_FOLLOWING = "Modified Following"
    PRECEDING = "Preceding"


class Calendar:
    """A business-day calendar: weekends plus an optional set of holidays.

    Market calendars (TARGET, NYC, LON...) can be built by passing their
    holiday lists; the default is weekends only, which is what most credit
    pricing tests in the literature use.
    """

    def __init__(self, holidays: Optional[Iterable[date]] = None, name: str = "WeekendsOnly"):
        self.holidays = frozenset(holidays or ())
        self.name = name

    def __repr__(self) -> str:
        return f"Calendar({self.name!r}, {len(self.holidays)} holidays)"

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def adjust(self, d: date, convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING) -> date:
        if convention is BusinessDayConvention.UNADJUSTED or self.is_business_day(d):
            return d
        if convention is BusinessDayConvention.PRECEDING:
            return self._roll(d, -1)
        following = self._roll(d, +1)
        if convention is BusinessDayConvention.MODIFIED_FOLLOWING and following.month != d.month:
            return self._roll(d, -1)
        return following

    def add_business_days(self, d: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        while remaining:
            d += timedelta(days=step)
            if self.is_business_day(d):
                remaining -= 1
        return d

    def _roll(self, d: date, step: int) -> date:
        while not self.is_business_day(d):
            d += timedelta(days=step)
        return d


WEEKENDS_ONLY = Calendar()


def add_months(d: date, months: int, end_of_month: bool = False) -> date:
    """Add calendar months, clipping to the last day of the target month.

    With ``end_of_month=True`` a date on the last day of its month maps to the
    last day of the target month (e.g. 28-Feb-2026 + 1M -> 31-Mar-2026).
    """
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last_day = _cal.monthrange(year, month)[1]
    if end_of_month and d.day == _cal.monthrange(d.year, d.month)[1]:
        return date(year, month, last_day)
    return date(year, month, min(d.day, last_day))


_TENOR_RE = re.compile(r"^\s*(\d+)\s*([DWMY])\s*$", re.IGNORECASE)


def add_tenor(d: date, tenor: str) -> date:
    """Add a tenor string such as ``'3M'``, ``'5Y'``, ``'2W'`` or ``'10D'``."""
    m = _TENOR_RE.match(tenor)
    if not m:
        raise ValueError(f"Invalid tenor: {tenor!r}")
    n, unit = int(m.group(1)), m.group(2).upper()
    if unit == "D":
        return d + timedelta(days=n)
    if unit == "W":
        return d + timedelta(weeks=n)
    if unit == "M":
        return add_months(d, n)
    return add_months(d, 12 * n)
