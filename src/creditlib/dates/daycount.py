"""Day count conventions.

Each convention maps a pair of dates to a year fraction. Conventions follow the
ISDA 2006 definitions.
"""
from __future__ import annotations

import calendar as _cal
from datetime import date
from enum import Enum


class DayCount(Enum):
    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"          # 30/360 US (Bond Basis), ISDA 2006 4.16(f)
    THIRTY_E_360 = "30E/360"       # Eurobond basis, ISDA 2006 4.16(g)
    ACT_ACT_ISDA = "ACT/ACT ISDA"

    def year_fraction(self, start: date, end: date) -> float:
        return year_fraction(start, end, self)


def _thirty_360(start: date, end: date) -> float:
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30
    return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360.0


def _thirty_e_360(start: date, end: date) -> float:
    d1, d2 = min(start.day, 30), min(end.day, 30)
    return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360.0


def _act_act_isda(start: date, end: date) -> float:
    if start == end:
        return 0.0
    if start > end:
        return -_act_act_isda(end, start)
    total = 0.0
    current = start
    while current.year < end.year:
        year_end = date(current.year + 1, 1, 1)
        total += (year_end - current).days / (366.0 if _cal.isleap(current.year) else 365.0)
        current = year_end
    total += (end - current).days / (366.0 if _cal.isleap(end.year) else 365.0)
    return total


def year_fraction(start: date, end: date, day_count: DayCount) -> float:
    """Year fraction between ``start`` and ``end`` under ``day_count``."""
    if day_count is DayCount.ACT_360:
        return (end - start).days / 360.0
    if day_count is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if day_count is DayCount.THIRTY_360:
        return _thirty_360(start, end)
    if day_count is DayCount.THIRTY_E_360:
        return _thirty_e_360(start, end)
    if day_count is DayCount.ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    raise ValueError(f"Unsupported day count: {day_count}")
