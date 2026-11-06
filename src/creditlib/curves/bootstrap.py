"""Bootstrap a single discount curve from money-market deposits and par swaps.

This is a deliberately simple single-curve set-up (no OIS/IBOR basis), similar
to the yield curve used by the ISDA CDS Standard Model:

* deposits: simple interest ACT/360, ``DF(T) = 1 / (1 + r * tau)``;
* swaps: par fixed-vs-floating, fixed leg on a regular schedule, valued as
  ``rate * sum(tau_i * DF(t_i)) = 1 - DF(T)``.

Pillars are solved one at a time; the curve is flat-forward between pillars.
"""
from __future__ import annotations

from datetime import date
from typing import List, Sequence, Tuple

from scipy.optimize import brentq

from ..dates.calendar import WEEKENDS_ONLY, BusinessDayConvention, Calendar, add_tenor
from ..dates.daycount import DayCount, year_fraction
from ..dates.schedule import generate_schedule
from .discount import DiscountCurve


def bootstrap_discount_curve(
    reference_date: date,
    deposits: Sequence[Tuple[str, float]] = (),
    swaps: Sequence[Tuple[str, float]] = (),
    deposit_day_count: DayCount = DayCount.ACT_360,
    swap_day_count: DayCount = DayCount.THIRTY_360,
    swap_frequency: int = 2,
    calendar: Calendar = WEEKENDS_ONLY,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> DiscountCurve:
    """Build a :class:`DiscountCurve` from ``[(tenor, rate), ...]`` deposit and swap quotes.

    Instruments start on ``reference_date`` (spot lag ignored). Quotes must
    produce strictly increasing maturities: deposits first, then swaps.
    """
    times: List[float] = []
    forwards: List[float] = []

    def curve_with(last_forward: float, t: float) -> DiscountCurve:
        return DiscountCurve(reference_date, times + [t], forwards + [last_forward])

    def add_pillar(t: float, objective) -> None:
        if times and t <= times[-1]:
            raise ValueError("instrument maturities must be strictly increasing")
        forwards.append(brentq(objective, -0.5, 2.0, xtol=1e-14))
        times.append(t)

    for tenor, rate in deposits:
        maturity = calendar.adjust(add_tenor(reference_date, tenor), convention)
        t = year_fraction(reference_date, maturity, DayCount.ACT_365F)
        target_df = 1.0 / (1.0 + rate * year_fraction(reference_date, maturity, deposit_day_count))
        add_pillar(t, lambda f, t=t, target=target_df: curve_with(f, t).df(t) - target)

    for tenor, rate in swaps:
        end = add_tenor(reference_date, tenor)
        periods = generate_schedule(reference_date, end, swap_frequency, calendar, convention)
        maturity = periods[-1].payment_date
        t = year_fraction(reference_date, maturity, DayCount.ACT_365F)

        def swap_npv(f: float, t=t, periods=periods, rate=rate) -> float:
            curve = curve_with(f, t)
            annuity = sum(
                year_fraction(p.accrual_start, p.accrual_end, swap_day_count) * curve.df(p.payment_date)
                for p in periods
            )
            return rate * annuity - (1.0 - curve.df(maturity))

        add_pillar(t, swap_npv)

    if not times:
        raise ValueError("need at least one deposit or swap quote")
    return DiscountCurve(reference_date, times, forwards)
