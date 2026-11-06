"""Fixed-rate bullet bond."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from ..dates.calendar import WEEKENDS_ONLY, BusinessDayConvention, Calendar
from ..dates.daycount import DayCount, year_fraction
from ..dates.schedule import Period, generate_schedule


@dataclass(frozen=True)
class FixedRateBond:
    """A fixed-coupon bullet bond.

    Coupon amounts are ``coupon * face * year_fraction(accrual_start, accrual_end)``
    on unadjusted accrual dates, so under 30/360 a regular period pays exactly
    ``coupon / frequency * face``.
    """

    issue_date: date
    maturity: date
    coupon: float
    frequency: int = 2
    day_count: DayCount = DayCount.THIRTY_360
    face: float = 100.0
    calendar: Calendar = field(default=WEEKENDS_ONLY, compare=False)
    payment_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING

    @property
    def schedule(self) -> List[Period]:
        return generate_schedule(
            self.issue_date, self.maturity, self.frequency, self.calendar, self.payment_convention
        )

    def cashflows(self, after: Optional[date] = None) -> List[Tuple[Period, float]]:
        """``(period, amount)`` for each coupon, with the principal added to the last one.

        If ``after`` is given, only coupons whose accrual period ends strictly
        after it are returned (a bond settling on a coupon date does not receive
        that coupon). This keeps cashflows consistent with :meth:`accrued_interest`.
        """
        periods = self.schedule
        flows = []
        for i, p in enumerate(periods):
            amount = self.coupon * self.face * year_fraction(p.accrual_start, p.accrual_end, self.day_count)
            if i == len(periods) - 1:
                amount += self.face
            if after is None or p.accrual_end > after:
                flows.append((p, amount))
        return flows

    def current_period(self, settlement: date) -> Period:
        """The accrual period containing ``settlement`` (start < settlement <= end)."""
        for p in self.schedule:
            if p.accrual_start <= settlement < p.accrual_end:
                return p
        raise ValueError(f"settlement {settlement} is outside the bond's life")

    def accrued_interest(self, settlement: date) -> float:
        p = self.current_period(settlement)
        return self.coupon * self.face * year_fraction(p.accrual_start, settlement, self.day_count)
