"""Single-name credit default swap (standard ISDA contract)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from ..dates.calendar import WEEKENDS_ONLY, Calendar
from ..dates.daycount import DayCount
from ..dates.schedule import Period, cds_accrual_start, cds_schedule, cds_standard_maturity


@dataclass(frozen=True)
class CDS:
    """A CDS trade, seen from the protection buyer when ``notional > 0``.

    ``coupon`` is the running spread paid quarterly (ACT/360), e.g. 0.01 for the
    100bp standard coupon. A negative notional is a protection seller.
    """

    trade_date: date
    maturity: date
    coupon: float
    notional: float = 1.0
    accrual_start: Optional[date] = None
    day_count: DayCount = DayCount.ACT_360
    calendar: Calendar = field(default=WEEKENDS_ONLY, compare=False)

    @classmethod
    def standard(
        cls, trade_date: date, tenor_years: int, coupon: float, notional: float = 1.0,
        calendar: Calendar = WEEKENDS_ONLY,
    ) -> "CDS":
        """Standard contract: IMM maturity (semi-annual roll) and accrual from the previous IMM date."""
        return cls(
            trade_date=trade_date,
            maturity=cds_standard_maturity(trade_date, tenor_years),
            coupon=coupon,
            notional=notional,
            calendar=calendar,
        )

    @property
    def effective_accrual_start(self) -> date:
        return self.accrual_start or cds_accrual_start(self.trade_date, self.calendar)

    @property
    def schedule(self) -> List[Period]:
        return cds_schedule(self.effective_accrual_start, self.maturity, self.calendar)

    def with_coupon(self, coupon: float) -> "CDS":
        return CDS(self.trade_date, self.maturity, coupon, self.notional, self.accrual_start,
                   self.day_count, self.calendar)
