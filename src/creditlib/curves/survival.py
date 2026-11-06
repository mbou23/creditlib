"""Survival (credit) curves with piecewise-flat hazard rates."""
from __future__ import annotations

import math
from datetime import date

from .base import DateOrTime, PiecewiseConstantRateCurve


class SurvivalCurve(PiecewiseConstantRateCurve):
    """``rates`` are hazard rates: Q(t) = exp(-integral of hazard from 0 to t)."""

    @classmethod
    def flat(cls, reference_date: date, hazard_rate: float, horizon: float = 100.0) -> "SurvivalCurve":
        return cls(reference_date, [horizon], [hazard_rate])

    def survival_probability(self, d: DateOrTime) -> float:
        t = self.time(d)
        if t <= 0:
            return 1.0
        return math.exp(-self.integrated_rate(t))

    def q(self, d: DateOrTime) -> float:
        """Alias for :meth:`survival_probability`."""
        return self.survival_probability(d)

    def default_probability(self, d: DateOrTime) -> float:
        return 1.0 - self.survival_probability(d)

    def hazard_rate(self, d: DateOrTime) -> float:
        return self.rate_at(d)

    def shifted(self, shift: float) -> "SurvivalCurve":
        """Parallel shift of all hazard rates."""
        return SurvivalCurve(self.reference_date, self.times, self.rates + shift)
