"""Discount curves with piecewise-flat forward rates (log-linear discount factors)."""
from __future__ import annotations

import math
from datetime import date
from typing import Sequence

import numpy as np

from .base import DateOrTime, PiecewiseConstantRateCurve


class DiscountCurve(PiecewiseConstantRateCurve):
    """``rates`` are continuously-compounded instantaneous forwards on each interval."""

    @classmethod
    def flat(cls, reference_date: date, rate: float, horizon: float = 100.0) -> "DiscountCurve":
        """Flat continuously-compounded zero rate."""
        return cls(reference_date, [horizon], [rate])

    @classmethod
    def from_zero_rates(
        cls, reference_date: date, times: Sequence[float], zero_rates: Sequence[float]
    ) -> "DiscountCurve":
        """Build from continuously-compounded zero rates at pillar times."""
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        integrated = np.concatenate(([0.0], z * t))
        knots = np.concatenate(([0.0], t))
        forwards = np.diff(integrated) / np.diff(knots)
        return cls(reference_date, t, forwards)

    def discount_factor(self, d: DateOrTime) -> float:
        return math.exp(-self.integrated_rate(d))

    def df(self, d: DateOrTime) -> float:
        """Alias for :meth:`discount_factor`."""
        return self.discount_factor(d)

    def zero_rate(self, d: DateOrTime) -> float:
        """Continuously-compounded zero rate to t."""
        t = self.time(d)
        if t <= 0:
            return float(self.rates[0])
        return self.integrated_rate(t) / t

    def forward_rate(self, d1: DateOrTime, d2: DateOrTime) -> float:
        """Continuously-compounded forward rate between two times."""
        t1, t2 = self.time(d1), self.time(d2)
        return (self.integrated_rate(t2) - self.integrated_rate(t1)) / (t2 - t1)

    def shifted(self, shift: float) -> "DiscountCurve":
        """Parallel shift of all forwards (equivalently, of all zero rates) by ``shift``."""
        return DiscountCurve(self.reference_date, self.times, self.rates + shift)
