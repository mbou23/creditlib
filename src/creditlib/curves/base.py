"""Piecewise-constant rate curves.

Both discount curves (piecewise-flat instantaneous forwards, i.e. log-linear
discount factors) and survival curves (piecewise-flat hazard rates) share the
same structure: a rate that is constant on each interval ``(t[k-1], t[k]]``
and extrapolated flat beyond the last pillar. Time is measured in years
ACT/365F from the curve's reference date.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence, Union

import numpy as np

from ..dates.daycount import DayCount, year_fraction

DateOrTime = Union[date, float]


class PiecewiseConstantRateCurve:
    """Curve defined by pillar times and a constant rate on each interval up to that pillar."""

    def __init__(self, reference_date: date, times: Sequence[float], rates: Sequence[float]):
        times_arr = np.asarray(times, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)
        if times_arr.ndim != 1 or len(times_arr) == 0 or len(times_arr) != len(rates_arr):
            raise ValueError("times and rates must be non-empty 1-d sequences of equal length")
        if times_arr[0] <= 0 or np.any(np.diff(times_arr) <= 0):
            raise ValueError("pillar times must be positive and strictly increasing")
        self.reference_date = reference_date
        self.times = times_arr
        self.rates = rates_arr
        widths = np.diff(np.concatenate(([0.0], times_arr)))
        # Cumulative integral of the rate at each pillar.
        self._cum = np.concatenate(([0.0], np.cumsum(rates_arr * widths)))
        self._knots = np.concatenate(([0.0], times_arr))

    def time(self, d: DateOrTime) -> float:
        """Convert a date to a curve time (ACT/365F from the reference date); floats pass through."""
        if isinstance(d, date):
            return year_fraction(self.reference_date, d, DayCount.ACT_365F)
        return float(d)

    def integrated_rate(self, d: DateOrTime) -> float:
        """Integral of the rate from 0 to t."""
        t = self.time(d)
        if t <= 0.0:
            # Before the reference date: extrapolate the first rate backward.
            return self.rates[0] * t
        k = int(np.searchsorted(self._knots, t, side="left")) - 1  # interval index containing t
        k = min(k, len(self.rates) - 1)
        return self._cum[k] + self.rates[k] * (t - self._knots[k])

    def rate_at(self, d: DateOrTime) -> float:
        """Instantaneous rate at t (right-continuous convention at pillars: the left interval's rate)."""
        t = self.time(d)
        k = int(np.searchsorted(self.times, t, side="left"))
        return float(self.rates[min(k, len(self.rates) - 1)])

    def __repr__(self) -> str:
        pillars = ", ".join(f"{t:.3f}:{r:.5f}" for t, r in zip(self.times, self.rates))
        return f"{type(self).__name__}(ref={self.reference_date}, [{pillars}])"
