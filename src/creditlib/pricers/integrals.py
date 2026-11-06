"""Exact integrals for default-contingent legs.

With piecewise-flat forwards ``r`` and hazards ``h``, the product
``P(u) = DF(u) * Q(u)`` decays exponentially on every interval where both
rates are constant. On such an interval ``[a, b]`` (length ``d``) define

    x = ln(P(a) / P(b)) = (r + h) d,      hd = ln(Q(a) / Q(b)).

Then, exactly:

    integral_a^b DF(u) h Q(u) du             = hd * P(a) * g1(x)
    integral_a^b (u - s) DF(u) h Q(u) du     = hd * P(a) * [(a - s) g1(x) + d g2(x)]

with ``g1(x) = (1 - e^-x) / x`` and ``g2(x) = (1 - e^-x (1 + x)) / x^2``, both of
which are evaluated stably for small ``x``. This is the same piecewise-exact
approach as the ISDA CDS Standard Model.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from ..curves.discount import DiscountCurve
from ..curves.survival import SurvivalCurve

_SMALL = 1e-4


def g1(x: float) -> float:
    if abs(x) < _SMALL:
        return 1.0 - x / 2.0 + x * x / 6.0
    return -math.expm1(-x) / x


def g2(x: float) -> float:
    if abs(x) < _SMALL:
        return 0.5 - x / 3.0 + x * x / 8.0
    return (1.0 - math.exp(-x) * (1.0 + x)) / (x * x)


def integration_grid(discount: DiscountCurve, survival: SurvivalCurve, a: float, b: float) -> List[float]:
    """Sorted times ``a = t0 < ... < tn = b`` such that both curves have constant rates on each interval."""
    inner = np.concatenate((discount.times, survival.times))
    inner = inner[(inner > a) & (inner < b)]
    return [a] + sorted(set(inner.tolist())) + [b]


def default_leg_integral(discount: DiscountCurve, survival: SurvivalCurve, a: float, b: float) -> float:
    """``integral_a^b DF(u) dQ_default(u)``: present value of 1 paid at default in ``[a, b]``."""
    if b <= a:
        return 0.0
    grid = integration_grid(discount, survival, a, b)
    total = 0.0
    for lo, hi in zip(grid[:-1], grid[1:]):
        df_lo, df_hi = discount.df(lo), discount.df(hi)
        q_lo, q_hi = survival.q(lo), survival.q(hi)
        hd = math.log(q_lo / q_hi)
        x = math.log(df_lo * q_lo / (df_hi * q_hi))
        total += hd * df_lo * q_lo * g1(x)
    return total


def accrual_on_default_integral(
    discount: DiscountCurve, survival: SurvivalCurve, accrual_start: float, a: float, b: float
) -> float:
    """``integral_a^b (u - accrual_start) DF(u) dQ_default(u)``, with times in years ACT/365F.

    Multiply by ``365/360`` (for an ACT/360 premium leg) to turn the elapsed
    time into the accrued year fraction paid on default.
    """
    if b <= a:
        return 0.0
    grid = integration_grid(discount, survival, a, b)
    total = 0.0
    for lo, hi in zip(grid[:-1], grid[1:]):
        df_lo, df_hi = discount.df(lo), discount.df(hi)
        q_lo, q_hi = survival.q(lo), survival.q(hi)
        hd = math.log(q_lo / q_hi)
        x = math.log(df_lo * q_lo / (df_hi * q_hi))
        total += hd * df_lo * q_lo * ((lo - accrual_start) * g1(x) + (hi - lo) * g2(x))
    return total
