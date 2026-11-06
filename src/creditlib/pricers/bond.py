"""Bond analytics: yield, curve and credit-risky pricing, spreads and risk measures.

Prices are per ``bond.face`` (100 by default). "Dirty" includes accrued interest.
"""
from __future__ import annotations

import math
from datetime import date

from scipy.optimize import brentq

from ..curves.discount import DiscountCurve
from ..curves.survival import SurvivalCurve
from ..dates.daycount import year_fraction
from ..instruments.bond import FixedRateBond
from .integrals import default_leg_integral


# --------------------------------------------------------------------------- #
# Yield-based (street convention)
# --------------------------------------------------------------------------- #
def dirty_price_from_yield(bond: FixedRateBond, settlement: date, ytm: float) -> float:
    """Street-convention dirty price: compounding at the coupon frequency, fractional first period."""
    current = bond.current_period(settlement)
    w = year_fraction(settlement, current.accrual_end, bond.day_count) / year_fraction(
        current.accrual_start, current.accrual_end, bond.day_count
    )
    base = 1.0 + ytm / bond.frequency
    return sum(cf / base ** (w + i) for i, (_, cf) in enumerate(bond.cashflows(after=settlement)))


def clean_price_from_yield(bond: FixedRateBond, settlement: date, ytm: float) -> float:
    return dirty_price_from_yield(bond, settlement, ytm) - bond.accrued_interest(settlement)


def yield_from_clean_price(bond: FixedRateBond, settlement: date, clean_price: float) -> float:
    target = clean_price + bond.accrued_interest(settlement)
    return brentq(lambda y: dirty_price_from_yield(bond, settlement, y) - target, -0.5, 5.0, xtol=1e-14)


def modified_duration(bond: FixedRateBond, settlement: date, ytm: float, bump: float = 1e-5) -> float:
    """``-(1/P) dP/dy`` by central difference on the dirty price."""
    up = dirty_price_from_yield(bond, settlement, ytm + bump)
    down = dirty_price_from_yield(bond, settlement, ytm - bump)
    p = dirty_price_from_yield(bond, settlement, ytm)
    return -(up - down) / (2 * bump * p)


def macaulay_duration(bond: FixedRateBond, settlement: date, ytm: float) -> float:
    return modified_duration(bond, settlement, ytm) * (1.0 + ytm / bond.frequency)


def convexity(bond: FixedRateBond, settlement: date, ytm: float, bump: float = 1e-4) -> float:
    """``(1/P) d2P/dy2`` by central difference on the dirty price."""
    up = dirty_price_from_yield(bond, settlement, ytm + bump)
    down = dirty_price_from_yield(bond, settlement, ytm - bump)
    p = dirty_price_from_yield(bond, settlement, ytm)
    return (up - 2 * p + down) / (bump * bump * p)


def dv01(bond: FixedRateBond, settlement: date, ytm: float) -> float:
    """Dirty price change for a 1bp fall in yield (positive for a long bond)."""
    return modified_duration(bond, settlement, ytm) * dirty_price_from_yield(bond, settlement, ytm) * 1e-4


# --------------------------------------------------------------------------- #
# Curve-based
# --------------------------------------------------------------------------- #
def dirty_price_from_curve(
    bond: FixedRateBond, settlement: date, discount: DiscountCurve, z_spread: float = 0.0
) -> float:
    """Dirty price for ``settlement`` from a discount curve plus an optional continuous Z-spread.

    Cashflows are discounted to the curve date and carried forward to settlement.
    """
    ts = discount.time(settlement)
    pv = sum(
        cf * discount.df(p.payment_date) * math.exp(-z_spread * (discount.time(p.payment_date) - ts))
        for p, cf in bond.cashflows(after=settlement)
    )
    return pv / discount.df(settlement)


def z_spread(bond: FixedRateBond, settlement: date, clean_price: float, discount: DiscountCurve) -> float:
    """Continuously-compounded spread over the discount curve that reprices the bond."""
    target = clean_price + bond.accrued_interest(settlement)
    return brentq(
        lambda z: dirty_price_from_curve(bond, settlement, discount, z) - target, -0.5, 5.0, xtol=1e-14
    )


# --------------------------------------------------------------------------- #
# Credit-risky (reduced-form) pricing
# --------------------------------------------------------------------------- #
def risky_dirty_price(
    bond: FixedRateBond,
    settlement: date,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Dirty price of a defaultable bond under recovery of par (face), conditional on survival to settlement.

    PV = sum CF_i DF(t_i) Q(t_i) + R * face * integral DF(u) dQ_default(u)
    """
    if discount.reference_date != survival.reference_date:
        raise ValueError("discount and survival curves must share a reference date")
    flows = bond.cashflows(after=settlement)
    ts = discount.time(settlement)
    t_end = discount.time(flows[-1][0].payment_date)
    survival_leg = sum(cf * discount.df(p.payment_date) * survival.q(p.payment_date) for p, cf in flows)
    recovery_leg = recovery * bond.face * default_leg_integral(discount, survival, ts, t_end)
    return (survival_leg + recovery_leg) / (discount.df(ts) * survival.q(ts))


def risky_clean_price(
    bond: FixedRateBond, settlement: date, discount: DiscountCurve, survival: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    return risky_dirty_price(bond, settlement, discount, survival, recovery) - bond.accrued_interest(settlement)


def implied_hazard_rate(
    bond: FixedRateBond, settlement: date, clean_price: float, discount: DiscountCurve,
    recovery: float = 0.4,
) -> float:
    """Flat hazard rate that reprices the bond (bond-implied default intensity)."""
    ref = discount.reference_date

    def error(h: float) -> float:
        return risky_clean_price(bond, settlement, discount, SurvivalCurve.flat(ref, h), recovery) - clean_price

    return brentq(error, 0.0, 5.0, xtol=1e-14)
