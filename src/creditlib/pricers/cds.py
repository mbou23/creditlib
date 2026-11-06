"""CDS pricing in the reduced-form (hazard-rate) framework, following ISDA Standard Model conventions.

Conventions and simplifications (to be refined in later versions):

* the valuation date is the curve reference date and equals the trade date ``T``;
  protection and accrued interest run from the step-in date ``T+1``;
* the cash settlement lag (T+3) is ignored: values are as of ``T``;
* coupons pay the full period accrual (ACT/360) if the name survives to the
  accrual end, and accrued premium is paid on default (exact integration);
* the protection leg pays ``(1 - R)`` at the default time, between step-in and maturity.

Sign convention: values are for the protection buyer, scaled by ``cds.notional``
(a negative notional is a protection seller).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Sequence

from scipy.optimize import brentq

from ..curves.discount import DiscountCurve
from ..curves.survival import SurvivalCurve
from ..dates.daycount import DayCount, year_fraction
from ..instruments.cds import CDS
from .integrals import accrual_on_default_integral, default_leg_integral

_TIME_TO_ACCRUAL = {DayCount.ACT_360: 365.0 / 360.0, DayCount.ACT_365F: 1.0}


class CDSPricer:
    """Prices CDS contracts off a discount curve, a survival curve and a recovery rate."""

    def __init__(self, discount: DiscountCurve, survival: SurvivalCurve, recovery: float = 0.4):
        if discount.reference_date != survival.reference_date:
            raise ValueError("discount and survival curves must share a reference date")
        if not 0.0 <= recovery < 1.0:
            raise ValueError("recovery must be in [0, 1)")
        self.discount = discount
        self.survival = survival
        self.recovery = recovery

    @property
    def valuation_date(self) -> date:
        return self.discount.reference_date

    @property
    def step_in_date(self) -> date:
        return self.valuation_date + timedelta(days=1)

    # ---- legs, per unit notional ------------------------------------------- #
    def protection_leg(self, cds: CDS) -> float:
        """PV of ``(1 - R)`` paid on default between step-in and maturity."""
        a = self.discount.time(self.step_in_date)
        b = self.discount.time(cds.maturity)
        return (1.0 - self.recovery) * default_leg_integral(self.discount, self.survival, a, b)

    def rpv01(self, cds: CDS, clean: bool = True) -> float:
        """Risky annuity: PV of the premium leg per unit of running spread.

        The dirty annuity includes the full first coupon; the clean annuity
        subtracts the premium accrued up to the step-in date.
        """
        scale = _TIME_TO_ACCRUAL.get(cds.day_count)
        if scale is None:
            raise ValueError(f"premium day count {cds.day_count} not supported")
        step_in = self.step_in_date
        t_step_in = self.discount.time(step_in)
        total = 0.0
        for p in cds.schedule:
            if p.accrual_end <= step_in:
                continue
            tau = year_fraction(p.accrual_start, p.accrual_end, cds.day_count)
            total += tau * self.discount.df(p.payment_date) * self.survival.q(p.accrual_end)
            s = self.discount.time(p.accrual_start)
            a = max(s, t_step_in)
            b = self.discount.time(p.accrual_end)
            total += scale * accrual_on_default_integral(self.discount, self.survival, s, a, b)
        if clean:
            total -= self.accrued_fraction(cds)
        return total

    def accrued_fraction(self, cds: CDS) -> float:
        """Accrual year fraction from the current period's start to the step-in date."""
        step_in = self.step_in_date
        for p in cds.schedule:
            if p.accrual_start <= step_in < p.accrual_end:
                return year_fraction(p.accrual_start, step_in, cds.day_count)
        return 0.0

    # ---- contract values --------------------------------------------------- #
    def par_spread(self, cds: CDS) -> float:
        """Running spread at which the contract has zero (clean) value."""
        return self.protection_leg(cds) / self.rpv01(cds, clean=True)

    def upfront(self, cds: CDS) -> float:
        """Clean upfront paid by the protection buyer, as a fraction of notional ("points upfront")."""
        return self.protection_leg(cds) - cds.coupon * self.rpv01(cds, clean=True)

    def accrued_premium(self, cds: CDS) -> float:
        """Premium accrued since the last coupon date, owed to the buyer at settlement (signed by notional)."""
        return cds.notional * cds.coupon * self.accrued_fraction(cds)

    def pv(self, cds: CDS) -> float:
        """Dirty mark-to-market to the protection buyer, scaled by notional."""
        return cds.notional * (self.protection_leg(cds) - cds.coupon * self.rpv01(cds, clean=False))

    def clean_pv(self, cds: CDS) -> float:
        return cds.notional * self.upfront(cds)


# --------------------------------------------------------------------------- #
# Hazard-rate bootstrap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CDSQuote:
    """A market quote: either a ``par_spread``, or a running ``coupon`` with a clean ``upfront``."""

    maturity: date
    par_spread: Optional[float] = None
    coupon: Optional[float] = None
    upfront: Optional[float] = None

    def __post_init__(self) -> None:
        by_spread = self.par_spread is not None
        by_upfront = self.coupon is not None and self.upfront is not None
        if by_spread == by_upfront:
            raise ValueError("give either par_spread, or both coupon and upfront")

    def bumped(self, spread_shift: float) -> "CDSQuote":
        if self.par_spread is None:
            raise ValueError("spread bumps need par-spread quotes")
        return CDSQuote(self.maturity, par_spread=self.par_spread + spread_shift)


def bootstrap_survival_curve(
    discount: DiscountCurve,
    quotes: Sequence[CDSQuote],
    recovery: float = 0.4,
) -> SurvivalCurve:
    """Piecewise-flat hazard curve that reprices each quote exactly, solved pillar by pillar.

    Pillars sit at ``maturity + 1 day`` so the final premium period, which accrues
    through maturity inclusive, depends only on its own pillar's hazard.
    """
    ref = discount.reference_date
    ordered = sorted(quotes, key=lambda q: q.maturity)
    times: List[float] = []
    hazards: List[float] = []
    for quote in ordered:
        t = discount.time(quote.maturity + timedelta(days=1))
        coupon = quote.par_spread if quote.par_spread is not None else quote.coupon
        target = 0.0 if quote.par_spread is not None else quote.upfront
        cds = CDS(trade_date=ref, maturity=quote.maturity, coupon=coupon)

        def error(h: float, t=t, cds=cds, target=target) -> float:
            curve = SurvivalCurve(ref, times + [t], hazards + [h])
            return CDSPricer(discount, curve, recovery).upfront(cds) - target

        try:
            h = brentq(error, 1e-12, 10.0, xtol=1e-15)
        except ValueError as exc:
            raise ValueError(f"no hazard rate reprices the {quote.maturity} quote; check for arbitrage") from exc
        times.append(t)
        hazards.append(h)
    return SurvivalCurve(ref, times, hazards)


# --------------------------------------------------------------------------- #
# Quoting conventions
# --------------------------------------------------------------------------- #
def upfront_from_quoted_spread(
    cds: CDS, quoted_spread: float, discount: DiscountCurve, recovery: float = 0.4
) -> float:
    """Convert a conventional (quoted) spread to clean points upfront via a flat hazard curve.

    The flat curve is bootstrapped so that the CDS's own maturity has par spread
    ``quoted_spread``; the contract is then valued with its standard coupon.
    """
    curve = bootstrap_survival_curve(discount, [CDSQuote(cds.maturity, par_spread=quoted_spread)], recovery)
    return CDSPricer(discount, curve, recovery).upfront(cds)


def quoted_spread_from_upfront(
    cds: CDS, upfront: float, discount: DiscountCurve, recovery: float = 0.4
) -> float:
    """Inverse of :func:`upfront_from_quoted_spread`."""
    curve = bootstrap_survival_curve(
        discount, [CDSQuote(cds.maturity, coupon=cds.coupon, upfront=upfront)], recovery
    )
    return CDSPricer(discount, curve, recovery).par_spread(cds)
