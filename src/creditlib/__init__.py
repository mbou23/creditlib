"""creditlib: a credit derivatives pricing library.

Roadmap: bonds and CDS (v0.1) -> credit-linked notes -> repacks -> CDO tranches.
"""
from .curves import DiscountCurve, SurvivalCurve, bootstrap_discount_curve
from .dates import BusinessDayConvention, Calendar, DayCount
from .instruments import CDS, FixedRateBond
from .pricers import (
    CDSPricer,
    CDSQuote,
    bootstrap_survival_curve,
    quoted_spread_from_upfront,
    upfront_from_quoted_spread,
)

__version__ = "0.1.0"

__all__ = [
    "BusinessDayConvention",
    "CDS",
    "CDSPricer",
    "CDSQuote",
    "Calendar",
    "DayCount",
    "DiscountCurve",
    "FixedRateBond",
    "SurvivalCurve",
    "bootstrap_discount_curve",
    "bootstrap_survival_curve",
    "quoted_spread_from_upfront",
    "upfront_from_quoted_spread",
]
