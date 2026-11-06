"""CDS risk measures by bump-and-rebuild.

Each measure rebuilds the survival curve from the (bumped) market quotes, as a
trading desk would, rather than bumping hazard rates directly.
"""
from __future__ import annotations

from typing import Sequence

from ..curves.discount import DiscountCurve
from ..instruments.cds import CDS
from ..pricers.cds import CDSPricer, CDSQuote, bootstrap_survival_curve


def _pv(cds: CDS, discount: DiscountCurve, quotes: Sequence[CDSQuote], recovery: float) -> float:
    survival = bootstrap_survival_curve(discount, quotes, recovery)
    return CDSPricer(discount, survival, recovery).pv(cds)


def cs01(
    cds: CDS, discount: DiscountCurve, quotes: Sequence[CDSQuote], recovery: float = 0.4,
    bump: float = 1e-4,
) -> float:
    """PV change for a parallel +1bp move in all par-spread quotes (positive for a protection buyer)."""
    base = _pv(cds, discount, quotes, recovery)
    bumped = _pv(cds, discount, [q.bumped(bump) for q in quotes], recovery)
    return bumped - base


def ir01(
    cds: CDS, discount: DiscountCurve, quotes: Sequence[CDSQuote], recovery: float = 0.4,
    bump: float = 1e-4,
) -> float:
    """PV change for a parallel +1bp shift of zero rates, holding spread quotes fixed."""
    base = _pv(cds, discount, quotes, recovery)
    bumped = _pv(cds, discount.shifted(bump), quotes, recovery)
    return bumped - base


def recovery01(
    cds: CDS, discount: DiscountCurve, quotes: Sequence[CDSQuote], recovery: float = 0.4,
    bump: float = 0.01,
) -> float:
    """PV change for a +1% recovery assumption, re-calibrating to the same quotes."""
    base = _pv(cds, discount, quotes, recovery)
    bumped = _pv(cds, discount, quotes, recovery + bump)
    return bumped - base


def jump_to_default(
    cds: CDS, discount: DiscountCurve, quotes: Sequence[CDSQuote], recovery: float = 0.4,
) -> float:
    """Immediate P&L if the reference entity defaults now.

    The buyer receives ``(1 - R) * notional`` and the accrued premium is paid,
    while the current mark-to-market is lost.
    """
    survival = bootstrap_survival_curve(discount, quotes, recovery)
    pricer = CDSPricer(discount, survival, recovery)
    return cds.notional * (1.0 - recovery) - pricer.accrued_premium(cds) - pricer.pv(cds)
