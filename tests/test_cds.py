import math
from datetime import date, timedelta

import numpy as np
import pytest

from creditlib.curves import DiscountCurve, SurvivalCurve
from creditlib.dates import cds_standard_maturity
from creditlib.instruments import CDS
from creditlib.pricers import (
    CDSPricer,
    CDSQuote,
    bootstrap_survival_curve,
    quoted_spread_from_upfront,
    upfront_from_quoted_spread,
)
from creditlib.pricers.integrals import accrual_on_default_integral, default_leg_integral
from creditlib.risk import cs01, ir01, jump_to_default, recovery01

REF = date(2026, 11, 6)


@pytest.fixture
def disc():
    return DiscountCurve.from_zero_rates(REF, [0.5, 1, 2, 5, 10], [0.039, 0.038, 0.036, 0.035, 0.038])


@pytest.fixture
def quotes():
    spreads = {1: 0.005, 3: 0.008, 5: 0.012, 7: 0.014, 10: 0.016}
    return [CDSQuote(cds_standard_maturity(REF, n), par_spread=s) for n, s in spreads.items()]


# ---- exact integrals vs brute-force quadrature ----------------------------- #
def _trapezoid(f, a, b, n=200_001):
    x = np.linspace(a, b, n)
    y = np.array([f(u) for u in x])
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def test_default_leg_integral_matches_quadrature(disc):
    surv = SurvivalCurve(REF, [1.0, 3.0, 5.0], [0.01, 0.03, 0.05])
    exact = default_leg_integral(disc, surv, 0.2, 6.0)
    numeric = _trapezoid(lambda u: disc.df(u) * surv.hazard_rate(u) * surv.q(u), 0.2, 6.0)
    assert exact == pytest.approx(numeric, rel=2e-6)  # trapezoid error at hazard jumps


def test_accrual_on_default_integral_matches_quadrature(disc):
    surv = SurvivalCurve(REF, [1.0, 3.0], [0.02, 0.06])
    s, a, b = 0.7, 0.8, 2.4
    exact = accrual_on_default_integral(disc, surv, s, a, b)
    numeric = _trapezoid(lambda u: (u - s) * disc.df(u) * surv.hazard_rate(u) * surv.q(u), a, b)
    assert exact == pytest.approx(numeric, rel=2e-6)  # trapezoid error at hazard jumps


def test_integrals_with_zero_hazard():
    zero_rates = DiscountCurve.flat(REF, 0.0)
    no_default = SurvivalCurve.flat(REF, 0.0)
    assert default_leg_integral(zero_rates, no_default, 0.0, 5.0) == 0.0


# ---- pricer ---------------------------------------------------------------- #
def test_credit_triangle():
    """At zero rates and flat hazard h, par spread = h (1 - R) * 360/365 (ACT/360 premium accrual)."""
    h, r = 0.02, 0.4
    zero_rates = DiscountCurve.flat(REF, 0.0)
    pricer = CDSPricer(zero_rates, SurvivalCurve.flat(REF, h), r)
    cds = CDS.standard(REF, 5, coupon=0.01)
    assert pricer.par_spread(cds) == pytest.approx(h * (1 - r) * 360 / 365, rel=1e-3)


def test_credit_triangle_with_rates_is_approximate(disc):
    """With positive rates, coupons paid in arrears push the par spread slightly above the zero-rate value."""
    h, r = 0.02, 0.4
    pricer = CDSPricer(disc, SurvivalCurve.flat(REF, h), r)
    s = pricer.par_spread(CDS.standard(REF, 5, coupon=0.01))
    assert h * (1 - r) * 360 / 365 < s < h * (1 - r)


def test_rpv01_equals_survival_integral_at_zero_rates():
    """At zero rates, coupons plus accrual-on-default integrate exactly to the integral of Q (by parts)."""
    surv = SurvivalCurve(REF, [1.0, 3.0, 6.0], [0.01, 0.03, 0.05])
    pricer = CDSPricer(DiscountCurve.flat(REF, 0.0), surv, 0.4)
    cds = CDS.standard(REF, 5, coupon=0.01)
    t0 = surv.time(REF + timedelta(days=1))
    t1 = surv.time(cds.maturity + timedelta(days=1))
    grid = [t0, 1.0, 3.0, t1]  # integrate Q exactly on each flat-hazard piece
    integral = sum(
        (surv.q(a) - surv.q(b)) / surv.hazard_rate(b) for a, b in zip(grid[:-1], grid[1:])
    )
    assert pricer.rpv01(cds, clean=True) == pytest.approx(integral * 365 / 360, rel=1e-5)


def test_clean_and_dirty(disc):
    pricer = CDSPricer(disc, SurvivalCurve.flat(REF, 0.02), 0.4)
    cds = CDS.standard(REF, 5, coupon=0.01, notional=1e7)
    # Accrued from 21-Sep-2026 to step-in 7-Nov-2026 = 47 days ACT/360
    assert pricer.accrued_fraction(cds) == pytest.approx(47 / 360)
    assert pricer.clean_pv(cds) - pricer.pv(cds) == pytest.approx(pricer.accrued_premium(cds))


def test_upfront_zero_at_par(disc):
    pricer = CDSPricer(disc, SurvivalCurve.flat(REF, 0.03), 0.4)
    cds = CDS.standard(REF, 5, coupon=0.01)
    assert pricer.upfront(cds.with_coupon(pricer.par_spread(cds))) == pytest.approx(0.0, abs=1e-14)


def test_seller_is_mirror_of_buyer(disc):
    pricer = CDSPricer(disc, SurvivalCurve.flat(REF, 0.03), 0.4)
    buyer = CDS.standard(REF, 5, coupon=0.01, notional=1e7)
    seller = CDS.standard(REF, 5, coupon=0.01, notional=-1e7)
    assert pricer.pv(seller) == pytest.approx(-pricer.pv(buyer))


def test_expired_periods_are_ignored(disc):
    """Changing the accrual start before the step-in date only changes the accrued, not the clean value."""
    pricer = CDSPricer(disc, SurvivalCurve.flat(REF, 0.02), 0.4)
    standard = CDS.standard(REF, 5, coupon=0.01)
    later_start = CDS(REF, standard.maturity, 0.01, accrual_start=date(2026, 11, 1))
    assert pricer.upfront(standard) == pytest.approx(pricer.upfront(later_start), abs=2e-5)


# ---- bootstrap ------------------------------------------------------------- #
def test_bootstrap_reprices_every_quote(disc, quotes):
    surv = bootstrap_survival_curve(disc, quotes, 0.4)
    pricer = CDSPricer(disc, surv, 0.4)
    for q in quotes:
        assert pricer.par_spread(CDS(REF, q.maturity, coupon=0.01)) == pytest.approx(q.par_spread, abs=1e-12)
    assert np.all(surv.rates > 0)


def test_bootstrap_from_upfront_quotes(disc):
    mat = cds_standard_maturity(REF, 5)
    surv = bootstrap_survival_curve(disc, [CDSQuote(mat, coupon=0.05, upfront=-0.03)], 0.4)
    assert CDSPricer(disc, surv, 0.4).upfront(CDS(REF, mat, 0.05)) == pytest.approx(-0.03, abs=1e-12)


def test_flat_spreads_give_flat_hazard(disc):
    q = [CDSQuote(cds_standard_maturity(REF, n), par_spread=0.012) for n in (1, 3, 5, 7, 10)]
    surv = bootstrap_survival_curve(disc, q, 0.4)
    assert np.ptp(surv.rates) < 1e-4
    assert surv.rates[-1] == pytest.approx(0.012 / 0.6, rel=0.02)


def test_inverted_quote_raises(disc):
    # 1Y at 500bp then 2Y at 10bp implies negative forward default probability
    q = [
        CDSQuote(cds_standard_maturity(REF, 1), par_spread=0.05),
        CDSQuote(cds_standard_maturity(REF, 2), par_spread=0.001),
    ]
    with pytest.raises(ValueError):
        bootstrap_survival_curve(disc, q, 0.4)


def test_quote_validation():
    with pytest.raises(ValueError):
        CDSQuote(date(2031, 12, 20))
    with pytest.raises(ValueError):
        CDSQuote(date(2031, 12, 20), par_spread=0.01, coupon=0.01, upfront=0.0)


# ---- quoting conventions --------------------------------------------------- #
def test_quoted_spread_equal_to_coupon_has_zero_upfront(disc):
    cds = CDS.standard(REF, 5, coupon=0.01)
    assert upfront_from_quoted_spread(cds, 0.01, disc) == pytest.approx(0.0, abs=1e-12)


def test_upfront_quoted_spread_round_trip(disc):
    cds = CDS.standard(REF, 5, coupon=0.05)
    upfront = upfront_from_quoted_spread(cds, 0.0325, disc)
    assert upfront < 0  # spread below the 500bp coupon: buyer receives points
    assert quoted_spread_from_upfront(cds, upfront, disc) == pytest.approx(0.0325, abs=1e-12)


def test_upfront_increases_with_spread(disc):
    cds = CDS.standard(REF, 5, coupon=0.01)
    ups = [upfront_from_quoted_spread(cds, s, disc) for s in (0.005, 0.01, 0.02, 0.05)]
    assert all(a < b for a, b in zip(ups, ups[1:]))


# ---- risk ------------------------------------------------------------------ #
def test_cs01_close_to_rpv01(disc, quotes):
    cds = CDS.standard(REF, 5, coupon=0.01, notional=1e7)
    surv = bootstrap_survival_curve(disc, quotes, 0.4)
    rpv01 = CDSPricer(disc, surv, 0.4).rpv01(cds)
    assert cs01(cds, disc, quotes) == pytest.approx(rpv01 * 1e-4 * 1e7, rel=0.02)


def test_risk_signs_for_buyer(disc, quotes):
    cds = CDS.standard(REF, 5, coupon=0.01, notional=1e7)
    assert cs01(cds, disc, quotes) > 0
    assert jump_to_default(cds, disc, quotes) > 0
    # Higher recovery at fixed spreads means higher implied hazard; the buyer's PV
    # is roughly spread-driven, so the effect is small.
    assert abs(recovery01(cds, disc, quotes)) < 0.05 * cs01(cds, disc, quotes) * 100
    assert math.isfinite(ir01(cds, disc, quotes))


def test_jump_to_default_value(disc, quotes):
    cds = CDS.standard(REF, 5, coupon=0.01, notional=1e7)
    pricer = CDSPricer(disc, bootstrap_survival_curve(disc, quotes, 0.4), 0.4)
    expected = 0.6e7 - pricer.accrued_premium(cds) - pricer.pv(cds)
    assert jump_to_default(cds, disc, quotes) == pytest.approx(expected)
