from datetime import date

import pytest

from creditlib.curves import DiscountCurve, SurvivalCurve
from creditlib.instruments import FixedRateBond
from creditlib.pricers import bond as bp

REF = date(2026, 11, 6)


@pytest.fixture
def ten_year():
    return FixedRateBond(issue_date=date(2026, 6, 15), maturity=date(2036, 6, 15), coupon=0.05)


def annuity_price(coupon, ytm, n_periods, freq=2, face=100.0):
    """Closed-form price of a bond on a coupon date."""
    c, y = coupon / freq * face, ytm / freq
    return c * (1 - (1 + y) ** -n_periods) / y + face * (1 + y) ** -n_periods


def test_coupon_amounts(ten_year):
    flows = ten_year.cashflows()
    assert len(flows) == 20
    assert flows[0][1] == pytest.approx(2.5)
    assert flows[-1][1] == pytest.approx(102.5)


def test_price_on_coupon_date_matches_closed_form(ten_year):
    settle = date(2026, 6, 15)
    price = bp.clean_price_from_yield(ten_year, settle, 0.06)
    assert price == pytest.approx(annuity_price(0.05, 0.06, 20), abs=1e-10)
    assert price == pytest.approx(92.5613, abs=1e-4)  # textbook value


def test_par_bond(ten_year):
    assert bp.clean_price_from_yield(ten_year, date(2026, 12, 15), 0.05) == pytest.approx(100.0, abs=1e-10)


def test_accrued_interest(ten_year):
    # 15-Jun to 6-Nov under 30/360 = 141 days
    assert ten_year.accrued_interest(REF) == pytest.approx(5.0 * 141 / 360)


def test_yield_round_trip(ten_year):
    clean = bp.clean_price_from_yield(ten_year, REF, 0.0437)
    assert bp.yield_from_clean_price(ten_year, REF, clean) == pytest.approx(0.0437, abs=1e-12)


def test_zero_coupon_duration_equals_maturity():
    zcb = FixedRateBond(issue_date=date(2026, 11, 6), maturity=date(2031, 11, 6), coupon=0.0, frequency=1)
    y = 0.04
    assert bp.macaulay_duration(zcb, REF, y) == pytest.approx(5.0, abs=1e-6)
    assert bp.modified_duration(zcb, REF, y) == pytest.approx(5.0 / 1.04, abs=1e-6)
    assert bp.convexity(zcb, REF, y) == pytest.approx(5 * 6 / 1.04**2, rel=1e-4)


def test_dv01_consistent_with_price_change(ten_year):
    y = 0.05
    p0 = bp.dirty_price_from_yield(ten_year, REF, y)
    p1 = bp.dirty_price_from_yield(ten_year, REF, y - 1e-4)
    assert bp.dv01(ten_year, REF, y) == pytest.approx(p1 - p0, rel=1e-3)


def test_z_spread_round_trip(ten_year):
    curve = DiscountCurve.from_zero_rates(REF, [1, 2, 5, 10], [0.035, 0.036, 0.037, 0.04])
    clean = bp.dirty_price_from_curve(ten_year, REF, curve, z_spread=0.0125) - ten_year.accrued_interest(REF)
    assert bp.z_spread(ten_year, REF, clean, curve) == pytest.approx(0.0125, abs=1e-12)


def test_risky_bond_with_zero_hazard_is_riskless(ten_year):
    disc = DiscountCurve.flat(REF, 0.04)
    risky = bp.risky_dirty_price(ten_year, REF, disc, SurvivalCurve.flat(REF, 0.0), recovery=0.4)
    assert risky == pytest.approx(bp.dirty_price_from_curve(ten_year, REF, disc), abs=1e-10)


def test_risky_bond_zero_recovery_equals_z_spread_pricing(ten_year):
    """With R=0, a flat hazard h is equivalent to a continuous spread h over the curve."""
    disc = DiscountCurve.flat(REF, 0.04)
    risky = bp.risky_dirty_price(ten_year, REF, disc, SurvivalCurve.flat(REF, 0.02), recovery=0.0)
    assert risky == pytest.approx(bp.dirty_price_from_curve(ten_year, REF, disc, z_spread=0.02), abs=1e-10)


def test_recovery_increases_price(ten_year):
    disc, surv = DiscountCurve.flat(REF, 0.04), SurvivalCurve.flat(REF, 0.03)
    low = bp.risky_clean_price(ten_year, REF, disc, surv, recovery=0.2)
    high = bp.risky_clean_price(ten_year, REF, disc, surv, recovery=0.6)
    assert high > low


def test_implied_hazard_round_trip(ten_year):
    disc = DiscountCurve.flat(REF, 0.04)
    clean = bp.risky_clean_price(ten_year, REF, disc, SurvivalCurve.flat(REF, 0.025), recovery=0.4)
    assert bp.implied_hazard_rate(ten_year, REF, clean, disc, recovery=0.4) == pytest.approx(0.025, abs=1e-10)
