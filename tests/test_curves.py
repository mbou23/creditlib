import math
from datetime import date

import pytest

from creditlib.curves import DiscountCurve, SurvivalCurve, bootstrap_discount_curve
from creditlib.dates import DayCount, add_tenor, generate_schedule, year_fraction

REF = date(2026, 11, 6)


def test_flat_discount_curve():
    curve = DiscountCurve.flat(REF, 0.03)
    assert curve.df(5.0) == pytest.approx(math.exp(-0.15))
    assert curve.zero_rate(7.3) == pytest.approx(0.03)
    assert curve.df(REF) == 1.0


def test_zero_rates_round_trip_and_flat_forwards():
    times, zeros = [1.0, 2.0, 5.0], [0.02, 0.025, 0.03]
    curve = DiscountCurve.from_zero_rates(REF, times, zeros)
    for t, z in zip(times, zeros):
        assert curve.zero_rate(t) == pytest.approx(z)
    # Forward is constant between pillars: (2*0.025 - 0.02) / 1 = 0.03
    assert curve.forward_rate(1.2, 1.8) == pytest.approx(0.03)
    # Flat extrapolation of the last forward
    assert curve.forward_rate(6.0, 9.0) == pytest.approx(curve.forward_rate(4.0, 5.0))


def test_dates_and_times_agree():
    curve = DiscountCurve.flat(REF, 0.03)
    d = date(2031, 11, 6)
    assert curve.df(d) == pytest.approx(curve.df(year_fraction(REF, d, DayCount.ACT_365F)))


def test_shifted_curve():
    curve = DiscountCurve.from_zero_rates(REF, [1.0, 5.0], [0.02, 0.03])
    assert curve.shifted(0.001).zero_rate(3.0) == pytest.approx(curve.zero_rate(3.0) + 0.001)


def test_survival_curve_piecewise_hazard():
    curve = SurvivalCurve(REF, [1.0, 3.0], [0.01, 0.03])
    assert curve.q(0.0) == 1.0
    assert curve.q(1.0) == pytest.approx(math.exp(-0.01))
    assert curve.q(2.0) == pytest.approx(math.exp(-0.01 - 0.03))
    assert curve.q(5.0) == pytest.approx(math.exp(-0.01 - 0.03 * 4))
    assert curve.hazard_rate(0.5) == 0.01
    assert curve.hazard_rate(2.5) == 0.03
    assert curve.default_probability(1.0) == pytest.approx(1 - math.exp(-0.01))


def test_invalid_pillars_rejected():
    with pytest.raises(ValueError):
        SurvivalCurve(REF, [2.0, 1.0], [0.01, 0.02])


def test_bootstrap_reprices_deposits_and_swaps():
    deposits = [("1M", 0.039), ("3M", 0.0395), ("6M", 0.04)]
    swaps = [("1Y", 0.039), ("2Y", 0.037), ("5Y", 0.036), ("10Y", 0.038)]
    curve = bootstrap_discount_curve(REF, deposits, swaps)

    for tenor, rate in deposits:
        from creditlib.dates import BusinessDayConvention, WEEKENDS_ONLY

        mat = WEEKENDS_ONLY.adjust(add_tenor(REF, tenor), BusinessDayConvention.MODIFIED_FOLLOWING)
        expected = 1 / (1 + rate * year_fraction(REF, mat, DayCount.ACT_360))
        assert curve.df(mat) == pytest.approx(expected, abs=1e-12)

    for tenor, rate in swaps:
        from creditlib.dates import BusinessDayConvention

        periods = generate_schedule(
            REF, add_tenor(REF, tenor), 2, payment_convention=BusinessDayConvention.MODIFIED_FOLLOWING
        )
        annuity = sum(
            year_fraction(p.accrual_start, p.accrual_end, DayCount.THIRTY_360) * curve.df(p.payment_date)
            for p in periods
        )
        par = (1 - curve.df(periods[-1].payment_date)) / annuity
        assert par == pytest.approx(rate, abs=1e-12)
