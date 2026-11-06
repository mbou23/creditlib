from datetime import date

import pytest

from creditlib.dates import (
    BusinessDayConvention,
    Calendar,
    DayCount,
    add_months,
    add_tenor,
    cds_accrual_start,
    cds_schedule,
    cds_standard_maturity,
    generate_schedule,
    next_imm_date,
    previous_imm_date,
    year_fraction,
)


class TestDayCount:
    def test_act_360(self):
        assert year_fraction(date(2026, 1, 1), date(2026, 7, 1), DayCount.ACT_360) == pytest.approx(181 / 360)

    def test_act_365f(self):
        assert year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_365F) == pytest.approx(366 / 365)

    @pytest.mark.parametrize(
        "start, end, expected_days",
        [
            (date(2026, 1, 31), date(2026, 3, 31), 60),   # both 31sts -> 30
            (date(2026, 1, 30), date(2026, 3, 31), 60),   # d1=30 so d2 31 -> 30
            (date(2026, 1, 15), date(2026, 3, 31), 76),   # d1<30 so d2 stays 31
            (date(2026, 2, 28), date(2026, 8, 28), 180),
        ],
    )
    def test_thirty_360_us(self, start, end, expected_days):
        assert year_fraction(start, end, DayCount.THIRTY_360) == pytest.approx(expected_days / 360)

    def test_thirty_e_360(self):
        assert year_fraction(date(2026, 1, 15), date(2026, 3, 31), DayCount.THIRTY_E_360) == pytest.approx(75 / 360)

    def test_act_act_isda_non_leap_full_year(self):
        assert year_fraction(date(2025, 7, 1), date(2026, 7, 1), DayCount.ACT_ACT_ISDA) == pytest.approx(1.0)

    def test_act_act_isda_across_leap_year(self):
        expected = 184 / 365 + 182 / 366
        assert year_fraction(date(2023, 7, 1), date(2024, 7, 1), DayCount.ACT_ACT_ISDA) == pytest.approx(expected)


class TestCalendar:
    cal = Calendar(holidays=[date(2026, 12, 25)])

    def test_following_rolls_weekend_forward(self):
        assert self.cal.adjust(date(2026, 11, 7)) == date(2026, 11, 9)  # Saturday -> Monday

    def test_holiday(self):
        assert self.cal.adjust(date(2026, 12, 25)) == date(2026, 12, 28)

    def test_modified_following_stays_in_month(self):
        # 31-Jan-2026 is a Saturday: Following gives 2-Feb, Modified Following gives 30-Jan.
        assert self.cal.adjust(date(2026, 1, 31), BusinessDayConvention.MODIFIED_FOLLOWING) == date(2026, 1, 30)

    def test_preceding(self):
        assert self.cal.adjust(date(2026, 11, 8), BusinessDayConvention.PRECEDING) == date(2026, 11, 6)

    def test_add_business_days(self):
        assert self.cal.add_business_days(date(2026, 11, 6), 1) == date(2026, 11, 9)


def test_add_months_clips_to_month_end():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2026, 2, 28), 1, end_of_month=True) == date(2026, 3, 31)


def test_add_tenor():
    assert add_tenor(date(2026, 11, 6), "5Y") == date(2031, 11, 6)
    assert add_tenor(date(2026, 11, 6), "2W") == date(2026, 11, 20)
    with pytest.raises(ValueError):
        add_tenor(date(2026, 11, 6), "5X")


def test_bond_schedule_has_short_front_stub():
    periods = generate_schedule(date(2026, 1, 10), date(2028, 6, 15), 2)
    assert periods[0].accrual_start == date(2026, 1, 10)
    assert periods[0].accrual_end == date(2026, 6, 15)
    assert [p.accrual_end for p in periods[1:]] == [
        date(2026, 12, 15), date(2027, 6, 15), date(2027, 12, 15), date(2028, 6, 15)
    ]


class TestCDSDates:
    @pytest.mark.parametrize(
        "d, expected",
        [
            (date(2026, 11, 6), date(2026, 9, 20)),
            (date(2026, 9, 20), date(2026, 9, 20)),
            (date(2026, 9, 19), date(2026, 6, 20)),
            (date(2026, 2, 1), date(2025, 12, 20)),
        ],
    )
    def test_previous_imm(self, d, expected):
        assert previous_imm_date(d) == expected

    def test_next_imm(self):
        assert next_imm_date(date(2026, 12, 20)) == date(2027, 3, 20)

    @pytest.mark.parametrize(
        "trade, expected",
        [
            (date(2026, 11, 6), date(2031, 12, 20)),   # after Sep roll -> Dec series
            (date(2026, 3, 19), date(2030, 12, 20)),   # before Mar roll -> previous Dec series
            (date(2026, 3, 20), date(2031, 6, 20)),    # on Mar roll -> Jun series
            (date(2026, 9, 21), date(2031, 12, 20)),
        ],
    )
    def test_standard_maturity_semi_annual_roll(self, trade, expected):
        assert cds_standard_maturity(trade, 5) == expected

    def test_accrual_start_adjusted(self):
        # 20-Sep-2026 is a Sunday: accrual starts Monday 21-Sep.
        assert cds_accrual_start(date(2026, 11, 6)) == date(2026, 9, 21)

    def test_schedule_last_period_includes_maturity(self):
        periods = cds_schedule(date(2026, 9, 21), date(2031, 12, 20))
        assert len(periods) == 21
        assert periods[0].accrual_end == date(2026, 12, 21)    # 20-Dec-2026 is a Sunday
        assert periods[-1].accrual_end == date(2031, 12, 21)   # maturity + 1 day
        assert periods[-1].payment_date == date(2031, 12, 22)  # 20-Dec-2031 Saturday -> Monday
        for prev, nxt in zip(periods[:-1], periods[1:]):
            assert prev.accrual_end == nxt.accrual_start
