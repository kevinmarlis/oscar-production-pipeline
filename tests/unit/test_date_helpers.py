import pytest

from oscar.utils.date_helpers import get_month_info


@pytest.mark.parametrize(
    "ym, expected",
    [
        ("2020-07", ("2020", "07", "01", "31")),  # 31-day month
        ("2019-11", ("2019", "11", "01", "30")),  # 30-day month
        ("2021-02", ("2021", "02", "01", "28")),  # non-leap February
        ("2020-02", ("2020", "02", "01", "29")),  # leap February
        ("2000-02", ("2000", "02", "01", "29")),  # century leap year
        ("1900-02", ("1900", "02", "01", "28")),  # century non-leap year
    ],
)
def test_get_month_info(ym, expected):
    assert get_month_info(ym) == expected


def test_month_is_zero_padded():
    # single-digit input month must be padded (distinct from the parametrized cases).
    year, month, first, last = get_month_info("2023-9")
    assert month == "09"
    assert first == "01"
    assert last == "30"
