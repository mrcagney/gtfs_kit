from .context import gtfs_kit, sample, cairns
from gtfs_kit import *


def test_get_dates():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_dates(feed, as_date_obj=as_date_obj)
        d1 = "20140526"
        d2 = "20141228"
        if as_date_obj:
            d1, d2 = map(datestr_to_date, [d1, d2])
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2

        # Should work on empty calendar files too
        feed1 = feed.copy()
        c = feed1.calendar_dates
        feed1.calendar_dates = c[:0]
        c = feed1.calendar
        feed1.calendar = c[:0]
        dates = get_dates(feed1, as_date_obj=as_date_obj)
        assert not dates


def test_get_week():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_week(feed, 1, as_date_obj=as_date_obj)
        d1 = "20140526"
        d2 = "20140601"
        if as_date_obj:
            d1, d2 = map(datestr_to_date, [d1, d2])
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2


def test_get_first_week():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_first_week(feed, as_date_obj=as_date_obj)
        d1 = "20140526"
        d2 = "20140601"
        if as_date_obj:
            d1, d2 = map(datestr_to_date, [d1, d2])
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2


def test_subset_dates():
    feed = cairns.copy()
    dates = feed.get_dates()
    assert subset_dates(feed, [dates[0]]) == [dates[0]]
    assert subset_dates(feed, ["9999"]) == []
    assert subset_dates(feed, dates + ["999"]) == dates
