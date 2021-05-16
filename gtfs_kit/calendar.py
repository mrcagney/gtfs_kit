"""
Functions about calendar and calendar_dates.
"""
from __future__ import annotations
import dateutil.relativedelta as rd
from typing import TYPE_CHECKING

from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_dates(feed: "Feed", *, as_date_obj: bool = False) -> list[str]:
    """
    Return a list of YYYYMMDD date strings for which the given Feed is valid,
    which could be the empty list if the Feed has no calendar information.

    If ``as_date_obj``, then return datetime.date objects instead.
    """
    dates = []
    if feed.calendar is not None and not feed.calendar.empty:
        if "start_date" in feed.calendar.columns:
            dates.append(feed.calendar["start_date"].min())
        if "end_date" in feed.calendar.columns:
            dates.append(feed.calendar["end_date"].max())
    if feed.calendar_dates is not None and not feed.calendar_dates.empty:
        if "date" in feed.calendar_dates.columns:
            start = feed.calendar_dates["date"].min()
            end = feed.calendar_dates["date"].max()
            dates.extend([start, end])
    if not dates:
        return []

    start_date, end_date = min(dates), max(dates)
    start_date, end_date = map(hp.datestr_to_date, [start_date, end_date])
    num_days = (end_date - start_date).days
    result = [start_date + rd.relativedelta(days=+d) for d in range(num_days + 1)]

    # Convert dates back to strings if required
    if not as_date_obj:
        result = [hp.datestr_to_date(x, inverse=True) for x in result]

    return result


def get_week(feed: "Feed", k: int, *, as_date_obj: bool = False) -> list[str]:
    """
    Given a Feed and a positive integer ``k``,
    return a list of YYYYMMDD date strings corresponding to the kth Monday--Sunday week
    (or initial segment thereof) for which the Feed is valid.
    For example, k=1 returns the first Monday--Sunday week (or initial segment thereof).
    If the Feed does not have k Mondays, then return the empty list.

    If ``as_date_obj``, then return datetime.date objects instead.
    """
    dates = feed.get_dates(as_date_obj=True)
    n = len(dates)

    # Get first Monday
    monday_index = None
    for (i, date) in enumerate(dates):
        if date.weekday() == 0:
            monday_index = i
            break

    # Get week k
    if k < 1 or monday_index is None or monday_index + 7 * (k - 1) > n:
        result = []
    else:
        result = dates[monday_index + 7 * (k - 1) : monday_index + 7 * k]

    # Convert to date strings if requested
    if not as_date_obj:
        result = [hp.datestr_to_date(x, inverse=True) for x in result]

    return result


def get_first_week(feed: "Feed", *, as_date_obj: bool = False) -> list[str]:
    """
    Return a list of YYYYMMDD date strings for the first Monday--Sunday
    week (or initial segment thereof) for which the given Feed is valid.
    If the feed has no Mondays, then return the empty list.

    If ``as_date_obj``, then return date objects, otherwise return date strings.
    """
    return get_week(feed, 1, as_date_obj=as_date_obj)


def subset_dates(feed: "Feed", dates: list[str]) -> list[str]:
    """
    Given a Feed and a list of YYYYMMDD date strings,
    return the sublist of dates that lie in the Feed's dates
    (the output :func:`feed.get_dates`).
    """
    return [d for d in dates if d in feed.get_dates()]
