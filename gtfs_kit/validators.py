"""
Functions about validation.
"""
import re
import pytz
import datetime as dt
from typing import Optional, List, Union, TYPE_CHECKING

import pycountry
import numpy as np
import pandas as pd
from pandas import DataFrame

from . import constants as cs
from . import helpers as hp

if TYPE_CHECKING:
    from .feed import Feed


TIME_PATTERN1 = re.compile(r"^\d\d:\d\d:\d\d$")
TIME_PATTERN2 = re.compile(r"^\d:\d\d:\d\d$")
DATE_FORMAT = "%Y%m%d"
TIMEZONES = set(pytz.all_timezones)
# ISO639-1 language codes, both lower and upper case
LANGS = set(
    [lang.alpha_2 for lang in pycountry.languages if hasattr(lang, "alpha_2")]
)
LANGS |= set(x.upper() for x in LANGS)
CURRENCIES = set(
    [c.alpha_3 for c in pycountry.currencies if hasattr(c, "alpha_3")]
)
URL_PATTERN = re.compile(
    r"^(?:http)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[^@]+@[^@]+\.[^@]+")
COLOR_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}){3}$")


def valid_str(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a non-blank string;
    otherwise return ``False``.
    """
    if isinstance(x, str) and x.strip():
        return True
    else:
        return False


def valid_time(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a valid H:MM:SS or HH:MM:SS time;
    otherwise return ``False``.
    """
    if isinstance(x, str) and (
        re.match(TIME_PATTERN1, x) or re.match(TIME_PATTERN2, x)
    ):
        return True
    else:
        return False


def valid_date(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid YYYYMMDD date;
    otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False


def valid_timezone(x: str) -> bool:
    """
    Retrun ``True`` if ``x`` is a valid human-readable timezone string,
    e.g. 'Africa/Abidjan'; otherwise return ``False``.
    """
    return x in TIMEZONES


def valid_lang(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a valid two-letter ISO 639 language
    code, e.g. 'aa'; otherwise return ``False``.
    """
    return x in LANGS


def valid_currency(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a valid three-letter ISO 4217 currency
    code, e.g. 'AED'; otherwise return ``False``.
    """
    return x in CURRENCIES


def valid_url(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a valid URL; otherwise return ``False``.
    """
    if isinstance(x, str) and re.match(URL_PATTERN, x):
        return True
    else:
        return False


def valid_email(x: str) -> bool:
    """
    Return ``True`` if ``x`` is a valid email address; otherwise return
    ``False``.
    """
    if isinstance(x, str) and re.match(EMAIL_PATTERN, x):
        return True
    else:
        return False


def valid_color(x: str) -> bool:
    """
    Return ``True`` if ``x`` a valid hexadecimal color string without
    the leading hash; otherwise return ``False``.
    """
    if isinstance(x, str) and re.match(COLOR_PATTERN, x):
        return True
    else:
        return False


def check_for_required_columns(
    problems: List, table: str, df: DataFrame
) -> List:
    """
    Check that the given GTFS table has the required columns.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Check that the DataFrame contains the colums required by GTFS
        and append to the problems list one error for each column
        missing.

    """
    r = cs.GTFS_REF
    req_columns = r.loc[
        (r["table"] == table) & r["column_required"], "column"
    ].values
    for col in req_columns:
        if col not in df.columns:
            problems.append(["error", f"Missing column {col}", table, []])

    return problems


def check_for_invalid_columns(
    problems: List, table: str, df: DataFrame
) -> List:
    """
    Check for invalid columns in the given GTFS DataFrame.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Check whether the DataFrame contains extra columns not in the
        GTFS and append to the problems list one warning for each extra
        column.

    """
    r = cs.GTFS_REF
    valid_columns = r.loc[r["table"] == table, "column"].values
    for col in df.columns:
        if col not in valid_columns:
            problems.append(
                ["warning", f"Unrecognized column {col}", table, []]
            )

    return problems


def check_table(
    problems: List,
    table: str,
    df: DataFrame,
    condition,
    message: str,
    type_: str = "error",
) -> List:
    """
    Check the given GTFS table for the given problem condition.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``
    condition : boolean expression
        One involving ``df``, e.g.`df['route_id'].map(is_valid_str)``
    message : string
        Problem message, e.g. ``'Invalid route_id'``
    type_ : string
        ``'error'`` or ``'warning'`` indicating the type of problem
        encountered

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Record the indices of ``df`` that statisfy the condition.
        If the list of indices is nonempty, append to the
        problems the item ``[type_, message, table, indices]``;
        otherwise do not append anything.

    """
    indices = df.loc[condition].index.tolist()
    if indices:
        problems.append([type_, message, table, indices])

    return problems


def check_column(
    problems: List,
    table: str,
    df: DataFrame,
    column: str,
    checker,
    message: Optional[str] = None,
    type_: str = "error",
    *,
    column_required: bool = True,
) -> List:
    """
    Check the given column of the given GTFS with the given problem
    checker.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``
    column : string
        A column of ``df``
    column_required : boolean
        ``True`` if and only if ``column`` is required
        (and not optional) by the GTFS
    checker : boolean valued unary function
        Returns ``True`` if and only if no problem is encountered
    message : string (optional)
        Problem message, e.g. 'Invalid route_id'.
        Defaults to 'Invalid ``column``; maybe has extra space characters'
    type_ : string
        ``'error'`` or ``'warning'`` indicating the type of problem
        encountered

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Apply the checker to the column entries and record the indices
        of ``df`` where the checker returns ``False``.
        If the list of indices of is nonempty, append to the problems the
        item ``[type_, problem, table, indices]``; otherwise do not
        append anything.

        If not ``column_required``, then NaN entries will be ignored
        before applying the checker.

    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].map(checker)

    if not message:
        message = f"Invalid {column}; maybe has extra space characters"

    problems = check_table(problems, table, f, cond, message, type_)

    return problems


def check_column_id(
    problems: List,
    table: str,
    df: DataFrame,
    column: str,
    *,
    column_required: bool = True,
) -> List:
    """
    A specialization of :func:`check_column`.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``
    column : string
        A column of ``df``
    column_required : boolean
        ``True`` if and only if ``column`` is required
        (and not optional) by the GTFS

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Record the indices of ``df`` where the given column has
        duplicated entry or an invalid strings.
        If the list of indices is nonempty, append to the problems the
        item ``[type_, problem, table, indices]``; otherwise do not
        append anything.

        If not ``column_required``, then NaN entries will be ignored
        in the checking.

    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].map(valid_str)
    problems = check_table(
        problems,
        table,
        f,
        cond,
        f"Invalid {column}; maybe has extra space characters",
    )

    cond = f[column].duplicated()
    problems = check_table(problems, table, f, cond, f"Repeated {column}")

    return problems


def check_column_linked_id(
    problems: List,
    table: str,
    df: DataFrame,
    column: str,
    target_df: DataFrame,
    target_column: Optional[str] = None,
    *,
    column_required: bool = True,
) -> List:
    """
    A modified version of :func:`check_column_id`.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    table : string
        Name of a GTFS table
    df : DataFrame
        The GTFS table corresponding to ``table``
    column : string
        A column of ``df``
    column_required : boolean
        ``True`` if and only if ``column`` is required
        (and not optional) by the GTFS
    target_df : DataFrame
        A GTFS table
    target_column : string
        A column of ``target_df``; defaults to ``column_name``

    Returns
    -------
    list
        The ``problems`` list extended as follows.
        Record indices of ``df`` where the following condition is
        violated: ``column`` contain IDs that are valid strings and are
        present in ``target_df`` under the ``target_column`` name.
        If the list of indices is nonempty, append to the problems the
        item ``[type_, problem, table, indices]``; otherwise do not
        append anything.

        If not ``column_required``, then NaN entries will be ignored
        in the checking.

    """
    if target_column is None:
        target_column = column

    f = df.copy()

    if target_df is None:
        g = pd.DataFrame()
        g[target_column] = np.nan
    else:
        g = target_df.copy()

    if target_column not in g.columns:
        g[target_column] = np.nan

    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])
        g = g.dropna(subset=[target_column])

    cond = ~f[column].isin(g[target_column])
    problems = check_table(problems, table, f, cond, f"Undefined {column}")

    return problems


def format_problems(
    problems: List, *, as_df: bool = False
) -> Union[List, DataFrame]:
    """
    Format the given problems list as a DataFrame.

    Parameters
    ----------
    problems : list
        A four-tuple containing

        1. A problem type (string) equal to ``'error'`` or ``'warning'``;
           ``'error'`` means the GTFS is violated;
           ``'warning'`` means there is a problem but it is not a
           GTFS violation
        2. A message (string) that describes the problem
        3. A GTFS table name, e.g. ``'routes'``, in which the problem
           occurs
        4. A list of rows (integers) of the table's DataFrame where the
           problem occurs

    as_df : boolean

    Returns
    -------
    list or DataFrame
        Return ``problems`` if not ``as_df``; otherwise return a
        DataFrame with the problems as rows and the columns
        ``['type', 'message', 'table', 'rows']``.

    """
    if as_df:
        problems = pd.DataFrame(
            problems, columns=["type", "message", "table", "rows"]
        ).sort_values(["type", "table"])
    return problems


def check_agency(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Check that ``feed.agency`` follows the GTFS.
    Return a list of problems of the form described in
    :func:`check_table`;
    the list will be empty if no problems are found.
    """
    table = "agency"
    problems = []

    # Preliminary checks
    if feed.agency is None:
        problems.append(["error", "Missing table", table, []])
    else:
        f = feed.agency.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column_id(
        problems, table, f, "agency_id", column_required=False
    )

    # Check agency_name
    problems = check_column(problems, table, f, "agency_name", valid_str)

    # Check agency_url
    problems = check_column(problems, table, f, "agency_url", valid_url)

    # Check agency_timezone
    problems = check_column(
        problems, table, f, "agency_timezone", valid_timezone
    )

    # Check agency_fare_url
    problems = check_column(
        problems, table, f, "agency_fare_url", valid_url, column_required=False
    )

    # Check agency_lang
    problems = check_column(
        problems, table, f, "agency_lang", valid_lang, column_required=False
    )

    # Check agency_phone
    problems = check_column(
        problems, table, f, "agency_phone", valid_str, column_required=False
    )

    # Check agency_email
    problems = check_column(
        problems, table, f, "agency_email", valid_email, column_required=False
    )

    return format_problems(problems, as_df=as_df)


def check_calendar(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.calendar``.
    """
    table = "calendar"
    problems = []

    # Preliminary checks
    if feed.calendar is None:
        return problems

    f = feed.calendar.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column_id(problems, table, f, "service_id")

    # Check weekday columns
    v = lambda x: x in range(2)
    for col in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]:
        problems = check_column(problems, table, f, col, v)

    # Check start_date and end_date
    for col in ["start_date", "end_date"]:
        problems = check_column(problems, table, f, col, valid_date)

    if include_warnings:
        # Check if feed has expired
        d = f["end_date"].max()
        if feed.calendar_dates is not None and not feed.calendar_dates.empty:
            table += "/calendar_dates"
            d = max(d, feed.calendar_dates["date"].max())
        if d < dt.datetime.today().strftime(DATE_FORMAT):
            problems.append(["warning", "Feed expired", table, []])

    return format_problems(problems, as_df=as_df)


def check_calendar_dates(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """
    table = "calendar_dates"
    problems = []

    # Preliminary checks
    if feed.calendar_dates is None:
        return problems

    f = feed.calendar_dates.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column(problems, table, f, "service_id", valid_str)

    # Check date
    problems = check_column(problems, table, f, "date", valid_date)

    # No duplicate (service_id, date) pairs allowed
    cond = f[["service_id", "date"]].duplicated()
    problems = check_table(
        problems, table, f, cond, "Repeated pair (service_id, date)"
    )

    # Check exception_type
    v = lambda x: x in [1, 2]
    problems = check_column(problems, table, f, "exception_type", v)

    return format_problems(problems, as_df=as_df)


def check_fare_attributes(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """
    table = "fare_attributes"
    problems = []

    # Preliminary checks
    if feed.fare_attributes is None:
        return problems

    f = feed.fare_attributes.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check fare_id
    problems = check_column_id(problems, table, f, "fare_id")

    # Check currency_type
    problems = check_column(
        problems, table, f, "currency_type", valid_currency
    )

    # Check payment_method
    v = lambda x: x in range(2)
    problems = check_column(problems, table, f, "payment_method", v)

    # Check transfers
    v = lambda x: pd.isna(x) or x in range(3)
    problems = check_column(problems, table, f, "transfers", v)

    # Check transfer_duration
    v = lambda x: x >= 0
    problems = check_column(
        problems, table, f, "transfer_duration", v, column_required=False
    )

    return format_problems(problems, as_df=as_df)


def check_fare_rules(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """
    table = "fare_rules"
    problems = []

    # Preliminary checks
    if feed.fare_rules is None:
        return problems

    f = feed.fare_rules.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check fare_id
    problems = check_column_linked_id(
        problems, table, f, "fare_id", feed.fare_attributes
    )

    # Check route_id
    problems = check_column_linked_id(
        problems, table, f, "route_id", feed.routes, column_required=False
    )

    # Check origin_id, destination_id, contains_id
    for col in ["origin_id", "destination_id", "contains_id"]:
        problems = check_column_linked_id(
            problems,
            table,
            f,
            col,
            feed.stops,
            "zone_id",
            column_required=False,
        )

    return format_problems(problems, as_df=as_df)


def check_feed_info(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.feed_info``.
    """
    table = "feed_info"
    problems = []

    # Preliminary checks
    if feed.feed_info is None:
        return problems

    f = feed.feed_info.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check feed_publisher_name
    problems = check_column(
        problems, table, f, "feed_publisher_name", valid_str
    )

    # Check feed_publisher_url
    problems = check_column(
        problems, table, f, "feed_publisher_url", valid_url
    )

    # Check feed_lang
    problems = check_column(problems, table, f, "feed_lang", valid_lang)

    # Check feed_start_date and feed_end_date
    cols = ["feed_start_date", "feed_end_date"]
    for col in cols:
        problems = check_column(
            problems, table, f, col, valid_date, column_required=False
        )

    if set(cols) <= set(f.columns):
        d1, d2 = f.loc[0, ["feed_start_date", "feed_end_date"]].values
        if pd.notna(d1) and pd.notna(d2) and d1 > d1:
            problems.append(
                [
                    "error",
                    "feed_start_date later than feed_end_date",
                    table,
                    [0],
                ]
            )

    # Check feed_version
    problems = check_column(
        problems, table, f, "feed_version", valid_str, column_required=False
    )

    return format_problems(problems, as_df=as_df)


def check_frequencies(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.frequencies``.
    """
    table = "frequencies"
    problems = []

    # Preliminary checks
    if feed.frequencies is None:
        return problems

    f = feed.frequencies.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_linked_id(
        problems, table, f, "trip_id", feed.trips
    )

    # Check start_time and end_time
    time_cols = ["start_time", "end_time"]
    for col in time_cols:
        problems = check_column(problems, table, f, col, valid_time)

    for col in time_cols:
        f[col] = f[col].map(hp.timestr_to_seconds)

    # Start_time should be earlier than end_time
    cond = f["start_time"] >= f["end_time"]
    problems = check_table(
        problems, table, f, cond, "start_time not earlier than end_time"
    )

    # Headway periods should not overlap
    f = f.sort_values(["trip_id", "start_time"])
    for __, group in f.groupby("trip_id"):
        a = group["start_time"].values
        b = group["end_time"].values
        indices = np.flatnonzero(a[1:] < b[:-1]).tolist()
        if indices:
            problems.append(
                [
                    "error",
                    "Headway periods for the same trip overlap",
                    table,
                    indices,
                ]
            )

    # Check headway_secs
    v = lambda x: x >= 0
    problems = check_column(problems, table, f, "headway_secs", v)

    # Check exact_times
    v = lambda x: x in range(2)
    problems = check_column(
        problems, table, f, "exact_times", v, column_required=False
    )

    return format_problems(problems, as_df=as_df)


def check_routes(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.routes``.
    """
    table = "routes"
    problems = []

    # Preliminary checks
    if feed.routes is None:
        problems.append(["error", "Missing table", table, []])
    else:
        f = feed.routes.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check route_id
    problems = check_column_id(problems, table, f, "route_id")

    # Check agency_id
    if "agency_id" in f:
        if feed.agency is None:
            problems.append(
                [
                    "error",
                    "agency_id column present in routes agency table missing",
                    table,
                    [],
                ]
            )
        elif "agency_id" not in feed.agency.columns:
            problems.append(
                [
                    "error",
                    "agency_id column present in routes but not in agency",
                    table,
                    [],
                ]
            )
        else:
            g = f.dropna(subset=["agency_id"])
            cond = ~g["agency_id"].isin(feed.agency["agency_id"])
            problems = check_table(
                problems, table, g, cond, "Undefined agency_id"
            )

    # Check route_short_name and route_long_name
    for column in ["route_short_name", "route_long_name"]:
        problems = check_column(
            problems, table, f, column, valid_str, column_required=False
        )

    cond = ~(f["route_short_name"].notna() | f["route_long_name"].notna())
    problems = check_table(
        problems,
        table,
        f,
        cond,
        "route_short_name and route_long_name both empty",
    )

    # Check route_type
    v = lambda x: x in range(8)
    problems = check_column(problems, table, f, "route_type", v)

    # Check route_url
    problems = check_column(
        problems, table, f, "route_url", valid_url, column_required=False
    )

    # Check route_color and route_text_color
    for col in ["route_color", "route_text_color"]:
        problems = check_column(
            problems, table, f, col, valid_color, column_required=False
        )

    if include_warnings:
        # Check for duplicated (route_short_name, route_long_name) pairs
        cond = f[["route_short_name", "route_long_name"]].duplicated()
        problems = check_table(
            problems,
            table,
            f,
            cond,
            "Repeated pair (route_short_name, route_long_name)",
            "warning",
        )

        # Check for routes without trips
        s = feed.trips["route_id"]
        cond = ~f["route_id"].isin(s)
        problems = check_table(
            problems, table, f, cond, "Route has no trips", "warning"
        )

    return format_problems(problems, as_df=as_df)


def check_shapes(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.shapes``.
    """
    table = "shapes"
    problems = []

    # Preliminary checks
    if feed.shapes is None:
        return problems

    f = feed.shapes.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)
    f.sort_values(["shape_id", "shape_pt_sequence"], inplace=True)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check shape_id
    problems = check_column(problems, table, f, "shape_id", valid_str)

    # Check shape_pt_lon and shape_pt_lat
    for column, bound in [("shape_pt_lon", 180), ("shape_pt_lat", 90)]:
        v = lambda x: pd.notna(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        problems = check_table(
            problems,
            table,
            f,
            cond,
            f"{column} out of bounds {[-bound, bound]}",
        )

    # Check for duplicated (shape_id, shape_pt_sequence) pairs
    cond = f[["shape_id", "shape_pt_sequence"]].duplicated()
    problems = check_table(
        problems, table, f, cond, "Repeated pair (shape_id, shape_pt_sequence)"
    )

    # Check if shape_dist_traveled does decreases on a trip
    if "shape_dist_traveled" in f.columns:
        g = f.dropna(subset=["shape_dist_traveled"])
        indices = []
        prev_sid = None
        prev_index = None
        prev_dist = -1
        cols = ["shape_id", "shape_dist_traveled"]
        for i, sid, dist in g[cols].itertuples():
            if sid == prev_sid and dist < prev_dist:
                indices.append(prev_index)

            prev_sid = sid
            prev_index = i
            prev_dist = dist

        if indices:
            problems.append(
                [
                    "error",
                    "shape_dist_traveled decreases on a trip",
                    table,
                    indices,
                ]
            )

    return format_problems(problems, as_df=as_df)


def check_stops(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.stops``.
    """
    table = "stops"
    problems = []

    # Preliminary checks
    if feed.stops is None:
        problems.append(["error", "Missing table", table, []])
    else:
        f = feed.stops.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check stop_id
    problems = check_column_id(problems, table, f, "stop_id")

    # Check stop_code, stop_desc, zone_id, parent_station
    for column in ["stop_code", "stop_desc", "zone_id", "parent_station"]:
        problems = check_column(
            problems, table, f, column, valid_str, column_required=False
        )

    # Check stop_name
    problems = check_column(problems, table, f, "stop_name", valid_str)

    # Check stop_lon and stop_lat
    if "location_type" in f.columns:
        requires_location = f.location_type.isin([0, 1, 2])
    else:
        requires_location = True
    for column, bound in [("stop_lon", 180), ("stop_lat", 90)]:
        v = lambda x: pd.notna(x) and -bound <= x <= bound
        cond = requires_location & ~f[column].map(v)
        problems = check_table(
            problems,
            table,
            f,
            cond,
            f"{column} out of bounds {[-bound, bound]}",
        )

    # Check stop_url
    problems = check_column(
        problems, table, f, "stop_url", valid_url, column_required=False
    )

    # Check location_type
    v = lambda x: x in range(5)
    problems = check_column(
        problems, table, f, "location_type", v, column_required=False
    )

    # Check stop_timezone
    problems = check_column(
        problems,
        table,
        f,
        "stop_timezone",
        valid_timezone,
        column_required=False,
    )

    # Check wheelchair_boarding
    v = lambda x: x in range(3)
    problems = check_column(
        problems, table, f, "wheelchair_boarding", v, column_required=False
    )

    # Check further location_type and parent_station
    if "parent_station" in f.columns:
        if "location_type" not in f.columns:
            problems.append(
                [
                    "error",
                    "parent_station column present but location_type column missing",
                    table,
                    [],
                ]
            )
        else:
            # Parent stations must be well-defined
            S = set(f.stop_id) | {np.nan}
            v = lambda x: x in S
            problems = check_column(
                problems,
                table,
                f,
                "parent_station",
                v,
                "A parent station must be well-defined",
                column_required=False,
            )

            # Stations must have location type 1
            station_ids = f.loc[f.parent_station.notna(), "parent_station"]
            cond = f.stop_id.isin(station_ids) & (f.location_type != 1)
            problems = check_table(
                problems, table, f, cond, "A station must have location_type 1"
            )

            # Stations must not lie in stations
            cond = (f.location_type == 1) & f.parent_station.notna()
            problems = check_table(
                problems,
                table,
                f,
                cond,
                "A station must not lie in another station",
            )

            # Entrances (type 2), generic nodes (type 3) and boarding areas (type 4)
            # need to be part of a parent
            cond = f.location_type.isin([2, 3, 4]) & f.parent_station.isna()
            problems = check_table(
                problems,
                table,
                f,
                cond,
                "Entrances, nodes, and boarding areas must be part of a parent station",
            )

    if include_warnings:
        # Check for stops of location type 0 or NaN without stop times
        ids = []
        if feed.stop_times is not None:
            ids = feed.stop_times.stop_id.unique()

        cond = ~feed.stops.stop_id.isin(ids)
        if "location_type" in feed.stops.columns:
            cond &= f.location_type.isin([0, np.nan])

        problems = check_table(
            problems, table, f, cond, "Stop has no stop times", "warning"
        )

    return format_problems(problems, as_df=as_df)


def check_stop_times(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.stop_times``.
    """
    table = "stop_times"
    problems = []

    # Preliminary checks
    if feed.stop_times is None:
        problems.append(["error", "Missing table", table, []])
    else:
        f = feed.stop_times.copy().sort_values(["trip_id", "stop_sequence"])
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_linked_id(
        problems, table, f, "trip_id", feed.trips
    )

    # Check arrival_time and departure_time
    v = lambda x: pd.isna(x) or valid_time(x)
    for col in ["arrival_time", "departure_time"]:
        problems = check_column(problems, table, f, col, v)

    # Check that arrival and departure times exist for the first and last
    # stop of each trip and for each timepoint.
    # For feeds with many trips, iterating through the stop time rows is
    # faster than uisg groupby.
    if "timepoint" not in f.columns:
        f["timepoint"] = np.nan  # This will not mess up later timepoint check

    indices = []
    prev_tid = None
    prev_index = None
    prev_atime = 1
    prev_dtime = 1
    for i, tid, atime, dtime, tp in f[
        ["trip_id", "arrival_time", "departure_time", "timepoint"]
    ].itertuples():
        if tid != prev_tid:
            # Check last stop of previous trip
            if pd.isna(prev_atime) or pd.isna(prev_dtime):
                indices.append(prev_index)
            # Check first stop of current trip
            if pd.isna(atime) or pd.isna(dtime):
                indices.append(i)
        elif tp == 1 and (pd.isna(atime) or pd.isna(dtime)):
            # Failure at timepoint
            indices.append(i)

        prev_tid = tid
        prev_index = i
        prev_atime = atime
        prev_dtime = dtime

    if pd.isna(prev_atime) or pd.isna(prev_dtime):
        indices.append(prev_index)

    if indices:
        problems.append(
            [
                "error",
                "First/last/time point arrival/departure time missing",
                table,
                indices,
            ]
        )

    # Check stop_id
    problems = check_column_linked_id(
        problems, table, f, "stop_id", feed.stops
    )

    # Check for duplicated (trip_id, stop_sequence) pairs
    cond = f[["trip_id", "stop_sequence"]].dropna().duplicated()
    problems = check_table(
        problems, table, f, cond, "Repeated pair (trip_id, stop_sequence)"
    )

    # Check stop_headsign
    problems = check_column(
        problems, table, f, "stop_headsign", valid_str, column_required=False
    )

    # Check pickup_type and drop_off_type
    for col in ["pickup_type", "drop_off_type"]:
        v = lambda x: x in range(4)
        problems = check_column(
            problems, table, f, col, v, column_required=False
        )

    # Check if shape_dist_traveled decreases on a trip
    if "shape_dist_traveled" in f.columns:
        g = f.dropna(subset=["shape_dist_traveled"])
        indices = []
        prev_tid = None
        prev_dist = -1
        for i, tid, dist in g[["trip_id", "shape_dist_traveled"]].itertuples():
            if tid == prev_tid and dist < prev_dist:
                indices.append(i)

            prev_tid = tid
            prev_dist = dist

        if indices:
            problems.append(
                [
                    "error",
                    "shape_dist_traveled decreases on a trip",
                    table,
                    indices,
                ]
            )

    # Check timepoint
    v = lambda x: x in range(2)
    problems = check_column(
        problems, table, f, "timepoint", v, column_required=False
    )

    if include_warnings:
        # Check for duplicated (trip_id, departure_time) pairs
        cond = f[["trip_id", "departure_time"]].duplicated()
        problems = check_table(
            problems,
            table,
            f,
            cond,
            "Repeated pair (trip_id, departure_time)",
            "warning",
        )

    return format_problems(problems, as_df=as_df)


def check_transfers(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.transfers``.
    """
    table = "transfers"
    problems = []

    # Preliminary checks
    if feed.transfers is None:
        return problems

    f = feed.transfers.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check from_stop_id and to_stop_id
    for col in ["from_stop_id", "to_stop_id"]:
        problems = check_column_linked_id(
            problems, table, f, col, feed.stops, "stop_id"
        )

    # Check transfer_type
    v = lambda x: pd.isna(x) or x in range(5)
    problems = check_column(
        problems, table, f, "transfer_type", v, column_required=False
    )

    # Check min_transfer_time
    v = lambda x: x >= 0
    problems = check_column(
        problems, table, f, "min_transfer_time", v, column_required=False
    )

    return format_problems(problems, as_df=as_df)


def check_trips(
    feed: "Feed", *, as_df: bool = False, include_warnings: bool = False
) -> List:
    """
    Analog of :func:`check_agency` for ``feed.trips``.
    """
    table = "trips"
    problems = []

    # Preliminary checks
    if feed.trips is None:
        problems.append(["error", "Missing table", table, []])
    else:
        f = feed.trips.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df=as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_id(problems, table, f, "trip_id")

    # Check route_id
    problems = check_column_linked_id(
        problems, table, f, "route_id", feed.routes
    )

    # Check service_id
    g = pd.DataFrame()
    if feed.calendar is not None:
        g = pd.concat([g, feed.calendar], sort=False)
    if feed.calendar_dates is not None:
        g = pd.concat([g, feed.calendar_dates], sort=False)
    problems = check_column_linked_id(problems, table, f, "service_id", g)

    # Check direction_id
    v = lambda x: x in range(2)
    problems = check_column(
        problems, table, f, "direction_id", v, column_required=False
    )

    # Check block_id
    if "block_id" in f.columns:
        v = lambda x: pd.isna(x) or valid_str(x)
        cond = ~f["block_id"].map(v)
        problems = check_table(problems, table, f, cond, "Blank block_id")

    # Check shape_id
    problems = check_column_linked_id(
        problems, table, f, "shape_id", feed.shapes, column_required=False
    )

    # Check wheelchair_accessible and bikes_allowed
    v = lambda x: x in range(3)
    for column in ["wheelchair_accessible", "bikes_allowed"]:
        problems = check_column(
            problems, table, f, column, v, column_required=False
        )

    # Check for trips with no stop times
    if include_warnings:
        s = feed.stop_times["trip_id"] if feed.stop_times is not None else []
        cond = ~f["trip_id"].isin(s)
        problems = check_table(
            problems, table, f, cond, "Trip has no stop times", "warning"
        )

    return format_problems(problems, as_df=as_df)


def validate(
    feed: "Feed", *, as_df: bool = True, include_warnings: bool = True
) -> Union[List, DataFrame]:
    """
    Check whether the given feed satisfies the GTFS.

    Parameters
    ----------
    feed : Feed
    as_df : boolean
        If ``True``, then return the resulting report as a DataFrame;
        otherwise return the result as a list
    include_warnings : boolean
        If ``True``, then include problems of types ``'error'`` and
        ``'warning'``; otherwise, only return problems of type
        ``'error'``

    Returns
    -------
    list or DataFrame
        Run all the table-checking functions: :func:`check_agency`,
        :func:`check_calendar`, etc.
        This yields a possibly empty list of items
        [problem type, message, table, rows].
        If ``as_df``, then format the error list as a DataFrame with the
        columns

        - ``'type'``: 'error' or 'warning'; 'error' means the GTFS is
          violated; 'warning' means there is a problem but it's not a
          GTFS violation
        - ``'message'``: description of the problem
        - ``'table'``: table in which problem occurs, e.g. 'routes'
        - ``'rows'``: rows of the table's DataFrame where problem occurs

        Return early if the feed is missing required tables or required
        columns.

    Notes
    -----
    - This function interprets the GTFS liberally, classifying problems
      as warnings rather than errors where the GTFS is unclear.
      For example if a trip_id listed in the trips table is not listed
      in the stop times table (a trip with no stop times),
      then that's a warning and not an error.
    - Timing benchmark: on a 2.80 GHz processor machine with 16 GB of
      memory, this function checks `this 31 MB Southeast Queensland feed
      <http://transitfeeds.com/p/translink/21/20170310>`_
      in 22 seconds, including warnings.

    """
    problems = []

    # Check for invalid columns and check the required tables
    checkers = [
        "check_agency",
        "check_calendar",
        "check_calendar_dates",
        "check_fare_attributes",
        "check_fare_rules",
        "check_feed_info",
        "check_frequencies",
        "check_routes",
        "check_shapes",
        "check_stops",
        "check_stop_times",
        "check_transfers",
        "check_trips",
    ]
    for checker in checkers:
        problems.extend(
            globals()[checker](feed, include_warnings=include_warnings)
        )

    # Check calendar/calendar_dates combo
    if feed.calendar is None and feed.calendar_dates is None:
        problems.append(
            ["error", "Missing both tables", "calendar & calendar_dates", []]
        )

    return format_problems(problems, as_df=as_df)
