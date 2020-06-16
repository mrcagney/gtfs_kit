"""
Functions about miscellany.
"""
from collections import OrderedDict
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np
import shapely.geometry as sg
import geopandas as gp

from . import helpers as hp
from . import constants as cs

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def summarize(feed: "Feed", table: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame summarizing all GTFS tables in the given feed
    or in the given table if specified.

    The resulting DataFrame has the following columns.

    - ``'table'``: name of the GTFS table, e.g. ``'stops'``
    - ``'column'``: name of a column in the table,
      e.g. ``'stop_id'``
    - ``'num_values'``: number of values in the column
    - ``'num_nonnull_values'``: number of nonnull values in the
      column
    - ``'num_unique_values'``: number of unique values in the
      column, excluding null values
    - ``'min_value'``: minimum value in the column
    - ``'max_value'``: maximum value in the column

    If the table is not in the feed, then return an empty DataFrame
    If the table is not valid, raise a ValueError
    """
    gtfs_tables = cs.GTFS_REF.table.unique()

    if table is not None:
        if table not in gtfs_tables:
            raise ValueError(f"{table} is not a GTFS table")
        else:
            tables = [table]
    else:
        tables = gtfs_tables

    frames = []
    for table in tables:
        f = getattr(feed, table)
        if f is None:
            continue

        def my_agg(col):
            d = {}
            d["column"] = col.name
            d["num_values"] = col.size
            d["num_nonnull_values"] = col.count()
            d["num_unique_values"] = col.nunique()
            d["min_value"] = col.dropna().min()
            d["max_value"] = col.dropna().max()
            return pd.Series(d)

        g = f.apply(my_agg).T.reset_index(drop=True)
        g["table"] = table
        frames.append(g)

    cols = [
        "table",
        "column",
        "num_values",
        "num_nonnull_values",
        "num_unique_values",
        "min_value",
        "max_value",
    ]

    if not frames:
        f = pd.DataFrame()
    else:
        f = pd.concat(frames)
        # Rearrange columns
        f = f[cols].copy()

    return f


def describe(feed: "Feed", sample_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of routes.
    Specialize some those indicators to the given YYYYMMDD sample date string,
    e.g. number of routes active on the date.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    """
    from . import calendar as cl

    d = OrderedDict()
    dates = cl.get_dates(feed)
    d["agencies"] = feed.agency["agency_name"].tolist()
    d["timezone"] = feed.agency["agency_timezone"].iat[0]
    d["start_date"] = dates[0]
    d["end_date"] = dates[-1]
    d["num_routes"] = feed.routes.shape[0]
    d["num_trips"] = feed.trips.shape[0]
    d["num_stops"] = feed.stops.shape[0]
    if feed.shapes is not None:
        d["num_shapes"] = feed.shapes["shape_id"].nunique()
    else:
        d["num_shapes"] = 0

    if sample_date is None or sample_date not in feed.get_dates():
        sample_date = cl.get_first_week(feed)[3]
    d["sample_date"] = sample_date
    d["num_routes_active_on_sample_date"] = feed.get_routes(sample_date).shape[0]
    trips = feed.get_trips(sample_date)
    d["num_trips_active_on_sample_date"] = trips.shape[0]
    d["num_stops_active_on_sample_date"] = feed.get_stops(sample_date).shape[0]
    f = pd.DataFrame(list(d.items()), columns=["indicator", "value"])

    return f


def assess_quality(feed: "Feed") -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of trips missing shapes.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    This function is odd but useful for seeing roughly how broken a feed is
    This function is not a GTFS validator.
    """
    d = OrderedDict()

    # Count duplicate route short names
    r = feed.routes
    dup = r.duplicated(subset=["route_short_name"])
    n = dup[dup].count()
    d["num_route_short_names_duplicated"] = n
    d["frac_route_short_names_duplicated"] = n / r.shape[0]

    # Count stop times missing shape_dist_traveled values
    st = feed.stop_times.sort_values(["trip_id", "stop_sequence"])
    if "shape_dist_traveled" in st.columns:
        # Count missing distances
        n = st[st["shape_dist_traveled"].isnull()].shape[0]
        d["num_stop_time_dists_missing"] = n
        d["frac_stop_time_dists_missing"] = n / st.shape[0]
    else:
        d["num_stop_time_dists_missing"] = st.shape[0]
        d["frac_stop_time_dists_missing"] = 1

    # Count direction_ids missing
    t = feed.trips
    if "direction_id" in t.columns:
        n = t[t["direction_id"].isnull()].shape[0]
        d["num_direction_ids_missing"] = n
        d["frac_direction_ids_missing"] = n / t.shape[0]
    else:
        d["num_direction_ids_missing"] = t.shape[0]
        d["frac_direction_ids_missing"] = 1

    # Count trips missing shapes
    if feed.shapes is not None:
        n = t[t["shape_id"].isnull()].shape[0]
    else:
        n = t.shape[0]
    d["num_trips_missing_shapes"] = n
    d["frac_trips_missing_shapes"] = n / t.shape[0]

    # Count missing departure times
    n = st[st["departure_time"].isnull()].shape[0]
    d["num_departure_times_missing"] = n
    d["frac_departure_times_missing"] = n / st.shape[0]

    # Count missing first departure times missing
    g = st.groupby("trip_id").first().reset_index()
    n = g[g["departure_time"].isnull()].shape[0]
    d["num_first_departure_times_missing"] = n
    d["frac_first_departure_times_missing"] = n / st.shape[0]

    # Count missing last departure times
    g = st.groupby("trip_id").last().reset_index()
    n = g[g["departure_time"].isnull()].shape[0]
    d["num_last_departure_times_missing"] = n
    d["frac_last_departure_times_missing"] = n / st.shape[0]

    # Opine
    if (
        (d["frac_first_departure_times_missing"] >= 0.1)
        or (d["frac_last_departure_times_missing"] >= 0.1)
        or d["frac_trips_missing_shapes"] >= 0.8
    ):
        d["assessment"] = "bad feed"
    elif (
        d["frac_direction_ids_missing"]
        or d["frac_stop_time_dists_missing"]
        or d["num_route_short_names_duplicated"]
    ):
        d["assessment"] = "probably a fixable feed"
    else:
        d["assessment"] = "good feed"

    f = pd.DataFrame(list(d.items()), columns=["indicator", "value"])

    return f


def convert_dist(feed: "Feed", new_dist_units: str) -> "Feed":
    """
    Convert the distances recorded in the ``shape_dist_traveled``
    columns of the given Feed to the given distance units.
    New distance units must lie in :const:`.constants.DIST_UNITS`.
    Return the resulting Feed.
    """
    feed = feed.copy()

    if feed.dist_units == new_dist_units:
        # Nothing to do
        return feed

    old_dist_units = feed.dist_units
    feed.dist_units = new_dist_units

    converter = hp.get_convert_dist(old_dist_units, new_dist_units)

    if hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        feed.stop_times["shape_dist_traveled"] = feed.stop_times[
            "shape_dist_traveled"
        ].map(converter)

    if hp.is_not_null(feed.shapes, "shape_dist_traveled"):
        feed.shapes["shape_dist_traveled"] = feed.shapes["shape_dist_traveled"].map(
            converter
        )

    return feed


def compute_feed_stats_0(
    feed: "Feed", trip_stats_subset: pd.DataFrame, *, split_route_types=False
) -> pd.DataFrame:
    """
    Helper function for :func:`compute_feed_stats`.
    """
    ts = trip_stats_subset.copy()
    stop_times = feed.stop_times.copy()

    # Convert timestrings to seconds for quicker calculations
    ts[["start_time", "end_time"]] = ts[["start_time", "end_time"]].applymap(
        hp.timestr_to_seconds
    )

    if split_route_types:
        # Compute stats
        stats_list = []
        for route_type, g in ts.groupby("route_type"):
            d = {}
            d["route_type"] = route_type
            d["num_stops"] = stop_times.loc[
                lambda x: x.trip_id.isin(g.trip_id), "stop_id"
            ].nunique()
            d["num_routes"] = g.route_id.nunique()
            d["num_trips"] = g.shape[0]
            d["num_trip_starts"] = g.start_time.count()
            d["num_trip_ends"] = g.loc[g.end_time < 24 * 3600, "end_time"].count()
            d["service_distance"] = g.distance.sum()
            d["service_duration"] = g.duration.sum()
            if d["service_distance"]:
                d["service_speed"] = d["service_distance"] / d["service_duration"]
            else:
                d["service_speed"] = 0

            # Compute peak stats, which is the slowest part
            active_trips = hp.get_active_trips_df(g[["start_time", "end_time"]])
            times, counts = (active_trips.index.values, active_trips.values)
            start, end = hp.get_peak_indices(times, counts)
            d["peak_num_trips"] = counts[start]
            d["peak_start_time"] = times[start]
            d["peak_end_time"] = times[end]

            stats_list.append(pd.Series(d))

        stats = pd.DataFrame(stats_list)

    else:
        # Compute stats
        d = {}
        d["num_stops"] = stop_times.stop_id.nunique()
        d["num_routes"] = ts.route_id.nunique()
        d["num_trips"] = ts.shape[0]
        d["num_trip_starts"] = ts.start_time.count()
        d["num_trip_ends"] = ts.loc[ts.end_time < 24 * 3600, "end_time"].count()
        d["service_distance"] = ts.distance.sum()
        d["service_duration"] = ts.duration.sum()
        d["service_speed"] = d["service_distance"] / d["service_duration"]

        # Compute peak stats, which is the slowest part
        active_trips = hp.get_active_trips_df(ts[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        stats = pd.DataFrame(d, index=[0])

    # Convert seconds back to timestrings
    times = ["peak_start_time", "peak_end_time"]
    stats[times] = stats[times].applymap(
        lambda t: hp.timestr_to_seconds(t, inverse=True)
    )

    return stats


def compute_feed_stats(
    feed: "Feed",
    trip_stats: pd.DataFrame,
    dates: List[str],
    *,
    split_route_types=False,
) -> pd.DataFrame:
    """
    Compute some stats for the given Feed, trip stats (in the format output by the
    function :func:`.trips.compute_trip_stats`) and dates (YYYYMMDD date stings).

    Return a DataFrame with the columns

    - ``'date'``
    - ``'route_type'`` (optional): presest if and only if ``split_route_types``
    - ``'num_stops'``: number of stops active on the date
    - ``'num_routes'``: number of routes active on the date
    - ``'num_trips'``: number of trips that start on the date
    - ``'num_trip_starts'``: number of trips with nonnull start
      times on the date
    - ``'num_trip_ends'``: number of trips with nonnull start times
      and nonnull end times on the date, ignoring trips that end
      after 23:59:59 on the date
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service on the date
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs on the date
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs on the date
    - ``'service_distance'``: sum of the service distances for the
      active routes on the date
    - ``'service_duration'``: sum of the service durations for the
      active routes on the date
    - ``'service_speed'``: service_distance/service_duration on the
      date

    Exclude dates with no active stops, which could yield the empty DataFrame.

    The route and trip stats for date d contain stats for trips that
    start on date d only and ignore trips that start on date d-1 and
    end on date d.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFarme.
    stats_by_ids = {}

    activity = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        ids = tuple(activity.loc[activity[date] > 0, "trip_id"])
        if ids in stats_by_ids:
            stats = (
                stats_by_ids[ids]
                # Assign date
                .assign(date=date)
            )
        elif ids:
            # Compute stats
            ts = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = (
                compute_feed_stats_0(feed, ts, split_route_types=split_route_types)
                # Assign date
                .assign(date=date)
            )

            # Memoize stats
            stats_by_ids[ids] = stats
        else:
            stats = pd.DataFrame()

        frames.append(stats)

    # Assemble stats into a single DataFrame
    return pd.concat(frames)


def compute_feed_time_series(
    feed: "Feed",
    trip_stats: pd.DataFrame,
    dates: List[str],
    freq: str = "5Min",
    *,
    split_route_types: bool = False,
) -> pd.DataFrame:
    """
    Compute some feed stats in time series form for the given dates
    (YYYYMMDD date strings) and trip stats (of the form output by the function
    :func:`.trips.compute_trip_stats`).
    Use the given Pandas frequency string ``freq`` to specify the frequency of the
    resulting time series, e.g. '5Min'; highest frequency allowable is one minute
    ('1Min').
    If ``split_route_types``, then split stats by route type; otherwise don't

    Return a time series DataFrame with a datetime index across the given dates sampled
    at the given frequency across the given dates.
    The columns are

    - ``'num_trips'``: number of trips in service during during the
      time period
    - ``'num_trip_starts'``: number of trips with starting during the
      time period
    - ``'num_trip_ends'``: number of trips ending during the
      time period, ignoring the trips the end past midnight
    - ``'service_distance'``: distance traveled during the time
      period by all trips active during the time period
    - ``'service_duration'``: duration traveled during the time
      period by all trips active during the time period
    - ``'service_speed'``: ``service_distance/service_duration``

    Exclude dates that lie outside of the Feed's date range.
    If all the dates given lie outside of the Feed's date range,
    then return an empty DataFrame with the specified columns.

    If ``split_route_types``, then multi-index the columns with

    - top level: name is ``'indicator'``; values are
      ``'num_trip_starts'``, ``'num_trip_ends'``, ``'num_trips'``,
      ``'service_distance'``, ``'service_duration'``, and
      ``'service_speed'``
    - bottom level: name is ``'route_type'``; values are route type values

    If all dates lie outside the Feed's date range, then return an
    empty DataFrame
    """
    rts = feed.compute_route_time_series(trip_stats, dates, freq=freq)
    if rts.empty:
        return pd.DataFrame()

    cols = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_distance",
        "service_duration",
    ]

    if split_route_types:
        f = (
            hp.unstack_time_series(rts)
            .merge(feed.routes.filter(["route_id", "route_type"]), how="left")
            .groupby(["datetime", "indicator", "route_type"])
            .agg({"value": lambda x: x.sum(min_count=1)})  # All-NaNs should sum to NaN
            .reset_index()
            .pipe(hp.restack_time_series)
        )
    else:
        f = (
            pd.concat(
                [rts[col].sum(axis="columns", min_count=1) for col in cols],
                axis=1,
                keys=cols,
            )
            .sort_index(axis="columns")
            .rename_axis(index="datetime")
        )
        f.columns.name = "indicator"

        # Set time series frequency
        f.index.freq = freq

    # Calculate service speed
    f["service_speed"] = (f.service_distance / f.service_duration).fillna(
        f.service_distance
    )

    return f


def create_shapes(feed: "Feed", *, all_trips: bool = False) -> "Feed":
    """
    Given a feed, create a shape for every trip that is missing a
    shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips
    tables.

    If ``all_trips``, then create new shapes for all trips by
    connecting stops, and remove the old shapes.
    """
    feed = feed.copy()

    if all_trips:
        trip_ids = feed.trips["trip_id"]
    else:
        trip_ids = feed.trips[feed.trips["shape_id"].isnull()]["trip_id"]

    # Get stop times for given trips
    f = feed.stop_times[feed.stop_times["trip_id"].isin(trip_ids)][
        ["trip_id", "stop_sequence", "stop_id"]
    ]
    f = f.sort_values(["trip_id", "stop_sequence"])

    if f.empty:
        # Nothing to do
        return feed

    # Create new shape IDs for given trips.
    # To do this, collect unique stop sequences,
    # sort them to impose a canonical order, and
    # assign shape IDs to them
    stop_seqs = sorted(
        set(tuple(group["stop_id"].values) for trip, group in f.groupby("trip_id"))
    )
    ids = hp.make_ids(len(stop_seqs), "shape_")
    shape_by_stop_seq = {seq: ids[i] for i, seq in enumerate(stop_seqs)}

    # Assign these new shape IDs to given trips
    shape_by_trip = {
        trip: shape_by_stop_seq[tuple(group["stop_id"].values)]
        for trip, group in f.groupby("trip_id")
    }
    trip_cond = feed.trips["trip_id"].isin(trip_ids)
    feed.trips.loc[trip_cond, "shape_id"] = feed.trips.loc[trip_cond, "trip_id"].map(
        lambda x: shape_by_trip[x]
    )

    # Build new shapes for given trips
    G = [
        [shape, i, stop]
        for stop_seq, shape in shape_by_stop_seq.items()
        for i, stop in enumerate(stop_seq)
    ]
    g = pd.DataFrame(G, columns=["shape_id", "shape_pt_sequence", "stop_id"])
    g = g.merge(feed.stops[["stop_id", "stop_lon", "stop_lat"]]).sort_values(
        ["shape_id", "shape_pt_sequence"]
    )
    g = g.drop(["stop_id"], axis=1)
    g = g.rename(columns={"stop_lon": "shape_pt_lon", "stop_lat": "shape_pt_lat"})

    if feed.shapes is not None and not all_trips:
        # Update feed shapes with new shapes
        feed.shapes = pd.concat([feed.shapes, g], sort=False)
    else:
        # Create all new shapes
        feed.shapes = g

    return feed


def compute_bounds(feed: "Feed", stop_ids: Optional[List[str]] = None) -> np.array:
    """
    Return the bounding box (Numpy array [min longitude, min latitude, max longitude,
    max latitude]) of the given Feed's stops or of the subset of stops
    specified by the given stop IDs.
    """
    from .stops import geometrize_stops

    return geometrize_stops(feed, stop_ids=stop_ids).total_bounds


def compute_convex_hull(
    feed: "Feed", stop_ids: Optional[List[str]] = None
) -> sg.Polygon:
    """
    Return a convex hull (Shapely Polygon) representing the convex hull of the given 
    Feed's stops or of the subset of stops specified by the given stop IDs.
    """
    from .stops import geometrize_stops

    return geometrize_stops(feed, stop_ids=stop_ids).unary_union.convex_hull


def compute_centroid(feed: "Feed", stop_ids: Optional[List[str]] = None) -> sg.Point:
    """
    Return the centroid (Shapely Point) of the convex hull the given Feed's stops
    or of the subset of stops specified by the given stop IDs.
    """
    from .stops import geometrize_stops

    return geometrize_stops(feed, stop_ids=stop_ids).unary_union.convex_hull.centroid


def restrict_to_dates(feed: "Feed", dates: List[str]) -> "Feed":
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. active on at least one of the given dates
    (YYYYMMDD strings).
    Return the resulting feed, which will have empty non-agency tables
    if no trip is active on any of the given dates.
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Get every trip that is active on at least one of the dates
    trip_activity = feed.compute_trip_activity(dates)
    if trip_activity.empty:
        trip_ids = []
    else:
        trip_ids = trip_activity.loc[
            lambda x: x.filter(dates).sum(axis=1) > 0, "trip_id",
        ]

    # Slice trips
    feed.trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Slice routes
    feed.routes = feed.routes.loc[lambda x: x.route_id.isin(feed.trips.route_id)].copy()

    # Slice stop times
    feed.stop_times = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times.stop_id.unique()
    f = feed.stops.copy()
    cond = f.stop_id.isin(stop_ids)
    if "location_type" in f.columns:
        cond |= ~f.location_type.isin([0, np.nan])
    feed.stops = f[cond].copy()

    # Slice calendar
    service_ids = feed.trips.service_id
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Get agency for trips
    if "agency_id" in feed.routes.columns:
        agency_ids = feed.routes.agency_id
        if len(agency_ids):
            feed.agency = feed.agency.loc[lambda x: x.agency_id.isin(agency_ids)].copy()

    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.loc[
            lambda x: x.service_id.isin(service_ids) & x.date.isin(dates)
        ].copy()

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[
            lambda x: x.trip_id.isin(trip_ids)
        ].copy()

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes.loc[lambda x: x.shape_id.isin(shape_ids)].copy()

    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers.loc[
            lambda x: x.from_stop_id.isin(stop_ids) & x.to_stop_id.isin(stop_ids)
        ].copy()

    return feed


def restrict_to_routes(feed: "Feed", route_ids: List[str]) -> "Feed":
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. used by the routes with the given list of
    route IDs.
    Return the resulting feed.
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Slice routes
    feed.routes = feed.routes.loc[lambda x: x.route_id.isin(route_ids)].copy()

    # Slice trips
    feed.trips = feed.trips.loc[lambda x: x.route_id.isin(route_ids)].copy()

    # Slice stop times
    trip_ids = feed.trips.trip_id
    feed.stop_times = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times.stop_id.unique()
    f = feed.stops.copy()
    cond = f.stop_id.isin(stop_ids)
    if "location_type" in f.columns:
        cond |= ~f.location_type.isin([0, np.nan])
    feed.stops = f[cond].copy()

    # Slice calendar
    service_ids = feed.trips.service_id
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Get agency for trips
    if "agency_id" in feed.routes.columns:
        agency_ids = feed.routes.agency_id
        if len(agency_ids):
            feed.agency = feed.agency.loc[lambda x: x.agency_id.isin(agency_ids)].copy()

    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[
            lambda x: x.trip_id.isin(trip_ids)
        ].copy()

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes[lambda x: x.shape_id.isin(shape_ids)].copy()

    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers.loc[
            lambda x: (x.from_stop_id.isin(stop_ids) & x.to_stop_id.isin(stop_ids))
        ].copy()

    return feed


def restrict_to_area(feed: "Feed", area: gp.GeoDataFrame) -> "Feed":
    """
    Build a new feed by restricting this one to only the trips
    that have at least one stop intersecting the given GeoDataFrame of polygons,
    then restricting stops, routes, stop times, etc. to those
    associated with that subset of trips.
    Return the resulting feed.
    """
    from .stops import get_stops_in_area

    feed = feed.copy()

    # Get IDs of stops within the polygon
    stop_ids = get_stops_in_area(feed, area).stop_id

    # Get all trips with at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st.loc[lambda x: x.stop_id.isin(stop_ids), "trip_id"]
    feed.trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Get stop times for trips
    feed.stop_times = st.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times.stop_id.unique()
    f = feed.stops.copy()
    cond = f.stop_id.isin(stop_ids)
    if "location_type" in f.columns:
        cond |= ~f.location_type.isin([0, np.nan])
    feed.stops = f[cond].copy()

    # Get routes for trips
    route_ids = feed.trips.route_id
    feed.routes = feed.routes.loc[lambda x: x.route_id.isin(route_ids)].copy()

    # Get calendar for trips
    service_ids = feed.trips.service_id
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Get agency for trips
    if "agency_id" in feed.routes.columns:
        agency_ids = feed.routes.agency_id
        if len(agency_ids):
            feed.agency = feed.agency.loc[lambda x: x.agency_id.isin(agency_ids)].copy()

    # Now for the optional files.
    # Get calendar dates for trips.
    cd = feed.calendar_dates
    if cd is not None:
        feed.calendar_dates = cd.loc[lambda x: x.service_id.isin(service_ids)].copy()

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[
            lambda x: x.trip_id.isin(trip_ids)
        ].copy()

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes.loc[lambda x: x.shape_id.isin(shape_ids)].copy()

    # Get transfers for stops
    if feed.transfers is not None:
        t = feed.transfers
        feed.transfers = t.loc[
            lambda x: x.from_stop_id.isin(stop_ids) & x.to_stop_id.isin(stop_ids)
        ].copy()

    return feed


def compute_screen_line_counts(
    feed: "Feed", screen_lines: gp.GeoDataFrame, dates: List[str]
) -> pd.DataFrame:
    """
    Find all the Feed trips active on the given YYYYMMDD dates whose shapes
    intersect the given GeoDataFrame of screen lines, that is, of straight WGS84
    LineStrings.
    Compute the intersection times and directions for each trip.

    Return a DataFrame with the columns

    - ``'date'``
    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'shape_id'``: shape ID of the trip
    - ``'screen_line_id'``: ID of the screen line as specified in ``screen_lines`` or as
      assigned after the fact.
    - ``'crossing_distance'``: distance along the trip shape of the screen line
      intersection
      ``'crossing_time'``: time that the trip's vehicle crosses
      the scren line; one trip could cross multiple times
    - ``'crossing_direction'``: 1 or -1; 1 indicates trip travel from the
      left side to the right side of the screen line;
      -1 indicates trip travel in the  opposite direction

    Notes:

    - Assume the Feed's stop times DataFrame has an accurate ``shape_dist_traveled`` column.
    - Assume that trips travel in the same direction as their shapes, an assumption
      that is part of the GTFS.
    - Assume that the screen line is straight and simple.
    - Probably does not give correct results for trips with self-intersecting shapes.
    - The algorithm works as follows

        1. Find the trip shapes that intersect the screen lines.
        2. For each such shape and screen line, compute the intersection points, the distance
           of the point along the shape, and the orientation of the screen line relative to
           the shape.
        3. For each given date, restrict to trips active on the
           date and interpolate a stop time for the intersection point using
           the ``shape_dist_traveled`` column.
        4. Use that interpolated time as the crossing time of the trip vehicle.

    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    # Get shapes as GeoDataFrame
    shapes_g = feed.geometrize_shapes(use_utm=True)

    # Convert screen lines to UTM
    crs = shapes_g.crs
    screen_lines = screen_lines.to_crs(crs)

    # Create screen line IDs if necessary
    n = screen_lines.shape[0]
    if "screen_line_id" not in screen_lines.columns:
        screen_lines["screen_line_id"] = hp.make_ids(n, "sl")

    # Make a vector in the direction of each screen line to calculate crossing orientation.
    # Does not work in case of a bent screen line.
    p1 = screen_lines.geometry.map(lambda x: np.array(x.coords[0]))
    p2 = screen_lines.geometry.map(lambda x: np.array(x.coords[-1]))
    screen_lines["screen_line_vector"] = p2 - p1

    # Get intersection points of shapes and screen lines
    g0 = (
        # Only keep shapes that intersect screen lines to reduce computations
        gp.sjoin(shapes_g, screen_lines.filter(["screen_line_id", "geometry"])).merge(
            screen_lines, on="screen_line_id"
        )
        # Compute intersection points
        .assign(
            int_point=lambda x: gp.GeoSeries(x.geometry_x, crs=crs).intersection(
                gp.GeoSeries(x.geometry_y, crs=crs)
            )
        )
    )

    # Unpack multipoint intersections to yield a new GeoDataFrame
    records = []
    for row in g0.itertuples(index=False):
        if isinstance(row.int_point, sg.Point):
            intersections = [row.int_point]
        else:
            intersections = row.int_point
        for int_point in intersections:
            record = {
                "shape_id": row.shape_id,
                "screen_line_id": row.screen_line_id,
                "geometry": row.geometry_x,
                "int_point": int_point,
                "screen_line_vector": row.screen_line_vector,
            }
            records.append(record)

    g = gp.GeoDataFrame.from_records(records)
    g.crs = crs

    # Get distance (in meters) of each intersection point along shape
    g["crossing_dist"] = g.apply(lambda x: x.geometry.project(x.int_point), axis=1)

    # Build a tiny vector along each shape
    p2 = g.apply(lambda x: x.geometry.interpolate(x.crossing_dist + 1), axis=1).map(
        lambda x: np.array(x.coords[0])
    )
    p1 = g.int_point.map(lambda x: np.array(x.coords[0]))
    g["shape_vector"] = p2 - p1

    # Compute crossing direction by taking the vector cross product of
    # the shape vector and the screen line vector
    det = g.apply(
        lambda x: np.linalg.det(np.array([x.shape_vector, x.screen_line_vector])),
        axis=1,
    )
    g["crossing_direction"] = det.map(lambda x: 1 if x >= 0 else -1)

    # Convert to feed distance units
    converter = hp.get_convert_dist("m", feed.dist_units)
    g["crossing_dist"] = g["crossing_dist"].map(converter)

    # Summarize work so far into a lookup table
    h = (
        g.filter(["shape_id", "screen_line_id", "crossing_direction", "crossing_dist"])
        .set_index("shape_id")
        .sort_values(
            ["shape_id", "crossing_dist"]
        )  # Need this sorting for interpolation to work
    )

    # Get stop times of trips whose shapes lie in h
    st = (
        feed.trips.loc[lambda x: x.shape_id.isin(h.index)]
        # Merge in route short names and stop times
        .merge(feed.routes[["route_id", "route_short_name"]]).merge(feed.stop_times)
        # Keep only non-NaN departure times
        .loc[lambda x: x.departure_time.notna()]
        # Convert to seconds past midnight
        .assign(departure_time=lambda x: x.departure_time.map(hp.timestr_to_seconds))
    )

    # Compute crossing times by date
    records = []
    ta = feed.compute_trip_activity(dates)
    for date in dates:
        # Subset to trips active on date and merge with g
        ids = ta.loc[lambda x: x[date] == 1, "trip_id"]
        f = st.loc[lambda x: x.trip_id.isin(ids)].sort_values(
            ["trip_id", "shape_dist_traveled"]
        )  # Need this sorting for interpolation to work

        # Get crossing time for each trip
        for tid, group in f.groupby("trip_id"):
            sid = group["shape_id"].iat[0]
            rid = group["route_id"].iat[0]
            rsn = group["route_short_name"].iat[0]
            dists = group["shape_dist_traveled"].values
            times = group["departure_time"].values
            crossing_dists = h.loc[[sid], "crossing_dist"].values
            crossing_times = np.interp(crossing_dists, dists, times)
            for i, row in enumerate(h.loc[[sid]].itertuples(index=False)):
                record = {
                    "date": date,
                    "trip_id": tid,
                    "route_id": group.route_id.iat[0],
                    "route_short_name": group.route_short_name.iat[0],
                    "shape_id": group.shape_id.iat[0],
                    "screen_line_id": row.screen_line_id,
                    "crossing_direction": row.crossing_direction,
                    "crossing_distance": row.crossing_dist,
                    "crossing_time": crossing_times[i],
                }
                records.append(record)

    result = pd.DataFrame.from_records(records).assign(
        crossing_time=lambda x: x.crossing_time.map(
            lambda x: hp.timestr_to_seconds(x, inverse=True)
        )
    )
    return result
