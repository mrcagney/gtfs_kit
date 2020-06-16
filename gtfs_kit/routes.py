"""
Functions about routes.
"""
from collections import OrderedDict
from typing import Optional, Iterable, List, Dict, TYPE_CHECKING
import json

import geopandas as gp
import pandas as pd
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import folium as fl

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def compute_route_stats_0(
    trip_stats_subset: pd.DataFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats for the given subset of trips stats (of the form output by the
    function :func:`.trips.compute_trip_stats`).

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``
    - ``'num_trips'``: number of trips on the route in the subset
    - ``'num_trip_starts'``: number of trips on the route with
      nonnull start times
    - ``'num_trip_ends'``: number of trips on the route with nonnull
      end times that end before 23:59:59
    - ``'is_loop'``: 1 if at least one of the trips on the route has
      its ``is_loop`` field equal to 1; 0 otherwise
    - ``'is_bidirectional'``: 1 if the route has trips in both
      directions; 0 otherwise
    - ``'start_time'``: start time of the earliest trip on the route
    - ``'end_time'``: end time of latest trip on the route
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip starts on the route between
      ``headway_start_time`` and ``headway_end_time`` on the given
      dates
    - ``'min_headway'``: minimum of the durations (in minutes)
      mentioned above
    - ``'mean_headway'``: mean of the durations (in minutes)
      mentioned above
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service (for the given direction, or for both directions when
      ``split_directions==False``)
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs
    - ``'service_duration'``: total of the duration of each trip on
      the route in the given subset of trips; measured in hours
    - ``'service_distance'``: total of the distance traveled by each
      trip on the route in the given subset of trips; measured in
      whatever distance units are present in ``trip_stats_subset``;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_speed'``: service_distance/service_duration;
      measured in distance units per hour
    - ``'mean_trip_distance'``: service_distance/num_trips
    - ``'mean_trip_duration'``: service_duration/num_trips

    If not ``split_directions``, then remove the
    direction_id column and compute each route's stats,
    except for headways, using its trips running in both directions.
    In this case, (1) compute max headway by taking the max of the
    max headways in both directions; (2) compute mean headway by
    taking the weighted mean of the mean headways in both
    directions.

    If ``trip_stats_subset`` is empty, return an empty DataFrame.

    Raise a ValueError if ``split_directions`` and no non-NaN
    direction ID values present
    """
    if trip_stats_subset.empty:
        return pd.DataFrame()

    # Convert trip start and end times to seconds to ease calculations below
    f = trip_stats_subset.copy()
    f[["start_time", "end_time"]] = f[["start_time", "end_time"]].applymap(
        hp.timestr_to_seconds
    )

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    def compute_route_stats_split_directions(group):
        # Take this group of all trips stats for a single route
        # and compute route-level stats.
        d = OrderedDict()
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["num_trips"] = group.shape[0]
        d["num_trip_starts"] = group["start_time"].count()
        d["num_trip_ends"] = group.loc[
            group["end_time"] < 24 * 3600, "end_time"
        ].count()
        d["is_loop"] = int(group["is_loop"].any())
        d["start_time"] = group["start_time"].min()
        d["end_time"] = group["end_time"].max()

        # Compute max and mean headway
        stimes = group["start_time"].values
        stimes = sorted(
            [stime for stime in stimes if headway_start <= stime <= headway_end]
        )
        headways = np.diff(stimes)
        if headways.size:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan

        # Compute peak num trips
        active_trips = hp.get_active_trips_df(group[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        d["service_distance"] = group["distance"].sum()
        d["service_duration"] = group["duration"].sum()

        return pd.Series(d)

    def compute_route_stats(group):
        d = OrderedDict()
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["num_trips"] = group.shape[0]
        d["num_trip_starts"] = group["start_time"].count()
        d["num_trip_ends"] = group.loc[
            group["end_time"] < 24 * 3600, "end_time"
        ].count()
        d["is_loop"] = int(group["is_loop"].any())
        d["is_bidirectional"] = int(group["direction_id"].unique().size > 1)
        d["start_time"] = group["start_time"].min()
        d["end_time"] = group["end_time"].max()

        # Compute headway stats
        headways = np.array([])
        for direction in [0, 1]:
            stimes = group[group["direction_id"] == direction]["start_time"].values
            stimes = sorted(
                [stime for stime in stimes if headway_start <= stime <= headway_end]
            )
            headways = np.concatenate([headways, np.diff(stimes)])
        if headways.size:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan

        # Compute peak num trips
        active_trips = hp.get_active_trips_df(group[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        d["service_distance"] = group["distance"].sum()
        d["service_duration"] = group["duration"].sum()

        return pd.Series(d)

    if split_directions:
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError(
                "At least one trip stats direction ID value " "must be non-NaN."
            )

        g = (
            f.groupby(["route_id", "direction_id"])
            .apply(compute_route_stats_split_directions)
            .reset_index()
        )

        # Add the is_bidirectional column
        def is_bidirectional(group):
            d = {}
            d["is_bidirectional"] = int(group["direction_id"].unique().size > 1)
            return pd.Series(d)

        gg = g.groupby("route_id").apply(is_bidirectional).reset_index()
        g = g.merge(gg)
    else:
        g = f.groupby("route_id").apply(compute_route_stats).reset_index()

    # Compute a few more stats
    g["service_speed"] = (g["service_distance"] / g["service_duration"]).fillna(
        g["service_distance"]
    )
    g["mean_trip_distance"] = g["service_distance"] / g["num_trips"]
    g["mean_trip_duration"] = g["service_duration"] / g["num_trips"]

    # Convert route times to time strings
    g[["start_time", "end_time", "peak_start_time", "peak_end_time"]] = g[
        ["start_time", "end_time", "peak_start_time", "peak_end_time"]
    ].applymap(lambda x: hp.timestr_to_seconds(x, inverse=True))

    return g


def compute_route_time_series_0(
    trip_stats_subset: pd.DataFrame,
    date_label: str = "20010101",
    freq: str = "5Min",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats in a 24-hour time series form for the given subset of trips (of the
    form output by the function :func:`.trips.compute_trip_stats`).

    If ``split_directions``, then separate each routes's stats by trip direction.
    Set the time series frequency according to the given frequency string;
    max frequency is one minute ('Min').
    Use the given YYYYMMDD date label as the date in the time series index.

    Return a DataFrame time series version the following route stats for each route.

    - ``num_trips``: number of trips in service on the route
      at any time within the time bin
    - ``num_trip_starts``: number of trips that start within
      the time bin
    - ``num_trip_ends``: number of trips that end within the
      time bin, ignoring trips that end past midnight
    - ``service_distance``: sum of the service duration accrued
      during the time bin across all trips on the route;
      measured in hours
    - ``service_distance``: sum of the service distance accrued
      during the time bin across all trips on the route; measured
      in kilometers
    - ``service_speed``: ``service_distance/service_duration``
      for the route

    The columns are hierarchical (multi-indexed) with

    - top level: name is ``'indicator'``; values are
      ``'num_trip_starts'``, ``'num_trip_ends'``, ``'num_trips'``,
      ``'service_distance'``, ``'service_duration'``, and
      ``'service_speed'``
    - middle level: name is ``'route_id'``;
      values are the active routes
    - bottom level: name is ``'direction_id'``; values are 0s and 1s

    If not ``split_directions``, then don't include the bottom level.

    The time series has a timestamp index for a 24-hour period
    sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    If ``trip_stats_subset`` is empty, then return an empty
    DataFrame with the columns ``'num_trip_starts'``,
    ``'num_trip_ends'``, ``'num_trips'``, ``'service_distance'``,
    ``'service_duration'``, and ``'service_speed'``.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Trips that lack start or end times are ignored, so the the
      aggregate ``num_trips`` across the day could be less than the
      ``num_trips`` column of :func:`compute_route_stats_0`
    - All trip departure times are taken modulo 24 hours.
      So routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series,
      except for their ``num_trip_ends`` indicator.
      Trip endings past 23:59:59 not binned so that resampling the
      ``num_trips`` indicator works efficiently.
    - Note that the total number of trips for two consecutive time bins
      t1 < t2 is the sum of the number of trips in bin t2 plus the
      number of trip endings in bin t1.
      Thus we can downsample the ``num_trips`` indicator by keeping
      track of only one extra count, ``num_trip_ends``, and can avoid
      recording individual trip IDs.
    - All other indicators are downsampled by summing.
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

    """
    if trip_stats_subset.empty:
        return pd.DataFrame()

    tss = trip_stats_subset.copy()
    if split_directions:
        tss = tss.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if tss.empty:
            raise ValueError(
                "At least one trip stats direction ID value " "must be non-NaN."
            )

        # Alter route IDs to encode direction:
        # <route ID>-0 and <route ID>-1 or <route ID>-NA
        tss["route_id"] = (
            tss["route_id"] + "-" + tss["direction_id"].map(lambda x: str(int(x)))
        )

    routes = tss["route_id"].unique()
    # Build a dictionary of time series and then merge them all
    # at the end.
    # Assign a uniform generic date for the index
    date_str = date_label
    day_start = pd.to_datetime(date_str + " 00:00:00")
    day_end = pd.to_datetime(date_str + " 23:59:00")
    rng = pd.period_range(day_start, day_end, freq="Min")
    indicators = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
    ]

    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    # Bin start and end times
    def F(x):
        return (hp.timestr_to_seconds(x) // 60) % (24 * 60)

    tss[["start_index", "end_index"]] = tss[["start_time", "end_time"]].applymap(F)
    routes = sorted(set(tss["route_id"].values))

    # Bin each trip according to its start and end time and weight
    series_by_route_by_indicator = {
        indicator: {route: [0 for i in range(num_bins)] for route in routes}
        for indicator in indicators
    }
    for index, row in tss.iterrows():
        route = row["route_id"]
        start = row["start_index"]
        end = row["end_index"]
        distance = row["distance"]

        if start is None or np.isnan(start) or start == end:
            continue

        # Get bins to fill
        if start <= end:
            bins_to_fill = bins[start:end]
        else:
            bins_to_fill = bins[start:] + bins[:end]

        # Bin trip.
        # Do num trip starts.
        series_by_route_by_indicator["num_trip_starts"][route][start] += 1
        # Don't mark trip ends for trips that run past midnight;
        # allows for easy resampling of num_trips later
        if start <= end:
            series_by_route_by_indicator["num_trip_ends"][route][end] += 1
        # Do rest of indicators
        for indicator in indicators[2:]:
            if indicator == "num_trips":
                weight = 1
            elif indicator == "service_duration":
                weight = 1 / 60
            else:
                weight = distance / len(bins_to_fill)
            for bin in bins_to_fill:
                series_by_route_by_indicator[indicator][route][bin] += weight

    # Create one time series per indicator
    rng = pd.date_range(date_str, periods=24 * 60, freq="Min")
    series_by_indicator = {
        indicator: pd.DataFrame(
            series_by_route_by_indicator[indicator], index=rng
        ).fillna(0)
        for indicator in indicators
    }

    # Combine all time series into one time series
    g = hp.combine_time_series(
        series_by_indicator, kind="route", split_directions=split_directions
    )

    return hp.downsample(g, freq=freq)


def get_routes(
    feed: "Feed", date: Optional[str] = None, time: Optional[str] = None
) -> pd.DataFrame:
    """
    Return ``feed.routes`` or a subset thereof.
    If a YYYYMMDD date string is given, then restrict routes to only those active on
    the date.
    If a HH:MM:SS time string is given, possibly with HH > 23, then restrict routes
    to only those active during the time.
    """
    if date is None:
        return feed.routes.copy()

    trips = feed.get_trips(date, time)
    R = trips["route_id"].unique()
    return feed.routes[feed.routes["route_id"].isin(R)]


def compute_route_stats(
    feed: "Feed",
    trip_stats_subset: pd.DataFrame,
    dates: List[str],
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute route stats for all the trips that lie in the given subset
    of trip stats (of the form output by the function :func:`.trips.compute_trip_stats`)
    and that start on the given dates (YYYYMMDD date strings).

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'date'``
    - the columns listed in :func:``compute_route_stats_0``

    Exclude dates with no active trips, which could yield the empty DataFrame.

    Notes
    -----
    - The route stats for date d contain stats for trips that start on
      date d only and ignore trips that start on date d-1 and end on
      date d
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

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
            t = trip_stats_subset.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = (
                compute_route_stats_0(
                    t,
                    split_directions=split_directions,
                    headway_start_time=headway_start_time,
                    headway_end_time=headway_end_time,
                )
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


def build_zero_route_time_series(
    feed: "Feed",
    date_label: str = "20010101",
    freq: str = "5Min",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Return a route time series with the same index and hierarchical columns
    as output by :func:`compute_route_time_series_0`,
    but fill it full of zero values.
    """
    start = date_label
    end = pd.to_datetime(date_label + " 23:59:00")
    rng = pd.date_range(start, end, freq=freq)
    inds = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
        "service_speed",
    ]
    rids = feed.routes.route_id
    if split_directions:
        product = [inds, rids, [0, 1]]
        names = ["indicator", "route_id", "direction_id"]
    else:
        product = [inds, rids]
        names = ["indicator", "route_id"]
    cols = pd.MultiIndex.from_product(product, names=names)
    return pd.DataFrame([[0 for c in cols]], index=rng, columns=cols).sort_index(
        axis="columns"
    )


def compute_route_time_series(
    feed: "Feed",
    trip_stats_subset: pd.DataFrame,
    dates: List[str],
    freq: str = "5Min",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute route stats in time series form for the trips that lie in
    the trip stats subset (of the form output by the function
    :func:`.trips.compute_trip_stats`) and that start on the given dates
    (YYYYMMDD date strings).

    If ``split_directions``, then separate each routes's stats by trip direction.
    Specify the time series frequency with a Pandas frequency string, e.g. ``'5Min'``;
    max frequency is one minute ('Min').

    Return a DataFrame of the same format output by the function
    :func:`compute_route_time_series_0` but with multiple dates

    Exclude dates that lie outside of the Feed's date range.
    If all dates lie outside the Feed's date range, then return an
    empty DataFrame.

    Notes
    -----
    - See the notes for :func:`compute_route_time_series_0`
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    activity = feed.compute_trip_activity(dates)
    ts = trip_stats_subset.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary re-computations.
    # Store in dictionary of the form
    # trip ID sequence ->
    # [stats DataFarme, date list that stats apply]
    stats_by_ids = {}
    zero_stats = build_zero_route_time_series(
        feed, split_directions=split_directions, freq=freq
    )
    for date in dates:
        ids = tuple(activity.loc[activity[date] > 0, "trip_id"])
        if ids in stats_by_ids:
            # Append date to date list
            stats_by_ids[ids][1].append(date)
        elif not ids:
            # Null stats
            stats_by_ids[ids] = [zero_stats, [date]]
        else:
            # Compute stats
            t = ts[ts["trip_id"].isin(ids)].copy()
            stats = compute_route_time_series_0(
                t, split_directions=split_directions, freq=freq, date_label=date
            )

            # Remember stats
            stats_by_ids[ids] = [stats, [date]]

    # Assemble stats into DataFrame
    frames = []
    for stats, dates_ in stats_by_ids.values():
        for date in dates_:
            f = stats.copy()
            # Replace date
            d = hp.datestr_to_date(date)
            f.index = f.index.map(
                lambda t: t.replace(year=d.year, month=d.month, day=d.day)
            )
            frames.append(f)

    f = pd.concat(frames).sort_index().sort_index(axis="columns")

    if len(dates) > 1:
        # Insert missing dates and zeros to complete series index
        end_datetime = pd.to_datetime(dates[-1] + " 23:59:59")
        new_index = pd.date_range(dates[0], end_datetime, freq=freq)
        f = f.reindex(new_index)
    else:
        # Set frequency
        f.index.freq = pd.tseries.frequencies.to_offset(freq)

    return f.rename_axis("datetime", axis="index")


def build_route_timetable(
    feed: "Feed", route_id: str, dates: List[str]
) -> pd.DataFrame:
    """
    Return a timetable for the given route and dates (YYYYMMDD date strings).

    Return a DataFrame with whose columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times`` plus ``'date'``.
    The trip IDs are restricted to the given route ID.
    The result is sorted first by date and then by grouping by
    trip ID and sorting the groups by their first departure time.

    Skip dates outside of the Feed's dates.

    If there is no route activity on the given dates, then return
    an empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    t = pd.merge(feed.trips, feed.stop_times)
    t = t[t["route_id"] == route_id].copy()
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to trips active on date
        ids = a.loc[a[date] == 1, "trip_id"]
        f = t[t["trip_id"].isin(ids)].copy()
        f["date"] = date
        # Groupby trip ID and sort groups by their minimum departure time.
        # For some reason NaN departure times mess up the transform below.
        # So temporarily fill NaN departure times as a workaround.
        f["dt"] = f["departure_time"].fillna(method="ffill")
        f["min_dt"] = f.groupby("trip_id")["dt"].transform(min)
        frames.append(f)

    f = pd.concat(frames)
    return f.sort_values(["date", "min_dt", "stop_sequence"]).drop(
        ["min_dt", "dt"], axis=1
    )


def geometrize_routes(
    feed: "Feed",
    route_ids: Optional[Iterable[str]] = None,
    *,
    use_utm: bool = False,
    split_directions: bool = False,
) -> gp.GeoDataFrame:
    """
    Given a Feed, return a GeoDataFrame with all the columns of ``feed.routes``
    plus a geometry column of (Multi)LineStrings, each of which represents the
    corresponding routes's shape.

    If an iterable of route IDs is given, then subset to those routes.
    If ``use_utm``, then use local UTM coordinates for the geometries.
    If ``split_directions``, then add the column ``direction_id`` and split each route
    route shapes into shapes in trip direction 0 and its shapes in trip direction 1.

    Raise a ValueError if the Feed has no shapes.
    """
    if feed.shapes is None:
        raise ValueError("This Feed has no shapes.")

    # Subset routes
    if route_ids is None:
        route_ids = feed.routes.route_id

    # Subset trips
    trip_ids = (
        feed.trips.loc[lambda x: x.route_id.isin(route_ids)]
        # Drop unnecessary duplicate shapes
        .drop_duplicates(subset="shape_id").loc[:, "trip_id"]
    )

    # Combine shape LineStrings within route and direction
    if split_directions:
        groupby_cols = ["route_id", "direction_id"]
    else:
        groupby_cols = ["route_id"]

    def merge_lines(group):
        d = {}
        d["geometry"] = so.linemerge(group.geometry.tolist())
        return pd.Series(d)

    if use_utm:
        lat, lon = feed.shapes[["shape_pt_lat", "shape_pt_lon"]].values[0]
        crs = hp.get_utm_crs(lat, lon)
    else:
        crs = cs.WGS84

    return (
        feed.geometrize_trips(trip_ids)
        .filter(["route_id", "direction_id", "geometry"])
        # GeoDataFrame disappears here
        .groupby(groupby_cols)
        .apply(merge_lines)
        .reset_index()
        .merge(feed.routes)
        .pipe(gp.GeoDataFrame, crs=crs)
    )


def routes_to_geojson(
    feed: "Feed",
    route_ids: Optional[Iterable[str]] = None,
    *,
    split_directions: bool = False,
    include_stops: bool = False,
) -> Dict:
    """
    Return a GeoJSON FeatureCollection of MultiLineString features representing this Feed's routes.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If ``include_stops``, then include the route stops as Point features .
    If an iterable of route IDs is given, then subset to those routes.
    If the subset is empty, then return a FeatureCollection with an empty list of
    features.
    If the Feed has no shapes, then raise a ValueError.
    If any of the given route IDs are not found in the feed, then raise a ValueError.
    """
    if route_ids is not None:
        D = set(route_ids) - set(feed.routes.route_id)
        if D:
            raise ValueError(f"Route IDs {D} not found in feed.")

    # Get routes
    g = geometrize_routes(feed, route_ids=route_ids, split_directions=split_directions)
    if g.empty:
        collection = {"type": "FeatureCollection", "features": []}
    else:
        collection = json.loads(g.to_json())

    # Get stops if desired
    if include_stops:
        if route_ids is not None:
            stop_ids = (
                feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
                .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
                .unique()
            )
        else:
            stop_ids = None

        stops_gj = feed.stops_to_geojson(stop_ids=stop_ids)
        collection["features"].extend(stops_gj["features"])

    return hp.drop_feature_ids(collection)


def map_routes(
    feed: "Feed",
    route_ids: Iterable[str],
    color_palette: List[str] = cs.COLORS_SET2,
    *,
    include_stops: bool = False,
):
    """
    Return a Folium map showing the given routes and (optionally)
    their stops.
    If any of the given route IDs are not found in the feed, then raise a ValueError.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron", prefer_canvas=True)

    # Create route colors
    n = len(route_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect route bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, route_id in enumerate(route_ids):
        collection = feed.routes_to_geojson(
            route_ids=[route_id], include_stops=include_stops
        )

        # Use route short name for group name if possible; otherwise use route ID
        route_name = route_id
        for f in collection["features"]:
            if "route_short_name" in f["properties"]:
                route_name = f["properties"]["route_short_name"]
                break

        group = fl.FeatureGroup(name=f"Route {route_name}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop
            if f["geometry"]["type"] == "Point":
                lon, lat = f["geometry"]["coordinates"]
                fl.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    fill=True,
                    color=color,
                    weight=1,
                    popup=fl.Popup(hp.make_html(prop)),
                ).add_to(group)

            # Add path
            else:
                prop["color"] = color
                path = fl.GeoJson(
                    f,
                    name=prop["route_short_name"],
                    style_function=lambda x: {"color": x["properties"]["color"]},
                )
                path.add_child(fl.Popup(hp.make_html(prop)))
                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    bounds2 = [bounds[1::-1], bounds[3:1:-1]]  # Folium expects this ordering
    my_map.fit_bounds(bounds2)

    return my_map
