"""
Functions about stops.
"""
from collections import Counter, OrderedDict
from typing import Optional, Iterable, List, Dict, TYPE_CHECKING
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import utm
import shapely.geometry as sg
from shapely.geometry import Polygon

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


#: Folium CircleMarker parameters for mapping stops
STOP_STYLE = {
    "radius": 8,
    "fill": True,
    "color": cs.COLORS_SET2[1],
    "weight": 1,
    "fill_opacity": 0.75,
}


def compute_stop_stats_0(
    stop_times_subset: pd.DataFrame,
    trip_subset: pd.DataFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Given a subset of a stop times DataFrame and a subset of a trips
    DataFrame, return a DataFrame that provides summary stats about the
    stops in the inner join of the two DataFrames.

    Parameters
    ----------
    stop_times_subset : DataFrame
        A valid GTFS stop times table
    trip_subset : DataFrame
        A valid GTFS trips table
    split_directions : boolean
        If ``True``, then separate the stop stats by direction (0 or 1)
        of the trips visiting the stops; otherwise aggregate trips
        visiting from both directions
    headway_start_time : string
        HH:MM:SS time string indicating the start time for computing
        headway stats
    headway_end_time : string
        HH:MM:SS time string indicating the end time for computing
        headway stats

    Returns
    -------
    DataFrame
        The columns are

        - stop_id
        - direction_id: present if and only if ``split_directions``
        - num_routes: number of routes visiting stop
          (in the given direction)
        - num_trips: number of trips visiting stop
          (in the givin direction)
        - max_headway: maximum of the durations (in minutes)
          between trip departures at the stop between
          ``headway_start_time`` and ``headway_end_time``
        - min_headway: minimum of the durations (in minutes) mentioned
          above
        - mean_headway: mean of the durations (in minutes) mentioned
          above
        - start_time: earliest departure time of a trip from this stop
        - end_time: latest departure time of a trip from this stop


    Notes
    -----
    - If ``trip_subset`` is empty, then return an empty DataFrame.
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present.

    """
    if trip_subset.empty:
        return pd.DataFrame()

    f = pd.merge(stop_times_subset, trip_subset)

    # Convert departure times to seconds to ease headway calculations
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    # Compute stats for each stop
    def compute_stop_stats(group):
        # Operate on the group of all stop times for an individual stop
        d = OrderedDict()
        d["num_routes"] = group["route_id"].unique().size
        d["num_trips"] = group.shape[0]
        d["start_time"] = group["departure_time"].min()
        d["end_time"] = group["departure_time"].max()
        headways = []
        dtimes = sorted(
            [
                dtime
                for dtime in group["departure_time"].values
                if headway_start <= dtime <= headway_end
            ]
        )
        headways.extend(
            [dtimes[i + 1] - dtimes[i] for i in range(len(dtimes) - 1)]
        )
        if headways:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan
        return pd.Series(d)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = np.nan
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError(
                "At least one trip direction ID value " "must be non-NaN."
            )
        g = f.groupby(["stop_id", "direction_id"])
    else:
        g = f.groupby("stop_id")

    result = g.apply(compute_stop_stats).reset_index()

    # Convert start and end times to time strings
    result[["start_time", "end_time"]] = result[
        ["start_time", "end_time"]
    ].applymap(lambda x: hp.timestr_to_seconds(x, inverse=True))

    return result


def compute_stop_time_series_0(
    stop_times_subset: pd.DataFrame,
    trip_subset: pd.DataFrame,
    freq: str = "5Min",
    date_label: str = "20010101",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Given a subset of a stop times DataFrame and a subset of a trips
    DataFrame, return a DataFrame that provides a summary time series
    about the stops in the inner join of the two DataFrames.

    Parameters
    ----------
    stop_times_subset : DataFrame
        A valid GTFS stop times table
    trip_subset : DataFrame
        A valid GTFS trips table
    split_directions : boolean
        If ``True``, then separate each stop's stats by trip direction;
        otherwise aggregate trips visiting from both directions
    freq : Pandas frequency string
        Specifices the frequency with which to resample the time series;
        max frequency is one minute ('Min')
    date_label : string
        YYYYMMDD date string used as the date in the time series index

    Returns
    -------
    DataFrame
        A time series with a timestamp index for a 24-hour period
        sampled at the given frequency.
        The only indicator variable for each stop is

        - ``num_trips``: the number of trips that visit the stop and
          have a nonnull departure time from the stop

        The maximum allowable frequency is 1 minute.

        The columns are hierarchical (multi-indexed) with

        - top level: name = 'indicator', values = ['num_trips']
        - middle level: name = 'stop_id', values = the active stop IDs
        - bottom level: name = 'direction_id', values = 0s and 1s

        If not ``split_directions``, then don't include the bottom level.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_0`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trip_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

    """
    if trip_subset.empty:
        return pd.DataFrame()

    f = pd.merge(stop_times_subset, trip_subset)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = np.nan
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError(
                "At least one trip direction ID value " "must be non-NaN."
            )

        # Alter stop IDs to encode trip direction:
        # <stop ID>-0 and <stop ID>-1
        f["stop_id"] = f["stop_id"] + "-" + f["direction_id"].map(str)
    stops = f["stop_id"].unique()

    # Bin each stop departure time
    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    def F(x):
        return (hp.timestr_to_seconds(x) // 60) % (24 * 60)

    f["departure_index"] = f["departure_time"].map(F)

    # Create one time series for each stop
    series_by_stop = {stop: [0 for i in range(num_bins)] for stop in stops}

    for stop, group in f.groupby("stop_id"):
        counts = Counter((bin, 0) for bin in bins) + Counter(
            group["departure_index"].values
        )
        series_by_stop[stop] = [counts[bin] for bin in bins]

    # Combine lists into dictionary of form indicator -> time series.
    # Only one indicator in this case, but could add more
    # in the future as was done with route time series.
    rng = pd.date_range(date_label, periods=24 * 60, freq="Min")
    series_by_indicator = {
        "num_trips": pd.DataFrame(series_by_stop, index=rng).fillna(0)
    }

    # Combine all time series into one time series
    g = hp.combine_time_series(
        series_by_indicator, kind="stop", split_directions=split_directions
    )
    return hp.downsample(g, freq=freq)


def get_stops(
    feed: "Feed",
    date: Optional[str] = None,
    trip_id: Optional[str] = None,
    route_id: Optional[str] = None,
    *,
    in_stations: bool = False,
) -> pd.DataFrame:
    """
    Return a section of ``feed.stops``.

    Parameters
    -----------
    feed : Feed
    date : string
        YYYYMMDD string; restricts the output to stops active
        (visited by trips) on the date
    trip_id : string
        ID of a trip in ``feed.trips``; restricts output to stops
        visited by the trip
    route_id : string
        ID of route in ``feed.routes``; restricts output to stops
        visited by the route
    in_stations : boolean
        If ``True``, then restricts output to stops in stations if
        station data is available in ``feed.stops``

    Returns
    -------
    DataFrame
        A subset of ``feed.stops`` defined by the parameters above

    Notes
    -----
    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
    - Those used in :func:`.stop_times.get_stop_times`

    """
    s = feed.stops.copy()
    if date is not None:
        A = feed.get_stop_times(date)["stop_id"]
        s = s[s["stop_id"].isin(A)].copy()
    if trip_id is not None:
        st = feed.stop_times.copy()
        B = st[st["trip_id"] == trip_id]["stop_id"]
        s = s[s["stop_id"].isin(B)].copy()
    elif route_id is not None:
        A = feed.trips[feed.trips["route_id"] == route_id]["trip_id"]
        st = feed.stop_times.copy()
        B = st[st["trip_id"].isin(A)]["stop_id"]
        s = s[s["stop_id"].isin(B)].copy()
    if in_stations and set(["location_type", "parent_station"]) <= set(
        s.columns
    ):
        s = s[(s["location_type"] != 1) & (s["parent_station"].notnull())]

    return s


def compute_stop_activity(feed: "Feed", dates: List[str]) -> pd.DataFrame:
    """
    Mark stops as active or inactive on the given dates.
    A stop is *active* on a given date if some trips that starts on the
    date visits the stop (possibly after midnight).

    Parameters
    ----------
    feed : Feed
    dates : string or list
        A YYYYMMDD date string or list thereof indicating the date(s)
        for which to compute activity

    Returns
    -------
    DataFrame
        Columns are

        - stop_id
        - ``dates[0]``: 1 if the stop has at least one trip visiting it
          on ``dates[0]``; 0 otherwise
        - ``dates[1]``: 1 if the stop has at least one trip visiting it
          on ``dates[1]``; 0 otherwise
        - etc.
        - ``dates[-1]``: 1 if the stop has at least one trip visiting it
          on ``dates[-1]``; 0 otherwise

    Notes
    -----
    - If all dates lie outside the Feed period, then return an empty
      DataFrame
    - Assume the following feed attributes are not ``None``:

        * ``feed.stop_times``
        * Those used in :func:`.trips.compute_trip_activity`

    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    trip_activity = feed.compute_trip_activity(dates)
    g = pd.merge(trip_activity, feed.stop_times).groupby("stop_id")
    # Pandas won't allow me to simply return g[dates].max().reset_index().
    # I get ``TypeError: unorderable types: datetime.date() < str()``.
    # So here's a workaround.
    for (i, date) in enumerate(dates):
        if i == 0:
            f = g[date].max().reset_index()
        else:
            f = f.merge(g[date].max().reset_index())
    return f


def compute_stop_stats(
    feed: "Feed",
    dates: List[str],
    stop_ids: Optional[List[str]] = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats for all stops for the given dates.
    Optionally, restrict to the stop IDs given.

    Parameters
    ----------
    feed : Feed
    dates : string or list
        A YYYYMMDD date string or list thereof indicating the date(s)
        for which to compute stats
    stop_ids : list
        Optional list of stop IDs to restrict stats to
    headway_start_time : string
        HH:MM:SS time string indicating the start time for computing
        headway stats
    headway_end_time : string
        HH:MM:SS time string indicating the end time for computing
        headway stats
    split_directions : boolean
        If ``True``, then separate the stop stats by direction (0 or 1)
        of the trips visiting the stops; otherwise aggregate trips
        visiting from both directions

    Returns
    -------
    DataFrame
        Columns are

        - ``'date'``
        - ``'stop_id'``
        - ``'direction_id'``: present if and only if ``split_directions``
        - ``'num_routes'``: number of routes visiting the stop
          (in the given direction) on the date
        - ``'num_trips'``: number of trips visiting stop
          (in the givin direction) on the date
        - ``'max_headway'``: maximum of the durations (in minutes)
          between trip departures at the stop between
          ``headway_start_time`` and ``headway_end_time`` on the date
        - ``'min_headway'``: minimum of the durations (in minutes) mentioned
          above
        - ``'mean_headway'``: mean of the durations (in minutes) mentioned
          above
        - ``'start_time'``: earliest departure time of a trip from this stop
          on the date
        - ``'end_time'``: latest departure time of a trip from this stop on
          the date

        Exclude dates with no active stops, which could yield the empty DataFrame.

    Notes
    -----
    - Assume the following feed attributes are not ``None``:
        * ``feed.stop_times``
        * Those used in :func:`.trips.get_trips`
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

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

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
            t = feed.trips
            trips = t[t["trip_id"].isin(ids)].copy()
            stats = (
                compute_stop_stats_0(
                    stop_times_subset,
                    trips,
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


def build_zero_stop_time_series(
    feed: "Feed",
    date_label: str = "20010101",
    freq: str = "5Min",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Return a stop time series with the same index and hierarchical columns
    as output by :func:`compute_stop_time_series_0`,
    but fill it full of zero values.
    """
    start = date_label
    end = pd.to_datetime(date_label + " 23:59:00")
    rng = pd.date_range(start, end, freq=freq)
    inds = ["num_trips"]
    sids = feed.stops.stop_id
    if split_directions:
        product = [inds, sids, [0, 1]]
        names = ["indicator", "stop_id", "direction_id"]
    else:
        product = [inds, sids]
        names = ["indicator", "stop_id"]
    cols = pd.MultiIndex.from_product(product, names=names)
    return pd.DataFrame(
        [[0 for c in cols]], index=rng, columns=cols
    ).sort_index(axis="columns")


def compute_stop_time_series(
    feed: "Feed",
    dates: List[str],
    stop_ids: Optional[List[str]] = None,
    freq: str = "5Min",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute time series for the stops on the given dates at the
    given frequency and return the result as a DataFrame of the same
    form as output by :func:`.stop_times.compute_stop_time_series_0`.
    Optionally restrict to stops in the given list of stop IDs.

    Parameters
    ----------
    feed : Feed
    dates : string or list
        A YYYYMMDD date string or list thereof indicating the date(s)
        for which to compute stats
    stop_ids : list
        Optional list of stop IDs to restrict to
    split_directions : boolean
        If ``True``, then separate the stop stats by direction (0 or 1)
        of the trips visiting the stops; otherwise aggregate trips
        visiting from both directions
    freq : Pandas frequency string
        Specifices the frequency with which to resample the time series;
        max frequency is one minute ('Min')

    Returns
    -------
    DataFrame
        A time series with a timestamp index across the given dates
        sampled at the given frequency.
        The maximum allowable frequency is 1 minute.

        The columns are the same as in
        :func:`compute_stop_time_series_0`.

        Exclude dates that lie outside of the Feed's date range.
        If all dates lie outside the Feed's date range, then return an
        empty DataFrame

    Notes
    -----
    - See the notes for :func:`compute_stop_time_series_0`
    - Assume the following feed attributes are not ``None``:

        * ``feed.stop_times``
        * Those used in :func:`.trips.get_trips`

    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    activity = feed.compute_trip_activity(dates)

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence ->
    # [stats DataFarme, date list that stats apply]
    stats_and_dates_by_ids = {}
    zero_stats = build_zero_stop_time_series(
        feed, split_directions=split_directions, freq=freq
    )
    for date in dates:
        ids = tuple(activity.loc[activity[date] > 0, "trip_id"])
        if ids in stats_and_dates_by_ids:
            # Append date to date list
            stats_and_dates_by_ids[ids][1].append(date)
        elif not ids:
            # Null stats
            stats_and_dates_by_ids[ids] = [zero_stats, [date]]
        else:
            # Compute stats
            t = feed.trips
            trips = t[t["trip_id"].isin(ids)].copy()
            stats = compute_stop_time_series_0(
                stop_times_subset,
                trips,
                split_directions=split_directions,
                freq=freq,
                date_label=date,
            )

            # Remember stats
            stats_and_dates_by_ids[ids] = [stats, [date]]

    # Assemble stats into DataFrame
    frames = []
    for stats, dates_ in stats_and_dates_by_ids.values():
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
        # Insert missing dates and NaNs to complete series index
        end_datetime = pd.to_datetime(dates[-1] + " 23:59:59")
        new_index = pd.date_range(f.index[0], end_datetime, freq=freq)
        f = f.reindex(new_index)
    else:
        # Set frequency
        f.index.freq = pd.tseries.frequencies.to_offset(freq)

    return f.rename_axis("datetime", axis="index")


def build_stop_timetable(
    feed: "Feed", stop_id: str, dates: List[str]
) -> pd.DataFrame:
    """
    Return a DataFrame containing the timetable for the given stop ID
    and dates.

    Parameters
    ----------
    feed : Feed
    stop_id : string
        ID of the stop for which to build the timetable
    dates : string or list
        A YYYYMMDD date string or list thereof

    Returns
    -------
    DataFrame
        The columns are all those in ``feed.trips`` plus those in
        ``feed.stop_times`` plus ``'date'``, and the stop IDs are
        restricted to the given stop ID.
        The result is sorted by date then departure time.

    Notes
    -----
    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`.stop_times.get_stop_times`

    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    t = pd.merge(feed.trips, feed.stop_times)
    t = t[t["stop_id"] == stop_id].copy()
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to stops active on date
        ids = a.loc[a[date] == 1, "trip_id"]
        f = t[t["trip_id"].isin(ids)].copy()
        f["date"] = date
        frames.append(f)

    f = pd.concat(frames)
    return f.sort_values(["date", "departure_time"])


def geometrize_stops_0(stops: pd.DataFrame, *, use_utm: bool = False) -> gpd.GeoDataFrame:
    """
    Given a stops DataFrame, convert it to a GeoPandas GeoDataFrame of Points
    and return the result, which will no longer have the columns ``'stop_lon'`` and
    ``'stop_lat'``.
    """
    g = (
        stops.assign(
            geometry=lambda x: [
                sg.Point(p) for p in x[["stop_lon", "stop_lat"]].values
            ]
        )
        .drop(["stop_lon", "stop_lat"], axis=1)
        .pipe(lambda x: gpd.GeoDataFrame(x, crs=cs.WGS84))
    )

    if use_utm:
        lat, lon = stops.loc[0, ["stop_lat", "stop_lon"]].values
        crs = hp.get_utm_crs(lat, lon)
        g = g.to_crs(crs)

    return g


def ungeometrize_stops_0(geo_stops: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    The inverse of :func:`geometrize_stops_0`.

    If ``geo_stops`` is in UTM coordinates (has a UTM CRS property),
    then convert those UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    f = geo_stops.copy().to_crs(cs.WGS84)
    f["stop_lon"], f["stop_lat"] = zip(
        *f["geometry"].map(lambda p: [p.x, p.y])
    )
    del f["geometry"]
    return f


def geometrize_stops(feed: "Feed", stop_ids:Optional[Iterable[str]]=None, *, use_utm:bool=False) -> gpd.GeoDataFrame:
    """
    Given a Feed instance, convert its stops DataFrame to a GeoDataFrame of
    Points and return the result, which will no longer have the columns
    ``'stop_lon'`` and ``'stop_lat'``.

    If an iterable of stop IDs is given, then subset to those stops.
    If ``use_utm``, then use local UTM coordinates for the geometries.
    """
    if stop_ids is not None:
        stops = feed.stops.loc[lambda x: x.stop_id.isin(stop_ids)]
    else:
        stops = feed.stops

    return geometrize_stops_0(stops, use_utm=use_utm)


def build_geometry_by_stop(feed: "Feed", stop_ids: Optional[Iterable[str]] = None, *, use_utm: bool = False) -> Dict:
    """
    Return a dictionary of the form <stop ID> -> <Shapely Point representing stop>. 
    """
    return dict(
        geometrize_stops(feed, stop_ids=stop_ids, use_utm=True)
        .filter(["stop_id", "geometry"])
        .values
    )
    

def stops_to_geojson(
    feed: "Feed", stop_ids: Optional[Iterable[str]] = None
) -> Dict:
    """
    Return a GeoJSON FeatureCollection of Point features
    representing ``feed.stops``.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of stop IDs is given, then subset to those stops.
    """
    return hp.drop_feature_ids(
        json.loads(geometrize_stops(feed, stop_ids=stop_ids).to_json())
    )


def get_stops_in_polygon(
    feed: "Feed", polygon: sg.Polygon, geo_stops:Optional[gpd.GeoDataFrame]=None, *, geometrized: bool = False
) -> pd.DataFrame:
    """
    Return the subset of ``feed.stops`` that contains all stops that lie
    within the given Shapely Polygon that is specified in
    WGS84 coordinates.

    If ``geometrized``, then return the stops as a GeoDataFrame.
    Specifying ``geo_stops`` will skip the first step of the
    algorithm, namely, geometrizing ``feed.stops``.
    """
    if geo_stops is not None:
        f = geo_stops.copy()
    else:
        f = geometrize_stops(feed)

    cols = f.columns
    f["hit"] = f["geometry"].within(polygon)
    f = f.loc[lambda x: x.hit].filter(cols)

    if geometrized:
        result = f
    else:
        result = ungeometrize_stops_0(f)

    return result


def map_stops(
    feed: "Feed", stop_ids: Iterable[str], stop_style: Dict = STOP_STYLE
):
    """
    Return a Folium map showing the given stops.

    Parameters
    ----------
    feed : Feed
    stop_ids : list
        IDs of trips in ``feed.stops``
    stop_style: dictionary
        Folium CircleMarker parameters to use for styling stops.

    Returns
    -------
    dictionary
        A Folium Map depicting the stops as CircleMarkers.

    Notes
    ------
    - Requires Folium

    """
    import folium as fl

    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create a feature group for the stops and add it to the map
    group = fl.FeatureGroup(name="Stops")

    # Add stops to feature group
    stops = feed.stops.loc[lambda x: x.stop_id.isin(stop_ids)].fillna("n/a")
    for prop in stops.to_dict(orient="records"):
        # Add stop
        lon = prop["stop_lon"]
        lat = prop["stop_lat"]
        fl.CircleMarker(
            location=[lat, lon],
            popup=fl.Popup(hp.make_html(prop)),
            **stop_style,
        ).add_to(group)

    group.add_to(my_map)

    # Add layer control
    fl.LayerControl().add_to(my_map)

    # Fit map to stop bounds
    bounds = [
        (stops.stop_lat.min(), stops.stop_lon.min()),
        (stops.stop_lat.max(), stops.stop_lon.max()),
    ]
    my_map.fit_bounds(bounds, padding=[1, 1])

    return my_map
