"""
Functions about stop times.
"""
from typing import Optional, List, TYPE_CHECKING

import pandas as pd
import numpy as np

from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_stop_times(feed: "Feed", date: Optional[str] = None) -> pd.DataFrame:
    """
    Return ``feed.stop_times``.
    If a date (YYYYMMDD date string) is given, then subset the result to only those
    stop times with trips active on the date.
    """
    if date is None:
        f = feed.stop_times.copy()
    else:
        trip_ids = feed.get_trips(date).trip_id
        f = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    return f

def append_dist_to_stop_times(feed: "Feed", trip_stats: pd.DataFrame) -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` field in
    ``feed.stop_times`` in terms of the distance units
    ``feed.dist_units``.
    Need trip stats in the form output by the function
    :func:`.trips.compute_trip_stats` for this.
    Return the resulting Feed.

    This does not always give accurate results.
    The algorithm works as follows.
    Compute the ``shape_dist_traveled`` field by using Shapely to
    measure the distance of a stop along its trip LineString.
    If for a given trip this process produces a non-monotonically
    increasing, hence incorrect, list of (cumulative) distances, then
    fall back to estimating the distances as follows.

    Get the average speed of the trip via ``trip_stats`` and use is to
    linearly interpolate distances for stop times, assuming that the
    first stop is at shape_dist_traveled = 0 (the start of the shape)
    and the last stop is at shape_dist_traveled = the length of the trip
    (taken from trip_stats and equal to the length of the shape, unless
    ``trip_stats`` was called with ``get_dist_from_shapes == False``).
    This fallback method usually kicks in on trips with
    self-intersecting LineStrings.
    Unfortunately, this fallback method will produce incorrect results
    when the first stop does not start at the start of its shape
    (so shape_dist_traveled != 0).
    This is the case for several trips in `this Portland feed
    <https://transitfeeds.com/p/trimet/43/1400947517>`_, for example.
    """
    feed = feed.copy()
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)
    geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)

    # Initialize DataFrame
    f = pd.merge(
        feed.stop_times,
        trip_stats[["trip_id", "shape_id", "distance", "duration"]],
    ).sort_values(["trip_id", "stop_sequence"])

    # Convert departure times to seconds past midnight to ease calculations
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)
    dist_by_stop_by_shape = {shape: {} for shape in geometry_by_shape}
    m_to_dist = hp.get_convert_dist("m", feed.dist_units)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        shape = group["shape_id"].iat[0]
        if not isinstance(shape, str):
            group["shape_dist_traveled"] = np.nan
            return group
        elif np.isnan(group["distance"].iat[0]):
            group["shape_dist_traveled"] = np.nan
            return group
        linestring = geometry_by_shape[shape]
        distances = []
        for stop in group["stop_id"].values:
            if stop in dist_by_stop_by_shape[shape]:
                d = dist_by_stop_by_shape[shape][stop]
            else:
                d = m_to_dist(
                    hp.get_segment_length(linestring, geometry_by_stop[stop])
                )
                dist_by_stop_by_shape[shape][stop] = d
            distances.append(d)
        s = sorted(distances)
        D = linestring.length
        distances_are_reasonable = all([d < D + 100 for d in distances])
        if distances_are_reasonable and s == distances:
            # Good
            pass
        elif distances_are_reasonable and s == distances[::-1]:
            # Reverse. This happens when the direction of a linestring
            # opposes the direction of the bus trip.
            distances = distances[::-1]
        else:
            # Totally redo using trip length, first and last stop times,
            # and linear interpolation
            dt = group["departure_time"]
            times = dt.values  # seconds
            t0, t1 = times[0], times[-1]
            d0, d1 = 0, group["distance"].iat[0]
            # Get indices of nan departure times and
            # temporarily forward fill them
            # for the purposes of using np.interp smoothly
            nan_indices = np.where(dt.isnull())[0]
            dt.fillna(method="ffill")
            # Interpolate
            distances = np.interp(times, [t0, t1], [d0, d1])
            # Nullify distances with nan departure times
            for i in nan_indices:
                distances[i] = np.nan

        group["shape_dist_traveled"] = distances
        return group

    g = f.groupby("trip_id", group_keys=False).apply(compute_dist)
    # Convert departure times back to time strings
    g["departure_time"] = g["departure_time"].map(
        lambda x: hp.timestr_to_seconds(x, inverse=True)
    )
    g = g.drop(["shape_id", "distance", "duration"], axis=1)
    feed.stop_times = g

    return feed


def get_start_and_end_times(
    feed: "Feed", date: Optional[str] = None
) -> List[str]:
    """
    Return the first departure time and last arrival time
    (HH:MM:SS time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date (YYYYMMDD string) if specified.
    """
    st = feed.get_stop_times(date)
    return (
        st["departure_time"].dropna().min(),
        st["arrival_time"].dropna().max(),
    )
