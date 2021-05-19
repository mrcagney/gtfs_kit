"""
Functions about trips.
"""
from __future__ import annotations
import json
from typing import Optional, Iterable, TYPE_CHECKING

import geopandas as gp
import pandas as pd
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import folium as fl
import folium.plugins as fp

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def is_active_trip(feed: "Feed", trip_id: str, date: str) -> bool:
    """
    Return ``True`` if the ``feed.calendar`` or ``feed.calendar_dates``
    says that the trip runs on the given date (YYYYMMDD date string);
    return ``False`` otherwise.

    Note that a trip that starts on date d, ends after 23:59:59, and
    does not start again on date d+1 is considered active on date d and
    not active on date d+1.
    This subtle point, which is a side effect of the GTFS, can
    lead to confusion.

    This function is key for getting all trips, routes, etc. that are
    active on a given date, so the function needs to be fast.
    """
    service = feed._trips_i.at[trip_id, "service_id"]
    # Check feed._calendar_dates_g.
    caldg = feed._calendar_dates_g
    if caldg is not None:
        if (service, date) in caldg.groups:
            et = caldg.get_group((service, date))["exception_type"].iat[0]
            if et == 1:
                return True
            else:
                # Exception type is 2
                return False
    # Check feed._calendar_i
    cali = feed._calendar_i
    if cali is not None:
        if service in cali.index:
            weekday_str = hp.weekday_to_str(hp.datestr_to_date(date).weekday())
            if (
                cali.at[service, "start_date"] <= date <= cali.at[service, "end_date"]
                and cali.at[service, weekday_str] == 1
            ):
                return True
            else:
                return False
    # If you made it here, then something went wrong
    return False


def get_trips(
    feed: "Feed", date: Optional[str] = None, time: Optional[str] = None
) -> pd.DataFrame:
    """
    Return ``feed.trips``.
    If date (YYYYMMDD date string) is given then subset the result to trips
    that start on that date.
    If a time (HH:MM:SS string, possibly with HH > 23) is given in addition to a date,
    then further subset the result to trips in service at that time.
    """
    if feed.trips is None or date is None:
        return feed.trips

    f = feed.trips.copy()
    f["is_active"] = f["trip_id"].map(
        lambda trip_id: feed.is_active_trip(trip_id, date)
    )
    f = f[f["is_active"]].copy()
    del f["is_active"]

    if time is not None:
        # Get trips active during given time
        g = pd.merge(f, feed.stop_times[["trip_id", "departure_time"]])

        def F(group):
            d = {}
            start = group["departure_time"].dropna().min()
            end = group["departure_time"].dropna().max()
            try:
                result = start <= time <= end
            except TypeError:
                result = False
            d["is_active"] = result
            return pd.Series(d)

        h = g.groupby("trip_id").apply(F).reset_index()
        f = pd.merge(f, h[h["is_active"]])
        del f["is_active"]

    return f


def compute_trip_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark trip as active or inactive on the given dates (YYYYMMDD date strings)
    as computed by the function :func:`is_active_trip`.

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``dates[0]``: 1 if the trip is active on ``dates[0]``;
      0 otherwise
    - ``dates[1]``: 1 if the trip is active on ``dates[1]``;
      0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``;
      0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an
    empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    f = feed.trips.copy()
    for date in dates:
        f[date] = f["trip_id"].map(
            lambda trip_id: int(feed.is_active_trip(trip_id, date))
        )
    return f[["trip_id"] + list(dates)]


def compute_busiest_date(feed: "Feed", dates: list[str]) -> str:
    """
    Given a list of dates (YYYYMMDD date strings), return the first date that has the
    maximum number of active trips.
    """
    f = feed.compute_trip_activity(dates)
    s = [(f[c].sum(), c) for c in f.columns if c != "trip_id"]
    return max(s)[1]


def compute_trip_stats(
    feed: "Feed",
    route_ids: Optional[list[str]] = None,
    *,
    compute_dist_from_shapes: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame with the following columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: NaN if missing from feed
    - ``'shape_id'``: NaN if missing from feed
    - ``'num_stops'``: number of stops on trip
    - ``'start_time'``: first departure time of the trip
    - ``'end_time'``: last departure time of the trip
    - ``'start_stop_id'``: stop ID of the first stop of the trip
    - ``'end_stop_id'``: stop ID of the last stop of the trip
    - ``'is_loop'``: 1 if the start and end stop are less than 400m apart and
      0 otherwise
    - ``'distance'``: distance of the trip;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'duration'``: duration of the trip in hours
    - ``'speed'``: distance/duration

    If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at
    least one non-NaN value and ``compute_dist_from_shapes == False``,
    then use that column to compute the distance column.
    Else if ``feed.shapes is not None``, then compute the distance
    column using the shapes and Shapely.
    Otherwise, set the distances to NaN.

    If route IDs are given, then restrict to trips on those routes.

    Notes
    -----
    - Assume the following feed attributes are not ``None``:

        * ``feed.trips``
        * ``feed.routes``
        * ``feed.stop_times``
        * ``feed.shapes`` (optionally)

    - Calculating trip distances with ``compute_dist_from_shapes=True``
      seems pretty accurate.  For example, calculating trip distances on
      `this Portland feed
      <https://transitfeeds.com/p/trimet/43/1400947517>`_
      using ``compute_dist_from_shapes=False`` and
      ``compute_dist_from_shapes=True``,
      yields a difference of at most 0.83km from the original values.

    """
    f = feed.trips.copy()

    # Restrict to given route IDs
    if route_ids is not None:
        f = f[f["route_id"].isin(route_ids)].copy()

    # Merge with stop times and extra trip info.
    # Convert departure times to seconds past midnight to
    # compute trip durations later.
    if "direction_id" not in f.columns:
        f["direction_id"] = np.nan
    if "shape_id" not in f.columns:
        f["shape_id"] = np.nan

    f = (
        f[["route_id", "trip_id", "direction_id", "shape_id"]]
        .merge(feed.routes[["route_id", "route_short_name", "route_type"]])
        .merge(feed.stop_times)
        .sort_values(["trip_id", "stop_sequence"])
        .assign(departure_time=lambda x: x["departure_time"].map(hp.timestr_to_seconds))
    )

    # Compute all trips stats except distance,
    # which is possibly more involved
    geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)
    g = f.groupby("trip_id")

    def my_agg(group):
        d = dict()
        d["route_id"] = group["route_id"].iat[0]
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["direction_id"] = group["direction_id"].iat[0]
        d["shape_id"] = group["shape_id"].iat[0]
        d["num_stops"] = group.shape[0]
        d["start_time"] = group["departure_time"].iat[0]
        d["end_time"] = group["departure_time"].iat[-1]
        d["start_stop_id"] = group["stop_id"].iat[0]
        d["end_stop_id"] = group["stop_id"].iat[-1]
        dist = geometry_by_stop[d["start_stop_id"]].distance(
            geometry_by_stop[d["end_stop_id"]]
        )
        d["is_loop"] = int(dist < 400)
        d["duration"] = (d["end_time"] - d["start_time"]) / 3600
        return pd.Series(d)

    # Apply my_agg, but don't reset index yet.
    # Need trip ID as index to line up the results of the
    # forthcoming distance calculation
    h = g.apply(my_agg)

    # Compute distance
    if hp.is_not_null(f, "shape_dist_traveled") and not compute_dist_from_shapes:
        # Compute distances using shape_dist_traveled column, converting to km or mi
        if hp.is_metric(feed.dist_units):
            convert_dist = hp.get_convert_dist(feed.dist_units, "km")
        else:
            convert_dist = hp.get_convert_dist(feed.dist_units, "mi")
        h["distance"] = g.apply(
            lambda group: convert_dist(group.shape_dist_traveled.max())
        )
    elif feed.shapes is not None:
        # Compute distances using the shapes and Shapely
        geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)
        # Convert to km or mi
        if hp.is_metric(feed.dist_units):
            m_to_dist = hp.get_convert_dist("m", "km")
        else:
            m_to_dist = hp.get_convert_dist("m", "mi")

        def compute_dist(group):
            """
            Return the distance traveled along the trip between the
            first and last stops.
            If that distance is negative or if the trip's linestring
            intersects itfeed, then return the length of the trip's
            linestring instead.
            """
            shape = group["shape_id"].iat[0]
            try:
                # Get the linestring for this trip
                linestring = geometry_by_shape[shape]
            except KeyError:
                # Shape ID is NaN or doesn't exist in shapes.
                # No can do.
                return np.nan

            # If the linestring intersects itfeed, then that can cause
            # errors in the computation below, so just
            # return the length of the linestring as a good approximation
            D = linestring.length
            if not linestring.is_simple:
                return D

            # Otherwise, return the difference of the distances along
            # the linestring of the first and last stop
            start_stop = group["stop_id"].iat[0]
            end_stop = group["stop_id"].iat[-1]
            try:
                start_point = geometry_by_stop[start_stop]
                end_point = geometry_by_stop[end_stop]
            except KeyError:
                # One of the two stop IDs is NaN, so just
                # return the length of the linestring
                return D
            d1 = linestring.project(start_point)
            d2 = linestring.project(end_point)
            d = d2 - d1
            if 0 < d < D + 100:
                return d
            else:
                # Something is probably wrong, so just
                # return the length of the linestring
                return D

        h["distance"] = g.apply(compute_dist)
        # Convert from meters
        h["distance"] = h["distance"].map(m_to_dist)
    else:
        h["distance"] = np.nan

    # Reset index and compute final stats
    h = h.reset_index()
    h["speed"] = h["distance"] / h["duration"]
    h[["start_time", "end_time"]] = h[["start_time", "end_time"]].applymap(
        lambda x: hp.timestr_to_seconds(x, inverse=True)
    )

    return h.sort_values(["route_id", "direction_id", "start_time"])


def locate_trips(feed: "Feed", date: str, times: list[str]) -> pd.DataFrame:
    """
    Return the positions of all trips active on the
    given date (YYYYMMDD date string) and times (HH:MM:SS time strings,
    possibly with HH > 23).

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``'route_id'``
    - ``'direction_id'``: all NaNs if ``feed.trips.direction_id`` is
      missing
    - ``'time'``
    - ``'rel_dist'``: number between 0 (start) and 1 (end)
      indicating the relative distance of the trip along its path
    - ``'lon'``: longitude of trip at given time
    - ``'lat'``: latitude of trip at given time

    Assume ``feed.stop_times`` has an accurate
    ``shape_dist_traveled`` column.
    """
    if not hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        raise ValueError(
            "feed.stop_times needs to have a non-null shape_dist_traveled "
            "column. You can create it, possibly with some inaccuracies, "
            "via feed2 = feed.append_dist_to_stop_times()."
        )

    if "shape_id" not in feed.trips.columns:
        raise ValueError("feed.trips.shape_id must exist.")

    # Start with stop times active on date
    f = feed.get_stop_times(date)
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    # Compute relative distance of each trip along its path
    # at the given time times.
    # Use linear interpolation based on stop departure times and
    # shape distance traveled.
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    sample_times = np.array([hp.timestr_to_seconds(s) for s in times])

    def compute_rel_dist(group):
        dists = sorted(group["shape_dist_traveled"].values)
        times = sorted(group["departure_time"].values)
        ts = sample_times[(sample_times >= times[0]) & (sample_times <= times[-1])]
        ds = np.interp(ts, times, dists)
        return pd.DataFrame({"time": ts, "rel_dist": ds / dists[-1]})

    # return f.groupby('trip_id', group_keys=False).\
    #   apply(compute_rel_dist).reset_index()
    g = f.groupby("trip_id").apply(compute_rel_dist).reset_index()

    # Delete extraneous multi-index column
    del g["level_1"]

    # Convert times back to time strings
    g["time"] = g["time"].map(lambda x: hp.timestr_to_seconds(x, inverse=True))

    # Merge in more trip info and
    # compute longitude and latitude of trip from relative distance
    t = feed.trips.copy()
    if "direction_id" not in t.columns:
        t["direction_id"] = np.nan

    h = pd.merge(g, t[["trip_id", "route_id", "direction_id", "shape_id"]])
    if not h.shape[0]:
        # Return a DataFrame with the promised headers but no data.
        # Without this check, result below could be an empty DataFrame.
        h["lon"] = pd.Series()
        h["lat"] = pd.Series()
        return h

    def get_lonlat(group):
        shape = group["shape_id"].iat[0]
        linestring = geometry_by_shape[shape]
        lonlats = [
            linestring.interpolate(d, normalized=True).coords[0]
            for d in group["rel_dist"].values
        ]
        group["lon"], group["lat"] = zip(*lonlats)
        return group

    return h.groupby("shape_id").apply(get_lonlat)


def geometrize_trips(
    feed: "Feed", trip_ids: Optional[Iterable[str]] = None, *, use_utm=False
) -> gp.GeoDataFrame:
    """
    Return a GeoDataFrame with the columns in ``feed.trips`` and a geometry column
    of LineStrings, each of which represents the shape of the corresponding trip.

    If an iterable of trip IDs is given, then subset to those trips.
    If ``use_utm``, then use local UTM coordinates for the geometries.

    Raise a ValueError if the Feed has no shapes.
    """
    if feed.shapes is None:
        raise ValueError("This Feed has no shapes.")

    if trip_ids is not None:
        trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)].copy()
    else:
        trips = feed.trips.copy()

    return (
        feed.geometrize_shapes(shape_ids=trips.shape_id.tolist(), use_utm=use_utm)
        .filter(["shape_id", "geometry"])
        .merge(trips, how="left")
    )


# def trips_to_geojson(
#     feed: "Feed",
#     trip_ids: Optional[Iterable[str]] = None,
#     *,
#     include_stops: bool = False,
# ) -> dict:
#     """
#     Return a GeoJSON FeatureCollection of LineString features representing the Feed's trips.
#     The coordinates reference system is the default one for GeoJSON,
#     namely WGS84.

#     If ``include_stops``, then include the trip stops as Point features .
#     If an iterable of trip IDs is given, then subset to those trips.
#     If the subset is empty, then return a FeatureCollection with an empty list of
#     features.
#     If the Feed has no shapes, then raise a ValueError.
#     If any of the given trip IDs are not found in the feed, then raise a ValueError.
#     """
#     if trip_ids is not None:
#         D = set(trip_ids) - set(feed.trips.trip_id)
#         if D:
#             raise ValueError(f"Trip IDs {D} not found in feed.")

#     # Get trips
#     g = geometrize_trips(feed, trip_ids=trip_ids)
#     if g.empty:
#         collection = {"type": "FeatureCollection", "features": []}
#     else:
#         collection = json.loads(g.to_json())

#     # Get stops if desired
#     if include_stops:
#         if trip_ids is not None:
#             stop_ids = feed.stop_times.loc[
#                 lambda x: x.trip_id.isin(trip_ids), "stop_id"
#             ].unique()
#         else:
#             stop_ids = None

#         stops_gj = feed.stops_to_geojson(stop_ids=stop_ids)
#         collection["features"].extend(stops_gj["features"])

#     return hp.drop_feature_ids(collection)


def trips_to_geojson(
    feed: "Feed",
    trip_ids: Optional[Iterable[str]] = None,
    *,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection of LineString features representing
    all the Feed's trips.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If ``include_stops``, then include the trip stops as Point features.
    If an iterable of trip IDs is given, then subset to those trips.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If the Feed has no shapes, then raise a ValueError.
    """
    if trip_ids is None or not list(trip_ids):
        trip_ids = feed.trips.trip_id

    D = set(trip_ids) - set(feed.trips.trip_id)
    if D:
        raise ValueError(f"Trip IDs {D} not found in feed.")

    # Get trips
    g = geometrize_trips(feed, trip_ids=trip_ids)
    trips_gj = json.loads(g.to_json())

    # Get stops if desired
    if include_stops:
        st_gj = feed.stop_times_to_geojson(trip_ids)
        trips_gj["features"].extend(st_gj["features"])

    return hp.drop_feature_ids(trips_gj)


def map_trips(
    feed: "Feed",
    trip_ids: Iterable[str],
    color_palette: list[str] = cs.COLORS_SET2,
    *,
    include_stops: bool = False,
    include_arrows: bool = False,
):
    """
    Return a Folium map showing the given trips and (optionally)
    their stops.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If ``include_arrows``, then use the Folium plugin PolyLineTextPath to draw arrows
    on each trip polyline indicating its direction of travel; this fails to work in some
    browsers, such as Brave 0.68.132.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create colors
    n = len(trip_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, trip_id in enumerate(trip_ids):
        collection = trips_to_geojson(feed, [trip_id], include_stops=include_stops)

        group = fl.FeatureGroup(name=f"Trip {trip_id}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop if present
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

            # Add trip
            else:
                path = fl.PolyLine(
                    [[x[1], x[0]] for x in f["geometry"]["coordinates"]],
                    color=color,
                    popup=hp.make_html(prop),
                )

                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

                if include_arrows:
                    # Direction arrows, assuming, as GTFS does, that
                    # trip direction equals LineString direction
                    fp.PolyLineTextPath(
                        path,
                        "        \u27A4        ",
                        repeat=True,
                        offset=5.5,
                        attributes={"fill": color, "font-size": "18"},
                    ).add_to(group)

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    # Folium wants a different ordering
    bounds = [(bounds[1], bounds[0]), (bounds[3], bounds[2])]
    my_map.fit_bounds(bounds)

    return my_map
