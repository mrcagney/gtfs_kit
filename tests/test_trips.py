import pandas as pd
import numpy as np
import pytest
import geopandas as gp
import folium as fl

from .context import (
    gtfs_kit,
    DATA_DIR,
    cairns,
    cairns_shapeless,
    cairns_dates,
    cairns_trip_stats,
)
from gtfs_kit import *


def test_is_active_trip():
    feed = cairns.copy()
    trip_id = "CNS2014-CNS_MUL-Weekday-00-4165878"
    date1 = "20140526"
    date2 = "20120322"
    assert is_active_trip(feed, trip_id, date1)
    assert not is_active_trip(feed, trip_id, date2)

    trip_id = "CNS2014-CNS_MUL-Sunday-00-4165971"
    date1 = "20140601"
    date2 = "20120602"
    assert is_active_trip(feed, trip_id, date1)
    assert not is_active_trip(feed, trip_id, date2)


def test_get_trips():
    feed = cairns.copy()
    date = cairns_dates[0]
    trips1 = get_trips(feed, date)
    # Should be a data frame
    assert isinstance(trips1, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips1.shape[0] <= feed.trips.shape[0]
    assert trips1.shape[1] == feed.trips.shape[1]
    # Should have correct columns
    assert set(trips1.columns) == set(feed.trips.columns)

    trips2 = get_trips(feed, date, "07:30:00")
    # Should be a data frame
    assert isinstance(trips2, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips2.shape[0] <= trips2.shape[0]
    assert trips2.shape[1] == trips1.shape[1]
    # Should have correct columns
    assert set(trips2.columns) == set(feed.trips.columns)


def test_compute_trip_activity():
    feed = cairns.copy()
    dates = get_first_week(feed)
    trips_activity = compute_trip_activity(feed, dates)
    # Should be a data frame
    assert isinstance(trips_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips_activity.shape[0] == feed.trips.shape[0]
    assert trips_activity.shape[1] == 1 + len(dates)
    # Date columns should contain only zeros and ones
    assert set(trips_activity[dates].values.flatten()) == {0, 1}


def test_compute_busiest_date():
    feed = cairns.copy()
    dates = get_first_week(feed)[:1]
    date = compute_busiest_date(feed, dates + ["999"])
    # Busiest day should lie in first week
    assert date in dates


def test_compute_trip_stats():
    feed = cairns.copy()
    n = 3
    rids = feed.routes.route_id.loc[:n]
    trip_stats = compute_trip_stats(feed, route_ids=rids)

    # Should be a data frame with the correct number of rows
    trip_subset = feed.trips.loc[lambda x: x["route_id"].isin(rids)].copy()
    assert isinstance(trip_stats, pd.core.frame.DataFrame)
    assert trip_stats.shape[0] == trip_subset.shape[0]

    # Should contain the correct columns
    expect_cols = set(
        [
            "trip_id",
            "direction_id",
            "route_id",
            "route_short_name",
            "route_type",
            "shape_id",
            "num_stops",
            "start_time",
            "end_time",
            "start_stop_id",
            "end_stop_id",
            "distance",
            "duration",
            "speed",
            "is_loop",
        ]
    )
    assert set(trip_stats.columns) == expect_cols

    # Distance units should be correct
    d1 = trip_stats.distance.iat[0]  # km
    trip_stats_2 = compute_trip_stats(feed.convert_dist("ft"), route_ids=rids)
    d2 = trip_stats_2.distance.iat[0]  # mi
    f = get_convert_dist("km", "mi")
    assert f(d1) == d2

    # Shapeless feeds should have null entries for distance column
    feed = cairns_shapeless.copy()
    trip_stats = compute_trip_stats(feed, route_ids=rids)
    assert len(trip_stats["distance"].unique()) == 1
    assert np.isnan(trip_stats["distance"].unique()[0])

    # Should contain the correct trips
    get_trips = set(trip_stats["trip_id"].values)
    expect_trips = set(trip_subset["trip_id"].values)
    assert get_trips == expect_trips

    # Missing the optional ``direction_id`` column in ``feed.trips``
    # should give ``direction_id`` column in stats with all NaNs
    feed = cairns.copy()
    del feed.trips["direction_id"]
    trip_stats = compute_trip_stats(feed, route_ids=rids)
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats.direction_id.isnull().all()

    # Missing the optional ``shape_id`` column in ``feed.trips``
    # should give ``shape_id`` column in stats with all NaNs
    feed = cairns.copy()
    del feed.trips["shape_id"]
    trip_stats = compute_trip_stats(feed, route_ids=rids)
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats.shape_id.isnull().all()


@pytest.mark.slow
def test_locate_trips():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats
    feed = append_dist_to_stop_times(feed)
    date = cairns_dates[0]
    times = ["08:00:00"]
    f = locate_trips(feed, date, times)
    g = get_trips(feed, date, times[0])
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct number of rows
    assert f.shape[0] == g.shape[0]
    # Should have the correct columns
    expect_cols = set(
        [
            "route_id",
            "trip_id",
            "direction_id",
            "shape_id",
            "time",
            "rel_dist",
            "lon",
            "lat",
        ]
    )
    assert set(f.columns) == expect_cols

    # Missing feed.trips.shape_id should raise an error
    feed = cairns_shapeless.copy()
    del feed.trips["shape_id"]
    with pytest.raises(ValueError):
        locate_trips(feed, date, times)


def test_geometrize_trips():
    feed = cairns.copy()
    trip_ids = feed.trips.trip_id.loc[:1]
    g = geometrize_trips(feed, trip_ids, use_utm=True)
    assert isinstance(g, gp.GeoDataFrame)
    assert g.shape[0] == len(trip_ids)
    assert g.crs != WGS84

    with pytest.raises(ValueError):
        geometrize_trips(cairns_shapeless)


def test_trips_to_geojson():
    feed = cairns.copy()
    trip_ids = feed.trips.trip_id.loc[:1]
    n = len(trip_ids)
    gj = trips_to_geojson(feed, trip_ids)
    assert len(gj["features"]) == n

    gj = trips_to_geojson(feed, trip_ids, include_stops=True)
    k = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids), "stop_id"].nunique()
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        trips_to_geojson(cairns_shapeless)

    with pytest.raises(ValueError):
        trips_to_geojson(cairns, trip_ids=["bingo"])


def test_map_trips():
    feed = cairns.copy()
    tids = feed.trips["trip_id"].values[:2]
    m = map_trips(feed, tids, include_stops=True)
    assert isinstance(m, fl.Map)
