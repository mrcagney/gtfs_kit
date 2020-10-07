import itertools

import pytest
import pandas as pd
import geopandas as gp
import folium as fl

from .context import gtfs_kit, cairns, cairns_shapeless, cairns_dates, cairns_trip_stats
from gtfs_kit import *


@pytest.mark.slow
def test_compute_route_stats_0():
    feed = cairns.copy()
    ts1 = cairns_trip_stats.copy()
    ts2 = cairns_trip_stats.copy()
    ts2.direction_id = np.nan

    for ts, split_directions in itertools.product([ts1, ts2], [True, False]):
        if split_directions and ts.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                compute_route_stats_0(ts, split_directions=split_directions)
            continue

        rs = compute_route_stats_0(ts, split_directions=split_directions)

        # Should be a data frame of the correct shape
        assert isinstance(rs, pd.core.frame.DataFrame)
        if split_directions:
            max_num_routes = 2 * feed.routes.shape[0]
        else:
            max_num_routes = feed.routes.shape[0]
        assert rs.shape[0] <= max_num_routes

        # Should contain the correct columns
        expect_cols = set(
            [
                "route_id",
                "route_short_name",
                "route_type",
                "num_trips",
                "num_trip_ends",
                "num_trip_starts",
                "is_bidirectional",
                "is_loop",
                "start_time",
                "end_time",
                "max_headway",
                "min_headway",
                "mean_headway",
                "peak_num_trips",
                "peak_start_time",
                "peak_end_time",
                "service_duration",
                "service_distance",
                "service_speed",
                "mean_trip_distance",
                "mean_trip_duration",
            ]
        )
        if split_directions:
            expect_cols.add("direction_id")
        assert set(rs.columns) == expect_cols

    # Empty check
    rs = compute_route_stats_0(pd.DataFrame(), split_directions=split_directions)
    assert rs.empty


@pytest.mark.slow
def test_compute_route_time_series_0():
    feed = cairns.copy()
    ts1 = cairns_trip_stats.copy()
    ts2 = cairns_trip_stats.copy()
    ts2.direction_id = np.nan
    for ts, split_directions in itertools.product([ts1, ts2], [True, False]):
        if split_directions and ts.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                compute_route_stats_0(ts, split_directions=split_directions)
            continue

        rs = compute_route_stats_0(ts, split_directions=split_directions)
        rts = compute_route_time_series_0(
            ts, split_directions=split_directions, freq="H"
        )

        # Should be a data frame of the correct shape
        assert isinstance(rts, pd.core.frame.DataFrame)
        assert rts.shape[0] == 24
        assert rts.shape[1] == 6 * rs.shape[0]

        # Should have correct column names
        if split_directions:
            expect = ["indicator", "route_id", "direction_id"]
        else:
            expect = ["indicator", "route_id"]
        assert rts.columns.names == expect

        # Each route have a correct service distance total
        if not split_directions:
            g = ts.groupby("route_id")
            for route in ts["route_id"].values:
                get = rts["service_distance"][route].sum()
                expect = g.get_group(route)["distance"].sum()
                assert abs((get - expect) / expect) < 0.001

    # Empty check
    rts = compute_route_time_series_0(
        pd.DataFrame(), split_directions=split_directions, freq="1H"
    )
    assert rts.empty


def test_get_routes():
    feed = cairns.copy()
    date = cairns_dates[0]
    f = get_routes(feed, date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert f.shape[0] <= feed.routes.shape[0]
    assert f.shape[1] == feed.routes.shape[1]
    # Should have correct columns
    assert set(f.columns) == set(feed.routes.columns)

    g = get_routes(feed, date, "07:30:00")
    # Should be a data frame
    assert isinstance(g, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert g.shape[0] <= f.shape[0]
    assert g.shape[1] == f.shape[1]
    # Should have correct columns
    assert set(g.columns) == set(feed.routes.columns)


def test_compute_route_stats():
    feed = cairns.copy()
    dates = cairns_dates + ["20010101"]
    n = 3
    rids = feed.routes.route_id.loc[:n]
    trip_stats_subset = cairns_trip_stats.loc[lambda x: x["route_id"].isin(rids)]

    for split_directions in [True, False]:
        rs = compute_route_stats(
            feed, trip_stats_subset, dates, split_directions=split_directions
        )

        # Should be a data frame of the correct shape
        assert isinstance(rs, pd.core.frame.DataFrame)
        if split_directions:
            max_num_routes = 2 * n
        else:
            max_num_routes = n

        assert rs.shape[0] <= 2 * max_num_routes

        # Should contain the correct columns
        expect_cols = {
            "date",
            "route_id",
            "route_short_name",
            "route_type",
            "num_trips",
            "num_trip_ends",
            "num_trip_starts",
            "is_bidirectional",
            "is_loop",
            "start_time",
            "end_time",
            "max_headway",
            "min_headway",
            "mean_headway",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "mean_trip_distance",
            "mean_trip_duration",
        }
        if split_directions:
            expect_cols.add("direction_id")

        assert set(rs.columns) == expect_cols

        # Should only contains valid dates
        rs.date.unique().tolist() == cairns_dates

        # Empty dates should yield empty DataFrame
        rs = compute_route_stats(
            feed, trip_stats_subset, [], split_directions=split_directions
        )
        assert rs.empty


def test_build_zero_route_time_series():
    feed = cairns.copy()
    for split_directions in [True, False]:
        if split_directions:
            expect_names = ["indicator", "route_id", "direction_id"]
            expect_shape = (2, 6 * feed.routes.shape[0] * 2)
        else:
            expect_names = ["indicator", "route_id"]
            expect_shape = (2, 6 * feed.routes.shape[0])

        f = build_zero_route_time_series(
            feed, split_directions=split_directions, freq="12H"
        )

        assert isinstance(f, pd.core.frame.DataFrame)
        assert f.shape == expect_shape
        assert f.columns.names == expect_names
        assert not f.values.any()


def test_compute_route_time_series():
    feed = cairns.copy()
    dates = cairns_dates + ["20010101"]  # Spans 3 valid dates
    n = 3
    rids = feed.routes.route_id.loc[:n]
    trip_stats_subset = cairns_trip_stats.loc[lambda x: x["route_id"].isin(rids)]

    for split_directions in [True, False]:
        rs = compute_route_stats(
            feed, trip_stats_subset, dates, split_directions=split_directions
        )
        rts = compute_route_time_series(
            feed,
            trip_stats_subset,
            dates,
            split_directions=split_directions,
            freq="12H",
        )

        # Should be a data frame of the correct shape
        assert isinstance(rts, pd.core.frame.DataFrame)
        assert rts.shape[0] == 3 * 2  # 3-date span at 12H freq
        print(rts.columns)
        assert rts.shape[1] == 6 * rs.shape[0] / 2

        # Should have correct column names
        if split_directions:
            expect_names = ["indicator", "route_id", "direction_id"]
        else:
            expect_names = ["indicator", "route_id"]
        assert rts.columns.names, expect_names

        # Should have correct index name
        assert rts.index.name == "datetime"

        # Each route have a correct num_trip_starts
        if not split_directions:
            rsg = rs.groupby("route_id")
            for route in rs.route_id.values:
                get = rts["num_trip_starts"][route].sum()
                expect = rsg.get_group(route)["num_trips"].sum()
                assert get == expect

        # Empty dates should yield empty DataFrame
        rts = compute_route_time_series(
            feed, trip_stats_subset, [], split_directions=split_directions
        )
        assert rts.empty


def test_build_route_timetable():
    feed = cairns.copy()
    route_id = feed.routes["route_id"].values[0]
    dates = cairns_dates + ["20010101"]
    f = build_route_timetable(feed, route_id, dates)

    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)

    # Should have the correct columns
    expect_cols = set(feed.trips.columns) | set(feed.stop_times.columns) | set(["date"])
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == cairns_dates

    # Empty check
    f = build_route_timetable(feed, route_id, dates[2:])
    assert f.empty


def test_geometrize_routes():
    feed = cairns.copy()
    route_ids = feed.routes.route_id.loc[:1]
    g = geometrize_routes(feed, route_ids, use_utm=True)
    assert isinstance(g, gp.GeoDataFrame)
    assert g.shape[0] == len(route_ids)
    assert g.crs != WGS84

    g = geometrize_routes(feed, route_ids, split_directions=True)
    assert isinstance(g, gp.GeoDataFrame)
    assert g.shape[0] == 2 * len(route_ids)

    with pytest.raises(ValueError):
        geometrize_routes(cairns_shapeless)


def test_routes_to_geojson():
    feed = cairns.copy()
    route_ids = feed.routes.route_id.loc[:1]
    n = len(route_ids)

    gj = routes_to_geojson(feed, route_ids)
    assert len(gj["features"]) == n

    gj = routes_to_geojson(feed, route_ids, include_stops=True)
    k = (
        feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
        .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
        .nunique()
    )
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        routes_to_geojson(cairns_shapeless)

    with pytest.raises(ValueError):
        routes_to_geojson(cairns, route_ids=["bingo"])


def test_map_routes():
    feed = cairns.copy()
    rids = feed.routes.route_id.loc[:1]
    m = map_routes(feed, rids, include_stops=True)
    assert isinstance(m, fl.Map)
