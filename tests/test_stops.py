import pytest
import itertools

import pandas as pd
from pandas.util.testing import assert_frame_equal
import shapely.geometry as sg
import folium as fl

from .context import (
    gtfs_kit,
    DATA_DIR,
    cairns,
    cairns_dates,
)
from gtfs_kit import *
from gtfs_kit import _geometrize_stops, _ungeometrize_stops

def test_compute_stop_stats_base():
    feed1 = cairns.copy()
    feed2 = cairns.copy()
    feed2.trips.direction_id = np.nan

    for feed, split_directions in itertools.product(
        [feed1, feed2], [True, False]
    ):
        if split_directions and feed.trips.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                compute_stop_stats_base(
                    feed.stop_times,
                    feed.trips,
                    split_directions=split_directions,
                )
            continue

        stops_stats = compute_stop_stats_base(
            feed.stop_times, feed.trips, split_directions=split_directions
        )
        # Should be a data frame
        assert isinstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct columns
        expect_cols = set(
            [
                "stop_id",
                "num_routes",
                "num_trips",
                "max_headway",
                "min_headway",
                "mean_headway",
                "start_time",
                "end_time",
            ]
        )
        if split_directions:
            expect_cols.add("direction_id")
        assert set(stops_stats.columns) == expect_cols
        # Should contain the correct stops
        expect_stops = set(feed.stops["stop_id"].values)
        get_stops = set(stops_stats["stop_id"].values)
        assert get_stops == expect_stops

    # Empty check
    stats = compute_stop_stats_base(feed.stop_times, pd.DataFrame())
    assert stats.empty


@pytest.mark.slow
def test_compute_stop_time_series_base():
    feed1 = cairns.copy()
    feed2 = cairns.copy()
    feed2.trips.direction_id = np.nan

    for feed, split_directions in itertools.product(
        [feed1, feed2], [True, False]
    ):
        if split_directions and feed.trips.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                compute_stop_time_series_base(
                    feed.stop_times,
                    feed.trips,
                    split_directions=split_directions,
                )
            continue

        ss = compute_stop_stats_base(
            feed.stop_times, feed.trips, split_directions=split_directions
        )
        sts = compute_stop_time_series_base(
            feed.stop_times,
            feed.trips,
            freq="1H",
            split_directions=split_directions,
        )

        # Should be a data frame
        assert isinstance(sts, pd.core.frame.DataFrame)

        # Should have the correct shape
        assert sts.shape[0] == 24
        assert sts.shape[1] == ss.shape[0]

        # Should have correct column names
        if split_directions:
            expect = ["indicator", "stop_id", "direction_id"]
        else:
            expect = ["indicator", "stop_id"]
        assert sts.columns.names == expect

        # Each stop should have a correct total trip count
        if not split_directions:
            stg = feed.stop_times.groupby("stop_id")
            for stop in set(feed.stop_times["stop_id"].values):
                get = sts["num_trips"][stop].sum()
                expect = stg.get_group(stop)["departure_time"].count()
                assert get == expect

    # Empty check
    stops_ts = compute_stop_time_series_base(
        feed.stop_times,
        pd.DataFrame(),
        freq="1H",
        split_directions=split_directions,
    )
    assert stops_ts.empty


def test_get_stops():
    feed = cairns.copy()
    date = cairns_dates[0]
    trip_id = feed.trips["trip_id"].iat[0]
    route_id = feed.routes["route_id"].iat[0]
    frames = [
        get_stops(feed),
        get_stops(feed, date=date),
        get_stops(feed, trip_id=trip_id),
        get_stops(feed, route_id=route_id),
        get_stops(feed, date=date, trip_id=trip_id),
        get_stops(feed, date=date, route_id=route_id),
        get_stops(feed, date=date, trip_id=trip_id, route_id=route_id),
    ]
    for f in frames:
        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        assert f.shape[0] <= feed.stops.shape[0]
        assert f.shape[1] == feed.stops.shape[1]
        # Should have correct columns
        set(f.columns) == set(feed.stops.columns)
    # Number of rows should be reasonable
    assert frames[0].shape[0] <= frames[1].shape[0]
    assert frames[2].shape[0] <= frames[4].shape[0]
    assert frames[4].shape == frames[6].shape


def test_compute_stop_activity():
    feed = cairns.copy()
    dates = get_first_week(feed)
    stop_activity = compute_stop_activity(feed, dates)
    # Should be a data frame
    assert isinstance(stop_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert stop_activity.shape[0] == feed.stops.shape[0]
    assert stop_activity.shape[1] == len(dates) + 1
    # Date columns should contain only zeros and ones
    assert set(stop_activity[dates].values.flatten()) == {0, 1}


def test_compute_stop_stats():
    dates = cairns_dates + ["20010101"]
    feed = cairns.copy()
    n = 3
    sids = feed.stops.stop_id.loc[:n]
    for split_directions in [True, False]:
        f = compute_stop_stats(
            feed, dates, stop_ids=sids, split_directions=split_directions
        )

        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)

        # Should contain the correct stops
        get = set(f["stop_id"].values)
        g = get_stops(feed, dates[0]).loc[lambda x: x["stop_id"].isin(sids)]
        expect = set(g["stop_id"].values)
        assert get == expect

        # Should contain the correct columns
        expect_cols = {
            "date",
            "stop_id",
            "num_routes",
            "num_trips",
            "max_headway",
            "min_headway",
            "mean_headway",
            "start_time",
            "end_time",
        }
        if split_directions:
            expect_cols.add("direction_id")

        assert set(f.columns) == expect_cols

        # Should have correct dates
        f.date.tolist() == cairns_dates

        # Empty dates should yield empty DataFrame
        f = compute_stop_stats(feed, [], split_directions=split_directions)
        assert f.empty


def test_build_zero_stop_time_series():
    feed = cairns.copy()
    for split_directions in [True, False]:
        if split_directions:
            expect_names = ["indicator", "stop_id", "direction_id"]
            expect_shape = (2, feed.stops.shape[0] * 2)
        else:
            expect_names = ["indicator", "stop_id"]
            expect_shape = (2, feed.stops.shape[0])

        f = build_zero_stop_time_series(
            feed, split_directions=split_directions, freq="12H"
        )

        assert isinstance(f, pd.core.frame.DataFrame)
        assert f.shape == expect_shape
        assert f.columns.names == expect_names
        assert not f.values.any()


def test_compute_stop_time_series():
    feed = cairns.copy()
    dates = cairns_dates + ["20010101"]  # Spans 3 valid dates
    n = 3
    sids = feed.stops.stop_id.loc[:n]

    for split_directions in [True, False]:
        s = compute_stop_stats(
            feed, dates, stop_ids=sids, split_directions=split_directions
        )
        ts = compute_stop_time_series(
            feed,
            dates,
            stop_ids=sids,
            freq="12H",
            split_directions=split_directions,
        )

        # Should be a data frame
        assert isinstance(ts, pd.core.frame.DataFrame)

        # Should have the correct shape
        assert ts.shape[0] == 3 * 2  # 3 dates at 12H freq
        assert ts.shape[1] == s.shape[0] / 2

        # Should have correct column names
        if split_directions:
            expect_names = ["indicator", "stop_id", "direction_id"]
        else:
            expect_names = ["indicator", "stop_id"]
        assert ts.columns.names == expect_names

        # Should have correct index name
        assert ts.index.name == "datetime"

        # Each stop should have a correct total trip count
        if split_directions == False:
            sg = s.groupby("stop_id")
            for stop in s.stop_id.values:
                get = ts["num_trips"][stop].sum()
                expect = sg.get_group(stop)["num_trips"].sum()
                # Stop stats could have more num trips in case of
                # trips without departure times
                assert get <= expect

        # Empty dates should yield empty DataFrame
        ts = compute_stop_time_series(
            feed, [], split_directions=split_directions
        )
        assert ts.empty


def test_build_stop_timetable():
    feed = cairns.copy()
    stop_id = feed.stops["stop_id"].values[0]
    dates = cairns_dates + ["20010101"]
    f = build_stop_timetable(feed, stop_id, dates)

    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)

    # Should have the correct columns
    expect_cols = (
        set(feed.trips.columns) | set(feed.stop_times.columns) | set(["date"])
    )
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == cairns_dates

    # Empty check
    f = build_stop_timetable(feed, stop_id, [])
    assert f.empty


def test__geometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = _geometrize_stops(stops, use_utm=True)
    # Should be a GeoDataFrame
    assert isinstance(geo_stops, gpd.GeoDataFrame)
    # Should have the correct shape
    assert geo_stops.shape[0] == stops.shape[0]
    assert geo_stops.shape[1] == stops.shape[1] - 1
    # Should have the correct columns
    expect_cols = set(list(stops.columns) + ["geometry"]) - set(
        ["stop_lon", "stop_lat"]
    )
    assert set(geo_stops.columns) == expect_cols

def test__ungeometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = _geometrize_stops(stops)
    stops2 = _ungeometrize_stops(geo_stops)
    # Test columns are correct
    assert set(stops2.columns) == set(stops.columns)
    # Data frames should be equal after sorting columns
    cols = sorted(stops.columns)
    assert_frame_equal(stops2[cols], stops[cols])

def test_geometrize_shapes():
    g_1 = geometrize_stops(cairns)
    g_2 = _geometrize_stops(cairns.stops)
    assert g_1.equals(g_2)

def test_stops_to_geojson():
    feed = cairns.copy()
    stop_ids = feed.stops.stop_id.unique()[:2]
    collection = stops_to_geojson(feed, stop_ids)
    assert isinstance(collection, dict)
    assert len(collection["features"]) == len(stop_ids)

def test_get_stops_in_polygon():
    feed = cairns.copy()
    with (DATA_DIR / "cairns_square_stop_750070.geojson").open() as src:
        polygon = sg.shape(json.load(src)["features"][0]["geometry"])
    pstops = get_stops_in_polygon(feed, polygon)
    stop_ids = ["750070"]
    assert pstops["stop_id"].values == stop_ids

def test_map_stops():
    feed = cairns.copy()
    m = map_trips(feed, feed.stops.stop_id.iloc[:5])
    assert isinstance(m, fl.Map)
