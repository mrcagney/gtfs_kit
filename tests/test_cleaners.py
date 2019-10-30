from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np

from .context import gtfs_kit, sample
from gtfs_kit import *


def test_clean_column_names():
    f = sample.routes.copy()
    g = clean_column_names(f)
    assert_frame_equal(f, g)

    f = sample.routes.copy()
    f[" route_id  "] = f["route_id"].copy()
    del f["route_id"]
    g = clean_column_names(f)
    assert "route_id" in g.columns
    assert " route_id  " not in g.columns


def test_clean_ids():
    f1 = sample.copy()
    f1.routes.loc[0, "route_id"] = "  ho   ho ho "
    f2 = clean_ids(f1)
    expect_rid = "ho_ho_ho"
    f2.routes.loc[0, "route_id"] == expect_rid

    f3 = clean_ids(f2)
    f3 == f2


def test_clean_times():
    f1 = sample.copy()
    f1.stop_times["departure_time"].iat[0] = "7:00:00"
    f1.frequencies["start_time"].iat[0] = "7:00:00"
    f2 = clean_times(f1)
    assert f2.stop_times["departure_time"].iat[0] == "07:00:00"
    assert f2.frequencies["start_time"].iat[0] == "07:00:00"


def test_clean_route_short_names():
    f1 = sample.copy()

    # Should have no effect on a fine feed
    f2 = clean_route_short_names(f1)
    assert_series_equal(f2.routes["route_short_name"], f1.routes["route_short_name"])

    # Make route short name duplicates
    f1.routes.loc[1:5, "route_short_name"] = np.nan
    f1.routes.loc[6:, "route_short_name"] = "  he llo  "
    f2 = clean_route_short_names(f1)
    # Should have unique route short names
    assert f2.routes["route_short_name"].nunique() == f2.routes.shape[0]
    # NaNs should be replaced by n/a and route IDs
    expect_rsns = ("n/a-" + sample.routes.iloc[1:5]["route_id"]).tolist()
    assert f2.routes.iloc[1:5]["route_short_name"].values.tolist() == expect_rsns
    # Should have names without leading or trailing whitespace
    assert not f2.routes["route_short_name"].str.startswith(" ").any()
    assert not f2.routes["route_short_name"].str.endswith(" ").any()


def test_drop_zombies():
    # Should have no effect on sample feed
    f1 = sample.copy()
    f2 = drop_zombies(f1)
    assert_frame_equal(f2.routes, f1.routes)

    # Should drop stops with no stop times
    f1 = sample.copy()
    f1.stops["location_type"] = np.nan
    stop_id = f1.stops.stop_id.iat[0]
    st = f1.stop_times.copy()
    st = st.loc[lambda x: x.stop_id != stop_id]
    f1.stop_times = st
    f2 = drop_zombies(f1)
    assert not stop_id in f2.stops.stop_id.values

    f2 = drop_zombies(f1)
    assert_frame_equal(f2.routes, f1.routes)

    # Create undefined parent stations
    f1 = sample.copy()
    f1.stops["parent_station"] = "bingo"
    f2 = drop_zombies(f1)
    assert f2.stops.parent_station.isna().all()

    # Create all zombie trips for one route
    rid = f1.routes["route_id"].iat[0]
    cond = f1.trips["route_id"] == rid
    f1.trips.loc[cond, "trip_id"] = "hoopla"
    f2 = drop_zombies(f1)
    # Trips should be gone
    assert "hoopla" not in f2.trips["trip_id"]
    # Route should be gone
    assert rid not in f2.routes["route_id"]


def test_build_aggregate_routes_dict():
    routes = sample.routes.copy()
    # Equalize all route short names
    routes["route_short_name"] = "bingo"
    nid_by_oid = build_aggregate_routes_dict(routes, route_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_1"}


def test_aggregate_routes():
    feed1 = sample.copy()
    # Equalize all route short names
    feed1.routes["route_short_name"] = "bingo"
    feed2 = aggregate_routes(feed1)

    # feed2 should have only one route ID
    assert feed2.routes.shape[0] == 1

    # Feeds should have same trip data frames excluding
    # route IDs
    feed1.trips["route_id"] = feed2.trips["route_id"]
    assert almost_equal(feed1.trips, feed2.trips)

    # Feeds should have equal attributes excluding
    # routes and trips data frames
    feed2.routes = feed1.routes
    feed2.trips = feed1.trips
    assert feed1 == feed2


def test_build_aggregate_stops_dict():
    stops = sample.stops.copy()
    # Equalize all stop codes
    stops["stop_code"] = "bingo"
    nid_by_oid = build_aggregate_stops_dict(stops, stop_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_1"}


def test_aggregate_stops():
    feed1 = sample.copy()
    # Equalize all stop codes
    feed1.stops["stop_code"] = "bingo"
    feed2 = aggregate_stops(feed1)

    # feed2 should have only one stop ID
    assert feed2.stops.shape[0] == 1

    # Feeds should have same stop times, excluding stop IDs
    feed1.stop_times["stop_id"] = feed2.stop_times.stop_id
    assert almost_equal(feed1.stop_times, feed2.stop_times)

    # Feeds should have equal attributes excluding
    # stops stop times data frames
    feed2.stops = feed1.stops
    feed2.stop_times = feed1.stop_times
    assert feed1 == feed2


def test_clean():
    f1 = sample.copy()
    rid = f1.routes["route_id"].iat[0]
    f1.routes["route_id"].iat[0] = " " + rid + "   "
    f2 = clean(f1)
    assert f2.routes["route_id"].iat[0] == rid
    assert_frame_equal(f2.trips, sample.trips)


def test_drop_invalid_columns():
    f1 = sample.copy()
    f1.routes["bingo"] = "bongo"
    f1.trips["wingo"] = "wongo"
    f2 = drop_invalid_columns(f1)
    assert f2 == sample
