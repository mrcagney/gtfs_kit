import pytest
import pandas as pd
import numpy as np

from .context import gtfs_kit, DATA_DIR, sample, cairns, cairns_dates, cairns_trip_stats
from gtfs_kit import *


def test_get_stop_times():
    feed = cairns.copy()
    date = cairns_dates[0]
    f = get_stop_times(feed, date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have a reasonable shape
    assert f.shape[0] <= feed.stop_times.shape[0]
    # Should have correct columns
    assert set(f.columns) == set(feed.stop_times.columns)


def test_get_start_and_end_times():
    feed = cairns.copy()
    date = cairns_dates[0]
    st = get_stop_times(feed, date)
    times = get_start_and_end_times(feed, date)
    # Should be strings
    for t in times:
        assert isinstance(t, str)
        # Should lie in stop times
        assert t in st[["departure_time", "arrival_time"]].values.flatten()

    # Should get null times in some cases
    times = get_start_and_end_times(feed, "19690711")
    for t in times:
        assert pd.isnull(t)
    feed.stop_times["departure_time"] = np.nan
    times = get_start_and_end_times(feed)
    assert pd.isnull(times[0])


@pytest.mark.slow
def test_append_dist_to_stop_times():
    feed1 = cairns.copy()
    st1 = feed1.stop_times
    feed2 = append_dist_to_stop_times(feed1)
    st2 = feed2.stop_times

    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = st1.columns.values.tolist() + ["shape_dist_traveled"]
    cols2 = st2.columns.values.tolist()
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for trip, group in st2.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        sdt = group.shape_dist_traveled.values.tolist()
        assert sdt == sorted(sdt)


def test_stop_times_to_geojson():
    feed = cairns.copy()
    trip_ids = feed.trips.trip_id.unique()[:2]
    collection = stop_times_to_geojson(feed, trip_ids)
    assert isinstance(collection, dict)

    n = (
        feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)]
        .drop_duplicates(subset=["trip_id", "stop_id"])
        .shape[0]
    )
    assert len(collection["features"]) == n

    with pytest.raises(ValueError):
        stop_times_to_geojson(feed, ["bingo"])
