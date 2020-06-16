import pytest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from .context import gtfs_kit, DATA_DIR
from gtfs_kit import *


def test_feed():
    feed = Feed(agency=pd.DataFrame(), dist_units="km")
    for key in cs.FEED_ATTRS:
        val = getattr(feed, key)
        if key == "dist_units":
            assert val == "km"
        elif key == "agency":
            assert isinstance(val, pd.DataFrame)
        else:
            assert val is None


def test_str():
    assert isinstance(str(feed), str)


def test_eq():
    assert Feed(dist_units="m") == Feed(dist_units="m")

    feed1 = Feed(
        dist_units="m", stops=pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]),
    )
    assert feed1 == feed1

    feed2 = Feed(
        dist_units="m", stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = Feed(
        dist_units="m", stops=pd.DataFrame([[3, 4], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 != feed2

    feed2 = Feed(
        dist_units="m", stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = Feed(dist_units="mi", stops=feed1.stops)
    assert feed1 != feed2


def test_copy():
    feed1 = read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    feed2 = feed1.copy()

    # Check attributes
    for key in cs.FEED_ATTRS:
        val = getattr(feed2, key)
        expect_val = getattr(feed1, key)
        if isinstance(val, pd.DataFrame):
            assert_frame_equal(val, expect_val)
        elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
            assert val.groups == expect_val.groups
        else:
            assert val == expect_val


# --------------------------------------------
# Test functions about inputs and outputs
# --------------------------------------------
def test_list_feed():
    # Bad path
    with pytest.raises(ValueError):
        list_feed("bad_path!")

    for path in [DATA_DIR / "sample_gtfs.zip", DATA_DIR / "sample_gtfs"]:
        f = list_feed(path)
        assert isinstance(f, pd.DataFrame)
        assert set(f.columns) == {"file_name", "file_size"}
        assert f.shape[0] in [12, 13]


def test_read_feed():
    # Bad path
    with pytest.raises(ValueError):
        read_feed("bad_path!", dist_units="km")

    # Bad dist_units:
    with pytest.raises(ValueError):
        read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="bingo")

    # Requires dist_units:
    with pytest.raises(TypeError):
        read_feed(path=DATA_DIR / "sample_gtfs.zip")

    # Success
    feed = read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="m")

    # Success
    feed = read_feed(DATA_DIR / "sample_gtfs", dist_units="m")

    # Feed should have None feed_info table
    assert feed.feed_info is None


def test_write():
    feed1 = read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")

    # Export feed1, import it as feed2, and then test equality
    for out_path in [DATA_DIR / "bingo.zip", DATA_DIR / "bingo"]:
        feed1.write(out_path)
        feed2 = read_feed(out_path, "km")
        assert feed1 == feed2
        try:
            out_path.unlink()
        except:
            shutil.rmtree(str(out_path))

    # Test that integer columns with NaNs get output properly.
    # To this end, put a NaN, 1.0, and 0.0 in the direction_id column of trips.txt, export it, and import the column as strings.
    # Should only get np.nan, '0', and '1' entries.
    feed3 = read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    f = feed3.trips.copy()
    f["direction_id"] = f["direction_id"].astype(object)
    f.loc[0, "direction_id"] = np.nan
    f.loc[1, "direction_id"] = 1.0
    f.loc[2, "direction_id"] = 0.0
    feed3.trips = f
    q = DATA_DIR / "bingo.zip"
    feed3.write(q)

    tmp_dir = tempfile.TemporaryDirectory()
    shutil.unpack_archive(str(q), tmp_dir.name, "zip")
    qq = Path(tmp_dir.name) / "trips.txt"
    t = pd.read_csv(qq, dtype={"direction_id": str})
    assert t[~t["direction_id"].isin([np.nan, "0", "1"])].empty
    tmp_dir.cleanup()
    q.unlink()
