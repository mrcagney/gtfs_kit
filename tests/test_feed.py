import pytest
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from .context import gtfs_kit, DATA_DIR
from gtfs_kit import feed as gkf
from gtfs_kit import constants as cs


def test_feed():
    feed = gkf.Feed(agency=pd.DataFrame(), dist_units="km")
    for key in cs.FEED_ATTRS:
        val = getattr(feed, key)
        if key == "dist_units":
            assert val == "km"
        elif key == "agency":
            assert isinstance(val, pd.DataFrame)
        else:
            assert val is None


def test_str():
    feed = gkf.Feed(agency=pd.DataFrame(), dist_units="km")
    assert isinstance(str(feed), str)


def test_eq():
    assert gkf.Feed(dist_units="m") == gkf.Feed(dist_units="m")

    feed1 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]),
    )
    assert feed1 == feed1

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[3, 4], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 != feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(dist_units="mi", stops=feed1.stops)
    assert feed1 != feed2


def test_copy():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
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
        gkf.list_feed("bad_path!")

    for path in [DATA_DIR / "sample_gtfs.zip", DATA_DIR / "sample_gtfs"]:
        f = gkf.list_feed(path)
        assert isinstance(f, pd.DataFrame)
        assert set(f.columns) == {"file_name", "file_size"}
        assert f.shape[0] in [12, 13]


def test_read_feed():
    # Bad path
    with pytest.raises(ValueError):
        gkf.read_feed("bad_path!", dist_units="km")

    # Bad dist_units:
    with pytest.raises(ValueError):
        gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="bingo")

    # Requires dist_units:
    with pytest.raises(TypeError):
        gkf.read_feed(path=DATA_DIR / "sample_gtfs.zip")

    # Success
    feed = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="m")

    # Success
    feed = gkf.read_feed(DATA_DIR / "sample_gtfs", dist_units="m")

    # Feed should have None feed_info table
    assert feed.feed_info is None


def test_to_file():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")

    # Export feed1, import it as feed2, and then test equality
    for out_path in [DATA_DIR / "bingo.zip", DATA_DIR / "bingo"]:
        feed1.to_file(out_path)
        feed2 = gkf.read_feed(out_path, "km")
        assert feed1 == feed2
        try:
            out_path.unlink()
        except Exception:
            shutil.rmtree(str(out_path))

    # Test that integer columns with NaNs get output properly.
    feed3 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    f = feed3.trips.copy()
    f.loc[0, "direction_id"] = np.nan
    f.loc[1, "direction_id"] = 1
    f.loc[2, "direction_id"] = 0
    feed3.trips = f
    q = DATA_DIR / "bingo.zip"
    feed3.to_file(q)

    tmp_dir = tempfile.TemporaryDirectory()
    shutil.unpack_archive(str(q), tmp_dir.name, "zip")
    qq = Path(tmp_dir.name) / "trips.txt"
    t = pd.read_csv(qq, dtype={"direction_id": "Int8"})
    assert t[~t["direction_id"].isin([pd.NA, 0, 1])].empty
    tmp_dir.cleanup()
    q.unlink()
