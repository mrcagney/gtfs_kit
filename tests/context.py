import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))

import gtfs_kit

# Load/create test feeds
DATA_DIR = Path("data")
sample = gtfs_kit.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
nyc_subway = gtfs_kit.read_feed(DATA_DIR / "nyc_subway_gtfs.zip", dist_units="mi")
cairns = gtfs_kit.read_feed(DATA_DIR / "cairns_gtfs.zip", dist_units="km")
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None
t = cairns_shapeless.trips
t["shape_id"] = np.nan
cairns_shapeless.trips = t
week = cairns.get_first_week()
cairns_dates = [week[0], week[6]]
cairns_trip_stats = pd.read_csv(
    DATA_DIR / "cairns_trip_stats.csv",
    dtype=(gtfs_kit.DTYPES["trips"] | gtfs_kit.DTYPES["routes"]),
)
