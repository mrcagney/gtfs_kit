import os
import sys
from pathlib import Path
import importlib

sys.path.insert(0, os.path.abspath(".."))

import geopandas as gpd
import pandas as pd
import numpy as np

import gtfs_kit
import pytest


# Load/create test feeds
DATA_DIR = Path("data")
sample = gtfs_kit.read_gtfs(DATA_DIR / "sample_gtfs.zip", dist_units="km")
cairns = gtfs_kit.read_gtfs(DATA_DIR / "cairns_gtfs.zip", dist_units="km")
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None
t = cairns_shapeless.trips
t["shape_id"] = np.nan
cairns_shapeless.trips = t
week = cairns.get_first_week()
cairns_dates = [week[0], week[2]]
cairns_trip_stats = pd.read_csv(
    DATA_DIR / "cairns_trip_stats.csv", dtype=gtfs_kit.DTYPE
)
