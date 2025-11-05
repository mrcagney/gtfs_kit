---
title: Examples
marimo-version: 0.17.0
width: medium
---

```python {.marimo}
# /// script
# [tool.marimo.display]
# theme = "light"
# ///
```

```python {.marimo}
import pathlib as pl
import json

import marimo as mo
import pandas as pd
import numpy as np
import geopandas as gp
import matplotlib
import folium as fl

import gtfs_kit as gk


HERE = pl.Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = HERE.parent  # notebooks/ -> project/
DATA = (PROJECT_ROOT / "data").resolve()
```

```python {.marimo}
# List feed

gk.list_feed(DATA / "cairns_gtfs.zip")
```

```python {.marimo}
# Read feed and describe

feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")
feed.describe()
```

```python {.marimo}
mo.output.append(feed.stop_times)
feed_1 = feed.append_dist_to_stop_times()
mo.output.append(feed_1.stop_times)
```

```python {.marimo}
week = feed_1.get_first_week()
dates = [week[0], week[6]]
dates
```

```python {.marimo}
# Trip stats; reuse these for later speed ups

trip_stats = feed_1.compute_trip_stats()
trip_stats
```

```python {.marimo}
# Pass in trip stats to avoid recomputing them

network_stats = feed_1.compute_network_stats(dates)
network_stats
```

```python {.marimo}
nts = feed_1.compute_network_time_series(dates, freq="6h")
nts
```

```python {.marimo}
gk.downsample(nts, freq="12h")
```

```python {.marimo}
# Stop time series
stop_ids = feed.stops.loc[:1, "stop_id"]
sts = feed_1.compute_stop_time_series(dates, stop_ids=stop_ids, freq="12h")
sts
```

```python {.marimo}
gk.downsample(sts, freq="d")
```

```python {.marimo}
# Route time series

rts = feed_1.compute_route_time_series(dates, freq="12h")
rts
```

```python {.marimo}
# Route timetable

route_id = feed_1.routes["route_id"].iat[0]
feed_1.build_route_timetable(route_id, dates)
```

```python {.marimo}
# Locate trips

rng = pd.date_range("1/1/2000", periods=24, freq="h")
times = [t.strftime("%H:%M:%S") for t in rng]
loc = feed_1.locate_trips(dates[0], times)
loc.head()
```

```python {.marimo}
# Map routes

rsns = feed_1.routes["route_short_name"].iloc[2:4]
feed_1.map_routes(route_short_names=rsns, show_stops=True)
```

```python {.marimo}
# Alternatively map routes without stops using GeoPandas's explore

(
    feed.get_routes(as_gdf=True).explore(
        column="route_short_name",
        style_kwds=dict(weight=3),
        highlight_kwds=dict(weight=8),
        tiles="CartoDB positron",
    )
)
```

```python {.marimo}
# Show screen line

trip_id = "CNS2014-CNS_MUL-Weekday-00-4166247"
m = feed_1.map_trips([trip_id], show_stops=True, show_direction=True)
screen_line = gp.read_file(DATA / "cairns_screen_line.geojson")
keys_to_remove = [
    key
    for key in m._children.keys()
    if key.startswith("layer_control_") or key.startswith("fit_bounds_")
]
for key in keys_to_remove:
    m._children.pop(key)
fg = fl.FeatureGroup(name="Screen lines")
fl.GeoJson(
    screen_line, style_function=lambda feature: {"color": "red", "weight": 2}
).add_to(fg)
fg.add_to(m)
fl.LayerControl().add_to(m)
m.fit_bounds(fg.get_bounds())
m
```

```python {.marimo}
# Screen line counts

slc = feed_1.compute_screen_line_counts(screen_line, dates=dates)
slc.loc[lambda x: x["trip_id"] == trip_id]
```

```python {.marimo}

```