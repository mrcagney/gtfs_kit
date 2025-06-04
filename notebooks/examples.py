

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib as pl
    import json

    import pandas as pd
    import numpy as np
    import geopandas as gp
    import matplotlib
    import folium as fl

    import gtfs_kit as gk

    DATA = pl.Path("data")
    return DATA, fl, gk, gp, pd


@app.cell
def _(DATA, gk):
    # List feed

    path = DATA / "cairns_gtfs.zip"
    gk.list_feed(path)
    return (path,)


@app.cell
def _(gk, path):
    # Read feed and describe

    feed = gk.read_feed(path, dist_units="m")
    feed.describe()
    return (feed,)


@app.cell
def _(feed):
    print(feed.stop_times)
    feed_1 = feed.append_dist_to_stop_times()
    print(feed_1.stop_times)
    return (feed_1,)


@app.cell
def _(feed_1):
    week = feed_1.get_first_week()
    dates = [week[4], week[6]]
    dates
    return dates, week


@app.cell
def _(dates, feed_1):
    trip_stats = feed_1.compute_trip_stats()
    trip_stats.head().T
    fts = feed_1.compute_feed_time_series(trip_stats, dates, freq="6h")
    fts
    return fts, trip_stats


@app.cell
def _(fts, gk):
    gk.downsample(fts, freq="12h")
    return


@app.cell
def _(feed_1, trip_stats, week):
    feed_stats = feed_1.compute_feed_stats(trip_stats, week)
    feed_stats
    return


@app.cell
def _(dates, feed_1, trip_stats):
    rts = feed_1.compute_route_time_series(trip_stats, dates, freq="12h")
    rts
    return (rts,)


@app.cell
def _(rts):
    # Slice time series

    inds = ["service_distance", "service_duration", "service_speed"]
    rids = ["110-423", "111-423"]

    rts.loc[:, (inds, rids)]
    return (rids,)


@app.cell
def _(rids, rts):
    # Slice again by cross-section

    rts.xs(rids[0], axis="columns", level=1)
    return


@app.cell
def _(dates, feed_1, pd):
    rng = pd.date_range("1/1/2000", periods=24, freq="h")
    times = [t.strftime("%H:%M:%S") for t in rng]
    loc = feed_1.locate_trips(dates[0], times)
    loc.head()
    return


@app.cell
def _(dates, feed_1):
    route_id = feed_1.routes["route_id"].iat[0]
    feed_1.build_route_timetable(route_id, dates).T
    return


@app.cell
def _(DATA, feed_1, fl, gp):
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
    return screen_line, trip_id


@app.cell
def _(dates, feed_1, screen_line, trip_id):
    slc = feed_1.compute_screen_line_counts(screen_line, dates=dates)
    slc.loc[lambda x: x["trip_id"] == trip_id]
    return


@app.cell
def _(feed_1):
    rsns = feed_1.routes["route_short_name"].iloc[2:4]
    feed_1.map_routes(route_short_names=rsns, show_stops=True)
    return


@app.cell
def _(feed):
    # Alternatively plot routes using GeoPandas's explore

    (
        feed.get_routes(as_gdf=True).explore(
            column="route_short_name",
            style_kwds=dict(weight=3),
            highlight_kwds=dict(weight=8),
            tiles="CartoDB positron",
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
