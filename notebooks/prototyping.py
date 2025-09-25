import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    import sys
    import os
    import dateutil.relativedelta as rd
    import json
    import pathlib as pl
    from typing import List
    import warnings

    import marimo as mo
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import shapely
    import shapely.geometry as sg
    import shapely.ops as so
    import folium as fl
    import plotly.express as px

    import gtfs_kit as gk

    warnings.filterwarnings("ignore")

    DATA = pl.Path("data")
    return DATA, gk


@app.cell
def _(DATA, gk):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    #feed = gk.read_feed(pl.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km")
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km")
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _(feed):
    trip_stats = feed.compute_trip_stats().iloc[:100]
    trip_stats
    return (trip_stats,)


@app.cell
def _(dates, feed, gk, trip_stats):
    ts = gk.compute_route_time_series(feed, dates, trip_stats=trip_stats, freq="h")
    ts
    return (ts,)


@app.cell
def _(gk, ts):
    gk.downsample(ts, freq="3h")
    return


@app.cell
def _(dates, feed, gk, trip_stats):
    gk.compute_network_stats(feed, dates, trip_stats=trip_stats)
    return


@app.cell
def _(dates, feed, trip_stats):
    feed.compute_route_stats(dates, trip_stats=trip_stats)
    return


@app.cell
def _(dates, feed, trip_stats):
    rts = feed.compute_route_time_series(dates, trip_stats=trip_stats, freq="6h")
    rts
    return


@app.cell
def _():
    # feed = gk.read_feed(DOWN / "gtfs_brevibus.zip", dist_units="km")
    # routes = feed.get_routes(as_gdf=True)
    # print(routes)
    # feed = feed.aggregate_routes()
    # feed.map_routes(feed.routes["route_id"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
