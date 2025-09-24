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
    return DATA, gk, mo


@app.cell
def _(DATA, gk):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
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
    trip_stats = feed.compute_trip_stats().iloc[:10]
    trip_stats
    return (trip_stats,)


@app.cell
def _(dates, feed, gk):
    gk.compute_network_stats(feed, dates)
    return


@app.cell
def _(dates, feed, gk, mo):
    ts = gk.compute_route_time_series(feed, dates, freq="h")
    mo.output.append(ts)
    ts2 = gk.downsample(ts, freq="12h")
    mo.output.append(ts2)
    return


@app.cell
def _():
    return


@app.cell
def _(dates, feed, trip_stats):
    rts = feed.compute_route_time_series(dates, trip_stats)
    rts
    return


@app.cell
def _(dates, feed, trip_stats):
    feed.compute_route_stats(trip_stats, dates=dates)
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
