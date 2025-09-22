import marimo

__generated_with = "0.13.15"
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
    DOWN = pl.Path.home() / "Downloads"

    return gk, pd, px


@app.cell
def _(gk):
    {table: dict(g[["column", "dtype"]].values) for table, g in gk.GTFS_REF.groupby("table")}
    return


@app.cell
def _(gk):
    akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    feed = gk.read_feed(akl_url, dist_units="km")

    return (feed,)


@app.cell
def _(feed):
    print(feed.routes)
    print(feed.stop_times.head())
    return


@app.cell
def _(gk, pd, px):
    def slice_trip_stats(feed: gk.Feed, trip_stats: pd.DataFrame, date: str, route_ids: list|None=None) -> pd.DataFrame:
        tids = feed.get_trips(date)["trip_id"]
        f = trip_stats.loc[lambda x: x["trip_id"].isin(tids)]
        if route_ids:
            f = f.loc[lambda x: x["route_id"].isin(route_ids)]
        return f.copy()

    def plot_route_speeds(feed: gk.Feed, trip_stats: pd.DataFrame, date: str, route_ids: list|None=None):
        f = slice_trip_stats(feed, trip_stats, date, route_ids)
        f['start_time'] = pd.to_datetime(f["start_time"].map(gk.timestr_mod24), format='%H:%M:%S')
        # Clean some
        f = f.drop_duplicates(subset=["route_id", "start_time"]).sort_values(["route_id", "start_time"])
        date_obj = pd.to_datetime(date, format="%Y%m%d")
        fig = px.line(
            f,
            x='start_time',
            y='speed',
            color='route_short_name',
            labels={
                'start_time': 'Scheduled start time',
                'speed': 'Avg speed (km/h)',
                'route_short_name': 'Route'
            },
            title=f"Avg trip speeds by route on {date_obj:%A %Y-%m-%d}"
        )
        fig.update_xaxes(
            tickformat="%H:%M",
            ticklabelmode="period"
        )
        return fig

    return (plot_route_speeds,)


@app.cell
def _(feed):
    trip_stats = feed.compute_trip_stats()
    print(trip_stats.head())

    return (trip_stats,)


@app.cell
def _(feed, plot_route_speeds, trip_stats):
    date = "20250609"
    plot_route_speeds(feed, trip_stats, date)
    return


@app.cell
def _(feed):
    rsn = "995"
    rid = feed.routes.loc[lambda x: x["route_short_name"] == rsn, "route_id"].iat[0]
    feed.map_routes([rid], show_stops=True)
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
