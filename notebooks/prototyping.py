

import marimo

__generated_with = "0.13.2"
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

    import gtfs_kit as gk

    warnings.filterwarnings("ignore")

    DATA = pl.Path("data")
    return DATA, gk, pd


@app.cell
def _(DATA, gk, pd):
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")

    # Turning a route's shapes into point geometries, should yield an empty route geometry
    # and should not throw an error
    rid = feed.routes["route_id"].iat[0]
    shids = feed.trips.loc[lambda x: x["route_id"] == rid, "shape_id"]
    f0 = feed.shapes.loc[lambda x: x["shape_id"].isin(shids)].drop_duplicates("shape_id")
    f1 = feed.shapes.loc[lambda x: ~x["shape_id"].isin(shids)]
    feed.shapes = pd.concat([f0, f1])

    assert feed.get_routes(as_gdf=True).loc[lambda x: x["route_id"] == rid, "geometry"].iat[0] == None


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
