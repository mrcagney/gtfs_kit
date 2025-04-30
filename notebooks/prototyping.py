import marimo

__generated_with = "0.13.2"
app = marimo.App()


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
    return DATA, gk


@app.cell
def _(DATA, gk):
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")
    feed.shapes
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
