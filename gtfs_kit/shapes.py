"""
Functions about shapes.
"""

from __future__ import annotations
from typing import Optional, Iterable, TYPE_CHECKING
import json

import geopandas as gp
import pandas as pd
import numpy as np
import utm
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def append_dist_to_shapes(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` field in
    ``feed.shapes`` in terms of the distance units ``feed.dist_units``.
    Return the resulting Feed.

    As a benchmark, using this function on `this Portland feed
    <https://transitfeeds.com/p/trimet/43/1400947517>`_
    produces a ``shape_dist_traveled`` column that differs by at most
    0.016 km in absolute value from of the original values.
    """
    if feed.shapes is None:
        raise ValueError("This function requires the feed to have a shapes.txt file")

    feed = feed.copy()
    f = feed.shapes
    m_to_dist = hp.get_convert_dist("m", feed.dist_units)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        group = group.sort_values("shape_pt_sequence")
        shape = group["shape_id"].iat[0]
        if not isinstance(shape, str):
            group["shape_dist_traveled"] = np.nan
            return group
        points = [
            sg.Point(utm.from_latlon(lat, lon)[:2])
            for lon, lat in group[["shape_pt_lon", "shape_pt_lat"]].values
        ]
        p_prev = points[0]
        d = 0
        distances = [0]
        for p in points[1:]:
            d += p.distance(p_prev)
            distances.append(d)
            p_prev = p
        group["shape_dist_traveled"] = distances
        return group

    g = f.groupby("shape_id", group_keys=False).apply(compute_dist)
    # Convert from meters
    g["shape_dist_traveled"] = g["shape_dist_traveled"].map(m_to_dist)

    feed.shapes = g
    return feed


def geometrize_shapes(
    shapes: pd.DataFrame, *, use_utm: bool = False
) -> gp.GeoDataFrame:
    """
    Given a GTFS shapes DataFrame, convert it to a GeoDataFrame of LineStrings
    and return the result, which will no longer have the columns
    ``'shape_pt_sequence'``, ``'shape_pt_lon'``,
    ``'shape_pt_lat'``, and ``'shape_dist_traveled'``.

    If ``use_utm``, then use local UTM coordinates for the geometries.
    """

    def my_agg(group):
        d = {}
        d["geometry"] = sg.LineString(group[["shape_pt_lon", "shape_pt_lat"]].values)
        return pd.Series(d)

    g = (
        shapes.sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id", sort=False)
        .apply(my_agg)
        .reset_index()
        .pipe(gp.GeoDataFrame, crs=cs.WGS84)
    )

    if use_utm:
        lat, lon = shapes[["shape_pt_lat", "shape_pt_lon"]].values[0]
        crs = hp.get_utm_crs(lat, lon)
        g = g.to_crs(crs)

    return g


def ungeometrize_shapes(shapes_g: gp.GeoDataFrame) -> pd.DataFrame:
    """
    The inverse of :func:`geometrize_shapes`.

    If ``shapes_g`` is in UTM coordinates (has a UTM CRS property),
    then convert those UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    shapes_g = shapes_g.to_crs(cs.WGS84)

    F = []
    for index, row in shapes_g.iterrows():
        F.extend(
            [
                [row["shape_id"], i, x, y]
                for i, (x, y) in enumerate(row["geometry"].coords)
            ]
        )

    return pd.DataFrame(
        F,
        columns=[
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lon",
            "shape_pt_lat",
        ],
    )


def get_shapes(
    feed: "Feed", *, as_gdf: bool = False, use_utm: bool = False
) -> gp.DataFrame | None:
    """
    Get the shapes DataFrame for the given feed, which could be ``None``.
    If ``as_gdf``, then return it as GeoDataFrame with a 'geometry' column
    of linestrings and no 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat',
    'shape_dist_traveled' columns.
    The GeoDataFrame will have a UTM CRS if ``use_utm``; otherwise it will have a
    WGS84 CRS.
    """
    f = feed.shapes
    if f is not None and as_gdf:
        f = geometrize_shapes(f, use_utm=use_utm)
    return f


def build_geometry_by_shape(
    feed: "Feed", shape_ids: Iterable[str] | None = None, *, use_utm: bool = False
) -> dict:
    """
    Return a dictionary of the form
    <shape ID> -> <Shapely LineString representing shape>.
    If the Feed has no shapes, then return the empty dictionary.
    If ``use_utm``, then use local UTM coordinates; otherwise, use WGS84 coordinates.
    """
    if feed.shapes is None:
        return dict()

    g = get_shapes(feed, as_gdf=True, use_utm=use_utm)
    if shape_ids is not None:
        g = g.loc[lambda x: x["shape_id"].isin(shape_ids)]
    return dict(g[["shape_id", "geometry"]].values)


def shapes_to_geojson(feed: "Feed", shape_ids: Iterable[str] | None = None) -> dict:
    """
    Return a GeoJSON FeatureCollection of LineString features
    representing ``feed.shapes``.
    If the Feed has no shapes, then the features will be an empty list.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of shape IDs is given, then subset to those shapes.
    If the subset is empty, then return a FeatureCollection with an empty list of
    features.
    """
    g = get_shapes(feed, as_gdf=True)
    if shape_ids is not None:
        g = g.loc[lambda x: x["shape_id"].isin(shape_ids)]
    if g is None or g.empty:
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = hp.drop_feature_ids(json.loads(g.to_json()))
    return result


def get_shapes_intersecting_geometry(
    feed: "Feed",
    geometry: sg.base.BaseGeometry,
    shapes_g: gp.GeoDataFrame | None = None,
    *,
    as_gdf: bool = False,
) -> pd.DataFrame | None:
    """
    If the Feed has no shapes, then return None.
    Otherwise, return the subset of ``feed.shapes`` that contains all shapes that
    intersect the given Shapely WGS84 geometry, e.g. a Polygon or LineString.

    If ``as_gdf``, then return the shapes as a GeoDataFrame.
    Specifying ``shapes_g`` will skip the first step of the
    algorithm, namely, geometrizing ``feed.shapes``.
    """
    if feed.shapes is None:
        return None

    if shapes_g is not None:
        g = shapes_g.copy()
    else:
        g = get_shapes(feed, as_gdf=True)

    cols = g.columns
    g["hit"] = g["geometry"].intersects(geometry)
    g = g.loc[lambda x: x["hit"]].filter(cols)

    if as_gdf:
        result = g
    else:
        result = ungeometrize_shapes(g)

    return result
