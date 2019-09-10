"""
Functions about shapes.
"""
from typing import Optional, List, Dict, TYPE_CHECKING

import pandas as pd
from pandas import DataFrame
import numpy as np
import utm
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def build_geometry_by_shape(
    feed: "Feed",
    shape_ids: Optional[List[str]] = None,
    *,
    use_utm: bool = False,
) -> Dict:
    """
    Return a dictionary with structure shape_id -> Shapely LineString
    of shape.

    Parameters
    ----------
    feed : Feed
    shape_ids : list
        IDs of shapes in ``feed.shapes`` to restrict output to; return
        all shapes if ``None``.
    use_utm : boolean
        If ``True``, then use local UTM coordinates; otherwise, use
        WGS84 coordinates

    Returns
    -------
    dictionary
        Has the structure
        shape_id -> Shapely LineString of shape.
        If ``feed.shapes is None``, then return ``None``.

        Return the empty dictionary if ``feed.shapes is None``.

    """
    if feed.shapes is None:
        return {}

    # Note the output for conversion to UTM with the utm package:
    # >>> u = utm.from_latlon(47.9941214, 7.8509671)
    # >>> print u
    # (414278, 5316285, 32, 'T')
    d = {}
    shapes = feed.shapes.copy()
    if shape_ids is not None:
        shapes = shapes[shapes["shape_id"].isin(shape_ids)]

    if use_utm:
        for shape, group in shapes.groupby("shape_id"):
            lons = group["shape_pt_lon"].values
            lats = group["shape_pt_lat"].values
            xys = [
                utm.from_latlon(lat, lon)[:2] for lat, lon in zip(lats, lons)
            ]
            d[shape] = sg.LineString(xys)
    else:
        for shape, group in shapes.groupby("shape_id"):
            lons = group["shape_pt_lon"].values
            lats = group["shape_pt_lat"].values
            lonlats = zip(lons, lats)
            d[shape] = sg.LineString(lonlats)
    return d


def shapes_to_geojson(
    feed: "Feed", shape_ids: Optional[List[str]] = None
) -> Dict:
    """
    Return a (decoded) GeoJSON FeatureCollection of LineString features
    representing ``feed.shapes``.
    Each feature will have a ``shape_id`` property.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If a list of shape IDs is given, then return only the LineString
    features corresponding to those shape IDS.
    Return the empty dictionary if ``feed.shapes is None``
    """
    geometry_by_shape = feed.build_geometry_by_shape(shape_ids=shape_ids)
    if geometry_by_shape:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {"shape_id": shape},
                    "type": "Feature",
                    "geometry": sg.mapping(linestring),
                }
                for shape, linestring in geometry_by_shape.items()
            ],
        }
    else:
        fc = {}
    return fc


def get_shapes_intersecting_geometry(
    feed: "Feed", geometry, geo_shapes=None, *, geometrized: bool = False
) -> DataFrame:
    """
    Return the slice of ``feed.shapes`` that contains all shapes that
    intersect the given Shapely geometry, e.g. a Polygon or LineString.

    Parameters
    ----------
    feed : Feed
    geometry : Shapley geometry, e.g. a Polygon
        Specified in WGS84 coordinates
    geo_shapes : GeoPandas GeoDataFrame
        The output of :func:`geometrize_shapes`
    geometrize : boolean
        If ``True``, then return the shapes DataFrame as a GeoDataFrame
        of the form output by :func:`geometrize_shapes`

    Returns
    -------
    DataFrame or GeoDataFrame

    Notes
    -----
    - Requires GeoPandas
    - Specifying ``geo_shapes`` will skip the first step of the
      algorithm, namely, geometrizing ``feed.shapes``
    - Assume the following feed attributes are not ``None``:

        * ``feed.shapes``, if ``geo_shapes`` is not given

    """
    if geo_shapes is not None:
        f = geo_shapes.copy()
    else:
        f = geometrize_shapes(feed.shapes)

    cols = f.columns
    f["hit"] = f["geometry"].intersects(geometry)
    f = f[f["hit"]][cols]

    if geometrized:
        return f
    else:
        return ungeometrize_shapes(f)


def append_dist_to_shapes(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` field in
    ``feed.shapes`` in terms of the distance units ``feed.dist_units``.
    Return the resulting Feed.

    Notes
    -----
    - As a benchmark, using this function on `this Portland feed
      <https://transitfeeds.com/p/trimet/43/1400947517>`_
      produces a ``shape_dist_traveled`` column that differs by at most
      0.016 km in absolute value from of the original values
    - Assume the following feed attributes are not ``None``:

        * ``feed.shapes``

    """
    if feed.shapes is None:
        raise ValueError(
            "This function requires the feed to have a shapes.txt file"
        )

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
    shapes: DataFrame, *, use_utm: bool = False
) -> DataFrame:
    """
    Given a GTFS shapes DataFrame, convert it to a GeoPandas
    GeoDataFrame and return the result.
    The result has a ``'geometry'`` column of WGS84 LineStrings
    instead of the columns ``'shape_pt_sequence'``, ``'shape_pt_lon'``,
    ``'shape_pt_lat'``, and ``'shape_dist_traveled'``.
    If ``use_utm``, then use local UTM coordinates for the geometries.

    Notes
    ------
    Requires GeoPandas.
    """
    import geopandas as gpd

    f = shapes.copy().sort_values(["shape_id", "shape_pt_sequence"])

    def my_agg(group):
        d = {}
        d["geometry"] = sg.LineString(
            group[["shape_pt_lon", "shape_pt_lat"]].values
        )
        return pd.Series(d)

    g = f.groupby("shape_id").apply(my_agg).reset_index()
    g = gpd.GeoDataFrame(g, crs=cs.WGS84)

    if use_utm:
        lat, lon = f.loc[0, ["shape_pt_lat", "shape_pt_lon"]].values
        crs = hp.get_utm_crs(lat, lon)
        g = g.to_crs(crs)

    return g


def ungeometrize_shapes(geo_shapes) -> DataFrame:
    """
    The inverse of :func:`geometrize_shapes`.
    Produces the columns:

    - ``'shape_id'``
    - ``'shape_pt_sequence'``
    - ``'shape_pt_lon'``
    - ``'shape_pt_lat'``

    If ``geo_shapes`` is in UTM coordinates (has a UTM CRS property),
    then convert thoes UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    geo_shapes = geo_shapes.to_crs(cs.WGS84)

    F = []
    for index, row in geo_shapes.iterrows():
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
