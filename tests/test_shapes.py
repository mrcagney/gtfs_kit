import json

import geopandas as gpd
import shapely.geometry as sg

from .context import gtfs_kit, DATA_DIR, cairns, cairns_shapeless
from gtfs_kit import shapes as gks
from gtfs_kit import constants as cs


def test_append_dist_to_shapes():
    feed1 = cairns.copy()
    s1 = feed1.shapes
    feed2 = gks.append_dist_to_shapes(feed1)
    s2 = feed2.shapes
    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = list(s1.columns.values) + ["shape_dist_traveled"]
    cols2 = list(s2.columns.values)
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for name, group in s2.groupby("shape_id"):
        sdt = list(group["shape_dist_traveled"].values)
        assert sdt == sorted(sdt)


def test_geometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = gks.geometrize_shapes(shapes)
    # Should be a GeoDataFrame
    assert isinstance(geo_shapes, gpd.GeoDataFrame)
    assert geo_shapes.crs == cs.WGS84
    # Should have the correct shape
    assert geo_shapes.shape[0] == shapes["shape_id"].nunique()
    assert geo_shapes.shape[1] == shapes.shape[1] - 2
    # Should have the correct columns
    expect_cols = set(list(shapes.columns) + ["geometry"]) - set(
        [
            "shape_pt_lon",
            "shape_pt_lat",
            "shape_pt_sequence",
            "shape_dist_traveled",
        ]
    )
    assert set(geo_shapes.columns) == expect_cols
    # A shape with only one point
    shapes = cairns.shapes.iloc[:1]
    geo_shapes = gks.geometrize_shapes(shapes)
    assert isinstance(geo_shapes, gpd.GeoDataFrame)
    assert geo_shapes.crs == cs.WGS84


def test_ungeometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = gks.geometrize_shapes(shapes)
    print(geo_shapes)
    shapes2 = gks.ungeometrize_shapes(geo_shapes)

    # Test columns are correct
    expect_cols = set(list(shapes.columns)) - set(["shape_dist_traveled"])
    assert set(shapes2.columns) == expect_cols

    # Data frames should agree on certain columns
    cols = ["shape_id", "shape_pt_lon", "shape_pt_lat"]
    print(shapes[cols])
    assert shapes2[cols].equals(shapes[cols])


def test_get_shapes():
    g = gks.get_shapes(cairns, as_gdf=True)
    assert g.crs == cs.WGS84
    assert set(g.columns) == {"shape_id", "geometry"}
    assert gks.get_shapes(cairns_shapeless, as_gdf=True) is None


def test_build_geometry_by_shape():
    d = gks.build_geometry_by_shape(cairns)
    assert isinstance(d, dict)
    assert len(d) == cairns.shapes.shape_id.nunique()
    assert gks.build_geometry_by_shape(cairns_shapeless) == {}


def test_shapes_to_geojson():
    feed = cairns.copy()
    shape_ids = feed.shapes.shape_id.unique()[:2]
    collection = gks.shapes_to_geojson(feed, shape_ids)
    assert isinstance(collection, dict)
    assert len(collection["features"]) == len(shape_ids)

    assert gks.shapes_to_geojson(cairns_shapeless) == {
        "type": "FeatureCollection",
        "features": [],
    }


def test_get_shapes_intersecting_geometry():
    feed = cairns.copy()
    path = DATA_DIR / "cairns_square_stop_750070.geojson"
    polygon = sg.shape(json.load(path.open())["features"][0]["geometry"])
    pshapes = gks.get_shapes_intersecting_geometry(feed, polygon)
    shape_ids = ["120N0005", "1200010", "1200001"]
    assert set(pshapes["shape_id"].unique()) == set(shape_ids)
    g = gks.get_shapes_intersecting_geometry(feed, polygon, as_gdf=True)
    assert g.crs == "epsg:4326"
    assert set(g["shape_id"].unique()) == set(shape_ids)
    assert gks.get_shapes_intersecting_geometry(cairns_shapeless, polygon) is None
