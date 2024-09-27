Changelog
=========

6.2.0, 2024-??-??
-----------------
- Switched from Poetry to UV for project management.

6.1.1, 2024-08-19
-----------------
- Changed grouped DataFrame ``feed._calendar_dates_g`` to indexed DataFrame ``feed._calendar_dates_i`` for consistency with ``feed._calendar_i`` and slight speedup in fucttion ``trips.is_active_trip``.
- Updated dependencies and dropped Python 3.8 support.
- Addressed some Pandas deprecation warnings.

6.1.0, 2024-02-02
-----------------
- Added ``cleaners.extend_id`` function in response to `Pull Request 7 <https://github.com/mrcagney/gtfs_kit/pull/7>`_.

6.0.1, 2024-01-30
-----------------
- Fixed a new GeoPandas 'set geometry' error.

6.0.0, 2023-10-03
-----------------
- Changed keywords in ``map_trips()`` and ``map_routes()`` because i keep remembering them wrong.

5.2.8, 2023-07-21
-----------------
- Bugfixed the ``check_attributions()`` validator.
- Changed ``compute_route_stats_0`` to ignore trips of zero duration, thereby addressing a different aspect of `Issue 2 <https://github.com/mrcagney/gtfs_kit/issues/2>`_.
- Updated dependencies.

5.2.7, 2023-06-06
-----------------
- Bugfixed ``get_peak_indices``, addressing `Issue 2 <https://github.com/mrcagney/gtfs_kit/issues/2>`_.

5.2.6, 2023-05-30
-----------------
- Bugfixed ``geometrize_routes``, addressing `Issue 1 <https://github.com/mrcagney/gtfs_kit/issues/1>`_.
- Removed star imports from tests.
- Updated dependencies.

5.2.5, 2023-04-26
-----------------
- Updated dependencies.
- Updated ``compute_screen_line_counts`` for Shapely >=2.

5.2.4, 2023-03-22
-----------------
- Updated dependencies and pre-commit hooks.
- Added a Github Action for testing.

5.2.3, 2022-06-28
-----------------
- Upgraded to Python 3.10 and updated dependencies.

5.2.2, 2022-04-27
-----------------
- Fixed ``transfer_type`` range in ``validators.py``.
  Was 0,...,4 but should have been 0,..,3.

5.2.1, 2022-04-12
-----------------
- Updated dependencies and removed version caps.
- Updated README.

5.2.0, 2022-01-17
-----------------
- Added support for ``attributions.txt``.
- Fixed ``aggregate_stops()`` docstring.

5.1.4, 2021-05-19
-----------------
- Bugfixed ``geometrize_routes(use_utm=True)`` to actually use UTM coordinates.

5.1.3, 2021-05-19
-----------------
- Bugfixed distance units in trip stats when shape_dist_traveled is present.

5.1.2, 2021-05-17
-----------------
- Changed distance units in trip stats, route stats, and feed stats to kilometers if the feed's distance units are metric and to miles otherwise.
- Added stop time information to stops when mapping trips with stops.

5.1.1, 2021-04-30
-----------------
- Handled fare rules in ``aggregate_routes()`` and dropped mistaken transfers code block therein.

5.1.0, 2021-04-29
-----------------
- Added support for Python 3.9 and dropped support for Python 3.6.

5.0.2, 2020-10-16
-----------------
- Specified in more detail the Rtree dependency.

5.0.1, 2020-10-08
-----------------
- Bugfix: properly set the ``use_utm`` flag in ``build_geometry_by_shape()`` and ``build_geometry_by_stop()``.

5.0.0, 2020-06-16
-----------------
- Breaking change: refactored ``get_stops_in_polygon()`` to ``get_stops_in_area()``, which accepts a GeoDataFrame.
- Breaking change: refactored ``restrict_to_polygon()`` to ``restrict_to_area()``, which accepts a GeoDataFrame.
- Breaking changes: refactored ``compute_center()`` to ``compute_centroid()``.
- Updated ``get_utm_crs()`` to differentiate between northern and southern hemispheres.
- Added more defensive copying after subsetting some DataFrames.
- Fixed calendar_dates table in ``restrict_to_dates()``.
- Added ``compute_convex_hull()`` to Feed methods. Forgot about that function.
- Switched from using route IDs to using route short names for layer names in ``map_routes()``.

4.0.2, 2020-05-07
-----------------
- Fixed a CRS deprecation warning as requested in `Pull Request 5 <https://github.com/mrcagney/gtfs_kit/pull/5>`_.
- Changed ``get_utm_crs()`` to output an EPSG CRS string, e.g. "EPSG:32655", instead of a PROJ4 definition string. Did this under the recommendation of the `GeoPandas docs <https://geopandas.org/projections.html#manually-specifying-the-crs>`_.
- Fixed CRS mismatch warning in ``compute_screen_line_counts()``.
- Updated dependencies and included Python 3.8 support.

4.0.1, 2020-04-24
-----------------
- Bugfix: got ``read_feed()`` working on Windows thanks to `Pull Request 4 <https://github.com/mrcagney/gtfs_kit/pull/4>`_.

4.0.0, 2020-03-06
-----------------
- Breaking changes: renamed ``list_gtfs()`` to ``list_feed()``, ``read_gtfs()`` to ``read_feed()``, and ``write_gtfs()`` to ``write()`` and made it a Feed method.
- Made ``read_feed()`` accept URLs as requested in `Pull Request 3 <https://github.com/mrcagney/gtfs_kit/pull/3>`_.

3.0.1, 2020-01-16
-----------------
- Optimized function ``geometrize_routes()`` by ignoring duplicate shapes.

3.0.0, 2020-01-10
-----------------
- Breaking change: improved function ``compute_screen_line_counts()`` to handle multiple screen lines at once.
- Added helper function ``make_ids()``.

2.2.1, 2019-11-07
-----------------
- Bugfix: updated function ``map_trips()`` to heed the ``include_arrows`` parameter.

2.2.0, 2019-10-31
-----------------
- Modularized some by added the functions ``build_aggregate_routes_dict()`` and ``build_aggregate_stops_dict()``.

2.1.0, 2019-10-10
-----------------
- Bugfix: updated ``aggregate_stops()`` to handle parent stations.
- Added optional direction arrows to ``map_trips()``.

2.0.0, 2019-10-04
-----------------
- Improved the fallback algorithm in ``append_dist_to_stop_times()``. Changed the function signature, so this is a major change, hence the major version bump.
- Removed optional direction arrows in ``map_trips()``, because the PolyLineTextPath Folium plugin needed for that seems to be broken.

1.1.1, 2019-09-25
-----------------
- Bugfix: imported ``aggregate_stops()`` as a Feed method.

1.1.0, 2019-09-25
-----------------
- Added ``aggregate_stops()`` function.
- Added optional direction arrows in ``map_trips()``.

1.0.2, 2019-09-20
-----------------
- Bugfix: Fixed CRS in ``geometrize_trips()`` and ``geometrize_routes()`` when ``use_utm=True``.

1.0.1, 2019-09-20
-----------------
- Bugfixed: Fixed occasional indexing error in ``geometrize_stops()`` and ``geometrize_shapes()`` when ``use_utm=True``.

1.0.0, 2019-09-18
-----------------
- First release based on prior work.
