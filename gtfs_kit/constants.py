"""
Constants useful across modules.
"""

import pandas as pd

# Record some data from the GTFS reference at https://gtfs.org/reference/static.
columns = ["table", "table_required", "column", "column_required", "dtype"]
rows = [
    ["agency", True, "agency_id", False, "str"],
    ["agency", True, "agency_name", True, "str"],
    ["agency", True, "agency_url", True, "str"],
    ["agency", True, "agency_timezone", True, "str"],
    ["agency", True, "agency_lang", False, "str"],
    ["agency", True, "agency_phone", False, "str"],
    ["agency", True, "agency_fare_url", False, "str"],
    ["agency", True, "agency_email", False, "str"],
    ["calendar", False, "service_id", True, "str"],
    ["calendar", False, "monday", True, "int"],
    ["calendar", False, "tuesday", True, "int"],
    ["calendar", False, "wednesday", True, "int"],
    ["calendar", False, "thursday", True, "int"],
    ["calendar", False, "friday", True, "int"],
    ["calendar", False, "saturday", True, "int"],
    ["calendar", False, "sunday", True, "int"],
    ["calendar", False, "start_date", True, "str"],
    ["calendar", False, "end_date", True, "str"],
    ["calendar_dates", False, "service_id", True, "str"],
    ["calendar_dates", False, "date", True, "str"],
    ["calendar_dates", False, "exception_type", True, "int"],
    ["fare_attributes", False, "fare_id", True, "str"],
    ["fare_attributes", False, "price", True, "float"],
    ["fare_attributes", False, "currency_type", True, "str"],
    ["fare_attributes", False, "payment_method", True, "int"],
    ["fare_attributes", False, "transfers", True, "int"],
    ["fare_attributes", False, "transfer_duration", False, "int"],
    ["fare_rules", False, "fare_id", True, "str"],
    ["fare_rules", False, "route_id", False, "str"],
    ["fare_rules", False, "origin_id", False, "str"],
    ["fare_rules", False, "destination_id", False, "str"],
    ["fare_rules", False, "contains_id", False, "str"],
    ["feed_info", False, "feed_publisher_name", True, "str"],
    ["feed_info", False, "feed_publisher_url", True, "str"],
    ["feed_info", False, "feed_lang", True, "str"],
    ["feed_info", False, "feed_start_date", False, "str"],
    ["feed_info", False, "feed_end_date", False, "str"],
    ["feed_info", False, "feed_version", False, "str"],
    ["frequencies", False, "trip_id", True, "str"],
    ["frequencies", False, "start_time", True, "str"],
    ["frequencies", False, "end_time", True, "str"],
    ["frequencies", False, "headway_secs", True, "int"],
    ["frequencies", False, "exact_times", False, "int"],
    ["routes", True, "route_id", True, "str"],
    ["routes", True, "agency_id", False, "str"],
    ["routes", True, "route_short_name", True, "str"],
    ["routes", True, "route_long_name", True, "str"],
    ["routes", True, "route_desc", False, "str"],
    ["routes", True, "route_type", True, "int"],
    ["routes", True, "route_url", False, "str"],
    ["routes", True, "route_color", False, "str"],
    ["routes", True, "route_text_color", False, "str"],
    ["shapes", False, "shape_id", True, "str"],
    ["shapes", False, "shape_pt_lat", True, "float"],
    ["shapes", False, "shape_pt_lon", True, "float"],
    ["shapes", False, "shape_pt_sequence", True, "int"],
    ["shapes", False, "shape_dist_traveled", False, "float"],
    ["stops", True, "stop_id", True, "str"],
    ["stops", True, "stop_code", False, "str"],
    ["stops", True, "stop_name", True, "str"],
    ["stops", True, "stop_desc", False, "str"],
    ["stops", True, "stop_lat", True, "float"],
    ["stops", True, "stop_lon", True, "float"],
    ["stops", True, "zone_id", False, "str"],
    ["stops", True, "stop_url", False, "str"],
    ["stops", True, "location_type", False, "int"],
    ["stops", True, "parent_station", False, "str"],
    ["stops", True, "stop_timezone", False, "str"],
    ["stops", True, "wheelchair_boarding", False, "int"],
    ["stop_times", True, "trip_id", True, "str"],
    ["stop_times", True, "arrival_time", True, "str"],
    ["stop_times", True, "departure_time", True, "str"],
    ["stop_times", True, "stop_id", True, "str"],
    ["stop_times", True, "stop_sequence", True, "int"],
    ["stop_times", True, "stop_headsign", False, "str"],
    ["stop_times", True, "pickup_type", False, "int"],
    ["stop_times", True, "drop_off_type", False, "int"],
    ["stop_times", True, "shape_dist_traveled", False, "float"],
    ["stop_times", True, "timepoint", False, "int"],
    ["transfers", False, "from_stop_id", True, "str"],
    ["transfers", False, "to_stop_id", True, "str"],
    ["transfers", False, "transfer_type", True, "int"],
    ["transfers", False, "min_transfer_time", False, "int"],
    ["trips", True, "route_id", True, "str"],
    ["trips", True, "service_id", True, "str"],
    ["trips", True, "trip_id", True, "str"],
    ["trips", True, "trip_headsign", False, "str"],
    ["trips", True, "trip_short_name", False, "str"],
    ["trips", True, "direction_id", False, "int"],
    ["trips", True, "block_id", False, "str"],
    ["trips", True, "shape_id", False, "str"],
    ["trips", True, "wheelchair_accessible", False, "int"],
    ["trips", True, "bikes_allowed", False, "int"],
]
GTFS_REF = pd.DataFrame(rows, columns=columns)

#: Columns that must be formatted as integers when outputting GTFS
INT_COLS = GTFS_REF.loc[GTFS_REF["dtype"] == "int", "column"].values.tolist()

#: Columns that must be read as strings by Pandas
STR_COLS = GTFS_REF.loc[GTFS_REF["dtype"] == "str", "column"].values.tolist()

DTYPE = {col: str for col in STR_COLS}

#: Valid distance units
DIST_UNITS = ["ft", "mi", "m", "km"]

#: Primary feed attributes
FEED_ATTRS_1 = [
    "agency",
    "calendar",
    "calendar_dates",
    "fare_attributes",
    "fare_rules",
    "feed_info",
    "frequencies",
    "routes",
    "shapes",
    "stops",
    "stop_times",
    "trips",
    "transfers",
    "dist_units",
]

#: Secondary feed attributes; derived from primary ones
FEED_ATTRS_2 = ["_trips_i", "_calendar_i", "_calendar_dates_g"]

#:
FEED_ATTRS = FEED_ATTRS_1 + FEED_ATTRS_2

#: WGS84 coordinate reference system for Geopandas
WGS84 = "EPSG:4326"

#: Colorbrewer 8-class Set2 colors
COLORS_SET2 = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
