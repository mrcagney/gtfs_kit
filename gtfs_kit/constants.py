"""
Constants useful across modules.
"""

DTYPES = {
    "agency": {
        "agency_id": "string",
        "agency_name": "string",
        "agency_url": "string",
        "agency_timezone": "string",
        "agency_lang": "string",
        "agency_phone": "string",
        "agency_fare_url": "string",
        "agency_email": "string",
    },
    "attributions": {
        "attribution_id": "string",
        "agency_id": "string",
        "route_id": "string",
        "trip_id": "string",
        "organization_name": "string",
        "is_producer": "int32",
        "is_operator": "int32",
        "is_authority": "int32",
        "attribution_url": "string",
        "attribution_email": "string",
        "attribution_phone": "string",
    },
    "calendar": {
        "service_id": "string",
        "monday": "int32",
        "tuesday": "int32",
        "wednesday": "int32",
        "thursday": "int32",
        "friday": "int32",
        "saturday": "int32",
        "sunday": "int32",
        "start_date": "string",
        "end_date": "string",
    },
    "calendar_dates": {
        "service_id": "string",
        "date": "string",
        "exception_type": "int32",
    },
    "fare_attributes": {
        "fare_id": "string",
        "price": "float",
        "currency_type": "string",
        "payment_method": "int32",
        "transfers": "int32",
        "transfer_duration": "int32",
    },
    "fare_rules": {
        "fare_id": "string",
        "route_id": "string",
        "origin_id": "string",
        "destination_id": "string",
        "contains_id": "string",
    },
    "feed_info": {
        "feed_publisher_name": "string",
        "feed_publisher_url": "string",
        "feed_lang": "string",
        "feed_start_date": "string",
        "feed_end_date": "string",
        "feed_version": "string",
    },
    "frequencies": {
        "trip_id": "string",
        "start_time": "string",
        "end_time": "string",
        "headway_secs": "int32",
        "exact_times": "int32",
    },
    "routes": {
        "route_id": "string",
        "agency_id": "string",
        "route_short_name": "string",
        "route_long_name": "string",
        "route_desc": "string",
        "route_type": "int32",
        "route_url": "string",
        "route_color": "string",
        "route_text_color": "string",
    },
    "shapes": {
        "shape_id": "string",
        "shape_pt_lat": "float",
        "shape_pt_lon": "float",
        "shape_pt_sequence": "int32",
        "shape_dist_traveled": "float",
    },
    "stops": {
        "stop_id": "string",
        "stop_code": "string",
        "stop_name": "string",
        "stop_desc": "string",
        "stop_lat": "float",
        "stop_lon": "float",
        "zone_id": "string",
        "stop_url": "string",
        "location_type": "int32",
        "parent_station": "string",
        "stop_timezone": "string",
        "wheelchair_boarding": "int32",
    },
    "stop_times": {
        "trip_id": "string",
        "arrival_time": "string",
        "departure_time": "string",
        "stop_id": "string",
        "stop_sequence": "int32",
        "stop_headsign": "string",
        "pickup_type": "int32",
        "drop_off_type": "int32",
        "shape_dist_traveled": "float",
        "timepoint": "int32",
    },
    "transfers": {
        "from_stop_id": "string",
        "to_stop_id": "string",
        "transfer_type": "int32",
        "min_transfer_time": "int32",
    },
    "trips": {
        "route_id": "string",
        "service_id": "string",
        "trip_id": "string",
        "trip_headsign": "string",
        "trip_short_name": "string",
        "direction_id": "int32",
        "block_id": "string",
        "shape_id": "string",
        "wheelchair_accessible": "int32",
        "bikes_allowed": "int32",
    },
}

#: Columns that must be formatted as integers when outputting GTFS
INT_COLS = [col for d in DTYPES.values() for col in d if d[col].startswith("int")]

#: Columns that must be read as strings
STR_COLS = [col for d in DTYPES.values() for col in d if d[col] == "string"]

#: Valid distance units
DIST_UNITS = ["ft", "mi", "m", "km"]

#: Primary feed attributes
FEED_ATTRS = [
    "agency",
    "attributions",
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
