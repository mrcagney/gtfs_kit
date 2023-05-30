import numpy as np
import pandas as pd

from .context import gtfs_kit, sample
from gtfs_kit import validators as gkv


def test_valid_str():
    assert gkv.valid_str("hello3")
    assert not gkv.valid_str(np.nan)
    assert not gkv.valid_str(" ")


def test_valid_time():
    assert gkv.valid_time("2:43:00")
    assert gkv.valid_time("22:43:00")
    assert gkv.valid_time("42:43:00")
    assert not gkv.valid_time("142:43:00")


def test_valid_date():
    assert gkv.valid_date("20140310")
    assert not gkv.valid_date("2014031")


def test_valid_timezone():
    assert gkv.valid_timezone("Africa/Abidjan")
    assert not gkv.valid_timezone("zoom")


def test_valid_lang():
    assert gkv.valid_lang("aa")
    assert not gkv.valid_lang("aaa")


def test_valid_currency():
    assert gkv.valid_currency("AED")
    assert not gkv.valid_currency("aed")


def test_valid_url():
    assert gkv.valid_url("http://www.example.com")
    assert not gkv.valid_url("www.example.com")


def test_valid_email():
    assert gkv.valid_email("a@b.c.com")
    assert not gkv.valid_email("a@b@c.com")


def test_valid_color():
    assert gkv.valid_color("00FFFF")
    assert not gkv.valid_color("0FF")
    assert not gkv.valid_color("GGFFFF")


def test_check_for_required_columns():
    assert not gkv.check_for_required_columns([], "routes", sample.routes)

    feed = sample.copy()
    del feed.routes["route_type"]
    assert gkv.check_for_required_columns([], "routes", feed.routes)


def test_check_for_invalid_columns():
    assert not gkv.check_for_invalid_columns([], "routes", sample.routes)

    feed = sample.copy()
    feed.routes["bingo"] = "snoop"
    assert gkv.check_for_invalid_columns([], "routes", feed.routes)


def test_check_table():
    feed = sample.copy()
    cond = feed.routes["route_id"].isnull()
    assert not gkv.check_table([], "routes", feed.routes, cond, "Bingo")
    assert gkv.check_table([], "routes", feed.routes, ~cond, "Bongo")


def test_check_column():
    feed = sample.copy()
    assert not gkv.check_column([], "agency", feed.agency, "agency_url", gkv.valid_url)
    feed.agency["agency_url"].iat[0] = "example.com"
    assert gkv.check_column([], "agency", feed.agency, "agency_url", gkv.valid_url)


def test_check_column_id():
    feed = sample.copy()
    assert not gkv.check_column_id([], "routes", feed.routes, "route_id")
    feed.routes["route_id"].iat[0] = np.nan
    assert gkv.check_column_id([], "routes", feed.routes, "route_id")


def test_check_column_linked_id():
    feed = sample.copy()
    assert not gkv.check_column_linked_id(
        [], "trips", feed.trips, "route_id", feed.routes
    )
    feed.trips["route_id"].iat[0] = "Hummus!"
    assert gkv.check_column_linked_id([], "trips", feed.trips, "route_id", feed.routes)


def test_format_problems():
    problems = [("ba", "da", "boom", "boom")]
    assert problems == gkv.format_problems(problems, as_df=False)

    e = gkv.format_problems(problems, as_df=True)
    assert isinstance(e, pd.DataFrame)
    assert e.columns.tolist() == ["type", "message", "table", "rows"]


def test_check_agency():
    assert not gkv.check_agency(sample)

    feed = sample.copy()
    feed.agency = None
    assert gkv.check_agency(feed)

    feed = sample.copy()
    del feed.agency["agency_name"]
    assert gkv.check_agency(feed)

    feed = sample.copy()
    feed.agency["b"] = 3
    assert gkv.check_agency(feed, include_warnings=True)

    feed = sample.copy()
    feed.agency = pd.concat([feed.agency, feed.agency.iloc[0]])
    assert gkv.check_agency(feed)

    feed = sample.copy()
    feed.agency["agency_name"] = ""
    assert gkv.check_agency(feed)

    for col in [
        "agency_timezone",
        "agency_url",
        "agency_fare_url",
        "agency_lang",
        "agency_phone",
        "agency_email",
    ]:
        feed = sample.copy()
        feed.agency[col] = ""
        assert gkv.check_agency(feed)


def test_check_attributions():
    assert not gkv.check_attributions(sample)

    feed = sample.copy()
    del feed.attributions["organization_name"]
    assert gkv.check_attributions(feed)

    feed = sample.copy()
    feed.attributions["route_id"] = "no such route ID"
    # Can't have invalid route_id
    assert gkv.check_attributions(feed)

    feed = sample.copy()
    feed.attributions["trip_id"] = "AB1"
    # Can't have both route_id and trip_id present
    assert gkv.check_attributions(feed)

    feed = sample.copy()
    feed.attributions["is_producer"] = 0
    feed.attributions["is_operator"] = 0
    feed.attributions["is_authority"] = 0
    # Can't have all these be zero
    assert gkv.check_attributions(feed)

    for col in [
        "attribution_url",
        "attribution_email",
    ]:
        feed = sample.copy()
        feed.attributions[col] = ""
        assert gkv.check_attributions(feed)


def test_check_calendar():
    assert not gkv.check_calendar(sample)
    assert gkv.check_calendar(sample, include_warnings=True)  # feed has expired

    feed = sample.copy()
    feed.calendar = None
    assert not gkv.check_calendar(feed)

    feed = sample.copy()
    del feed.calendar["service_id"]
    assert gkv.check_calendar(feed)

    feed = sample.copy()
    feed.calendar["yo"] = 3
    assert not gkv.check_calendar(feed)
    assert gkv.check_calendar(feed, include_warnings=True)

    feed = sample.copy()
    feed.calendar["service_id"].iat[0] = feed.calendar["service_id"].iat[1]
    assert gkv.check_calendar(feed)

    for col in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "start_date",
        "end_date",
    ]:
        feed = sample.copy()
        feed.calendar[col].iat[0] = "5"
        assert gkv.check_calendar(feed)


def test_check_calendar_dates():
    assert not gkv.check_calendar_dates(sample)

    feed = sample.copy()
    feed.calendar_dates = None
    assert not gkv.check_calendar_dates(feed)

    feed = sample.copy()
    del feed.calendar_dates["service_id"]
    assert gkv.check_calendar_dates(feed)

    feed = sample.copy()
    feed.calendar_dates["yo"] = 3
    assert not gkv.check_calendar_dates(feed)
    assert gkv.check_calendar_dates(feed, include_warnings=True)

    for col in ["date", "exception_type"]:
        feed = sample.copy()
        feed.calendar_dates[col].iat[0] = "5"
        assert gkv.check_calendar_dates(feed)


def test_check_fare_attributes():
    assert not gkv.check_fare_attributes(sample)

    feed = sample.copy()
    feed.fare_attributes = None
    assert not gkv.check_fare_attributes(feed)

    feed = sample.copy()
    del feed.fare_attributes["fare_id"]
    assert gkv.check_fare_attributes(feed)

    feed = sample.copy()
    feed.fare_attributes["yo"] = 3
    assert not gkv.check_fare_attributes(feed)
    assert gkv.check_fare_attributes(feed, include_warnings=True)

    feed = sample.copy()
    feed.fare_attributes["currency_type"] = "jubjub"
    assert gkv.check_fare_attributes(feed)

    for col in ["payment_method", "transfers", "transfer_duration"]:
        feed = sample.copy()
        feed.fare_attributes[col] = -7
        assert gkv.check_fare_attributes(feed)


def test_check_fare_rules():
    assert not gkv.check_fare_rules(sample)

    feed = sample.copy()
    feed.fare_rules = None
    assert not gkv.check_fare_rules(feed)

    feed = sample.copy()
    del feed.fare_rules["fare_id"]
    assert gkv.check_fare_rules(feed)

    feed = sample.copy()
    feed.fare_rules["yo"] = 3
    assert not gkv.check_fare_rules(feed)
    assert gkv.check_fare_rules(feed, include_warnings=True)

    for col in [
        "fare_id",
        "route_id",
        "origin_id",
        "destination_id",
        "contains_id",
    ]:
        feed = sample.copy()
        feed.fare_rules[col] = "tuberosity"
        print(col)
        print(feed.fare_rules)
        assert gkv.check_fare_rules(feed)


def test_check_feed_info():
    # Create feed_info table
    feed = sample.copy()
    columns = [
        "feed_publisher_name",
        "feed_publisher_url",
        "feed_lang",
        "feed_start_date",
        "feed_end_date",
        "feed_version",
    ]
    rows = [["slurp", "http://slurp.burp", "aa", "21110101", "21110102", "69"]]
    feed.feed_info = pd.DataFrame(rows, columns=columns)
    assert not gkv.check_feed_info(feed)

    feed1 = feed.copy()
    feed1.feed_info = None
    assert not gkv.check_feed_info(feed1)

    feed1 = feed.copy()
    del feed1.feed_info["feed_lang"]
    assert gkv.check_feed_info(feed1)

    feed1 = feed.copy()
    feed1.feed_info["yo"] = 3
    assert not gkv.check_feed_info(feed1)
    assert gkv.check_feed_info(feed1, include_warnings=True)

    for col in columns:
        feed1 = feed.copy()
        feed1.feed_info[col] = ""
        assert gkv.check_feed_info(feed1)


def test_check_frequencies():
    assert not gkv.check_frequencies(sample)

    feed = sample.copy()
    feed.frequencies = None
    assert not gkv.check_frequencies(feed)

    feed = sample.copy()
    del feed.frequencies["trip_id"]
    assert gkv.check_frequencies(feed)

    feed = sample.copy()
    feed.frequencies["yo"] = 3
    assert not gkv.check_frequencies(feed)
    assert gkv.check_frequencies(feed, include_warnings=True)

    feed = sample.copy()
    feed.frequencies["trip_id"].iat[0] = "ratatat"
    assert gkv.check_frequencies(feed)

    for col in ["start_time", "end_time"]:
        feed = sample.copy()
        feed.frequencies[col] = "07:00:00"
        assert gkv.check_frequencies(feed)

    for col in ["headway_secs", "exact_times"]:
        feed = sample.copy()
        feed.frequencies[col] = -7
        assert gkv.check_frequencies(feed)


def test_check_routes():
    assert not gkv.check_routes(sample)

    feed = sample.copy()
    feed.routes = None
    assert gkv.check_routes(feed)

    feed = sample.copy()
    del feed.routes["route_id"]
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["bingo"] = 3
    assert gkv.check_routes(feed, include_warnings=True)

    feed = sample.copy()
    feed.routes["route_id"].iat[0] = feed.routes["route_id"].iat[1]
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["agency_id"] = "Hubba hubba"
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_short_name"].iat[0] = ""
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_short_name"].iat[0] = ""
    feed.routes["route_long_name"].iat[0] = ""
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_type"].iat[0] = 8
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_color"].iat[0] = "FFF"
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_text_color"].iat[0] = "FFF"
    assert gkv.check_routes(feed)

    feed = sample.copy()
    feed.routes["route_short_name"].iat[1] = feed.routes["route_short_name"].iat[0]
    feed.routes["route_long_name"].iat[1] = feed.routes["route_long_name"].iat[0]
    assert not gkv.check_routes(feed)
    assert gkv.check_routes(feed, include_warnings=True)

    feed = sample.copy()
    feed.routes["route_id"].iat[0] = "Shwing"
    assert not gkv.check_routes(feed)
    assert gkv.check_routes(feed, include_warnings=True)

    feed = sample.copy()
    feed.agency = None
    assert gkv.check_routes(feed)


def test_check_shapes():
    assert not gkv.check_shapes(sample)

    # Make a nonempty shapes table to check
    feed = sample.copy()
    rows = [
        ["1100015", -16.743_632, 145.668_255, 10001, 1.2],
        ["1100015", -16.743_522, 145.668_394, 10002, 1.3],
    ]
    columns = [
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
        "shape_dist_traveled",
    ]
    feed.shapes = pd.DataFrame(rows, columns=columns)
    assert not gkv.check_shapes(feed)

    feed1 = feed.copy()
    del feed1.shapes["shape_id"]
    assert gkv.check_shapes(feed1)

    feed1 = feed.copy()
    feed1.shapes["yo"] = 3
    assert not gkv.check_shapes(feed1)
    assert gkv.check_shapes(feed1, include_warnings=True)

    feed1 = feed.copy()
    feed1.shapes["shape_id"].iat[0] = ""
    assert gkv.check_shapes(feed1)

    for column in ["shape_pt_lon", "shape_pt_lat"]:
        feed1 = feed.copy()
        feed1.shapes[column] = 185
        assert gkv.check_shapes(feed1)

    feed1 = feed.copy()
    feed1.shapes["shape_pt_sequence"].iat[1] = feed1.shapes["shape_pt_sequence"].iat[0]
    assert gkv.check_shapes(feed1)

    feed1 = feed.copy()
    feed1.shapes["shape_dist_traveled"].iat[1] = 0
    assert gkv.check_shapes(feed1)

    feed1 = feed.copy()
    t1 = feed1.shapes.iloc[0].copy()
    t2 = feed1.shapes.iloc[1].copy()
    feed1.shapes.iloc[0] = t2
    feed1.shapes.iloc[1] = t1
    assert not gkv.check_shapes(feed1)


def test_check_stops():
    assert not gkv.check_stops(sample)

    feed = sample.copy()
    feed.stops = None
    assert gkv.check_stops(feed)

    feed = sample.copy()
    del feed.stops["stop_id"]
    assert gkv.check_stops(feed)

    feed = sample.copy()
    feed.stops["b"] = 3
    assert gkv.check_stops(feed, include_warnings=True)

    feed = sample.copy()
    feed.stops["stop_id"].iat[0] = feed.stops["stop_id"].iat[1]
    assert gkv.check_stops(feed)

    for column in ["stop_code", "stop_desc", "zone_id", "parent_station"]:
        feed = sample.copy()
        feed.stops[column] = ""
        assert gkv.check_stops(feed)

    for column in ["stop_url", "stop_timezone"]:
        feed = sample.copy()
        feed.stops[column] = "Wa wa"
        assert gkv.check_stops(feed)

    for column in [
        "stop_lon",
        "stop_lat",
        "location_type",
        "wheelchair_boarding",
    ]:
        feed = sample.copy()
        feed.stops[column] = 185
        assert gkv.check_stops(feed)

    feed = sample.copy()
    feed.stops["parent_station"] = "bingo"
    assert gkv.check_stops(feed)

    feed = sample.copy()
    feed.stops["location_type"] = 1
    feed.stops["parent_station"] = "bingo"
    assert gkv.check_stops(feed)

    feed = sample.copy()
    feed.stops["location_type"] = 0
    feed.stops["parent_station"] = feed.stops["stop_id"].iat[1]
    assert gkv.check_stops(feed)

    feed = sample.copy()
    # valid location type
    feed.stops["location_type"] = 2
    assert not gkv.check_stops(feed)
    # requires a location
    feed.stops["stop_lat"] = np.nan
    assert gkv.check_stops(feed)
    # valid location_type, does not require location
    feed.stops["location_type"] = 3
    assert not gkv.check_stops(feed)
    # valid location_type, does not require location
    feed.stops["location_type"] = 4
    assert not gkv.check_stops(feed)
    # location type 4 requires a parent station
    feed.stops["parent_station"] = np.nan
    assert gkv.check_stops(feed)
    # valid parent station for location type 4
    feed.stops["stop_lat"] = 0.0
    feed.stops["parent_station"] = feed.stops["stop_id"].iat[1]
    feed.stops["parent_station"].iat[1] = np.nan
    feed.stops["location_type"].iat[1] = 1
    assert not gkv.check_stops(feed)

    feed = sample.copy()
    feed.stops["stop_id"].iat[0] = "Flippity flew"
    assert not gkv.check_stops(feed)
    assert gkv.check_stops(feed, include_warnings=True)

    feed = sample.copy()
    feed.stop_times = None
    assert not gkv.check_stops(feed)
    assert gkv.check_stops(feed, include_warnings=True)


def test_check_stop_times():
    assert not gkv.check_stop_times(sample)

    feed = sample.copy()
    feed.stop_times = None
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    del feed.stop_times["stop_id"]
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["b"] = 3
    assert gkv.check_stop_times(feed, include_warnings=True)

    feed = sample.copy()
    feed.stop_times["trip_id"].iat[0] = "bingo"
    assert gkv.check_stop_times(feed)

    for col in ["arrival_time", "departure_time"]:
        feed = sample.copy()
        feed.stop_times[col].iat[0] = "1:0:00"
        assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["arrival_time"].iat[-1] = np.nan
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["stop_id"].iat[0] = "bingo"
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["stop_headsign"].iat[0] = ""
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["stop_sequence"].iat[1] = feed.stop_times["stop_sequence"].iat[0]
    assert gkv.check_stop_times(feed)

    for col in ["pickup_type", "drop_off_type"]:
        feed = sample.copy()
        feed.stop_times[col] = "bongo"
        assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["shape_dist_traveled"] = 1
    feed.stop_times["shape_dist_traveled"].iat[1] = 0.9
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["timepoint"] = 3
    assert gkv.check_stop_times(feed)

    feed = sample.copy()
    feed.stop_times["departure_time"].iat[1] = feed.stop_times["departure_time"].iat[0]
    assert not gkv.check_stop_times(feed)
    assert gkv.check_stop_times(feed, include_warnings=True)

    # Return the correct index of the missing time
    feed = sample.copy()
    # Index 2 is the first stop of the second trip
    # Index 2 is the last stop of the second trip
    feed.stop_times["arrival_time"].iat[6] = np.nan
    t1, t2 = feed.stop_times.iloc[2].copy(), feed.stop_times.iloc[6].copy()
    feed.stop_times.iloc[2], feed.stop_times.iloc[6] = t2, t1
    assert gkv.check_stop_times(feed)[0][3][0] == 2

    # Check for last stop of last trip
    # Trips are ordered by trip_id so the STBA trip_id from the sample feed
    # is last and its last stop (index 1) is the last row
    feed = sample.copy()
    feed.stop_times["arrival_time"].iat[1] = np.nan
    assert gkv.check_stop_times(feed)


def test_check_transfers():
    assert not gkv.check_transfers(sample)

    # Create transfers table
    feed = sample.copy()
    columns = [
        "from_stop_id",
        "to_stop_id",
        "transfer_type",
        "min_transfer_time",
    ]
    rows = [[feed.stops["stop_id"].iat[0], feed.stops["stop_id"].iat[1], 2, 3600]]
    feed.transfers = pd.DataFrame(rows, columns=columns)
    assert not gkv.check_transfers(feed)

    feed1 = feed.copy()
    del feed1.transfers["from_stop_id"]
    assert gkv.check_transfers(feed1)

    feed1 = feed.copy()
    feed1.transfers["yo"] = 3
    assert not gkv.check_transfers(feed1)
    assert gkv.check_transfers(feed1, include_warnings=True)

    for col in set(columns) - set(["transfer_type", "min_transfer_time"]):
        feed1 = feed.copy()
        feed1.transfers[col].iat[0] = ""
        assert gkv.check_transfers(feed1)

    for col in ["transfer_type", "min_transfer_time"]:
        feed1 = feed.copy()
        feed1.transfers[col] = -7
        assert gkv.check_transfers(feed1)


def test_check_trips():
    assert not gkv.check_trips(sample)

    feed = sample.copy()
    feed.trips = None
    assert gkv.check_trips(feed)

    feed = sample.copy()
    del feed.trips["trip_id"]
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["b"] = 3
    assert gkv.check_trips(feed, include_warnings=True)

    feed = sample.copy()
    feed.trips["trip_id"].iat[0] = feed.trips["trip_id"].iat[1]
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["route_id"] = "Hubba hubba"
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["service_id"] = "Boom boom"
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["direction_id"].iat[0] = 7
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["block_id"].iat[0] = ""
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["shape_id"].iat[0] = "Hello"
    assert gkv.check_trips(feed)

    feed = sample.copy()
    feed.trips["wheelchair_accessible"] = ""
    assert gkv.check_trips(feed)

    feed = sample.copy()
    tid = feed.trips["trip_id"].iat[0]
    feed.stop_times = feed.stop_times[feed.stop_times["trip_id"] != tid].copy()
    assert not gkv.check_trips(feed)
    assert gkv.check_trips(feed, include_warnings=True)

    feed = sample.copy()
    feed.stop_times = None
    assert not gkv.check_trips(feed)
    assert gkv.check_trips(feed, include_warnings=True)


def test_validate():
    assert not gkv.validate(sample, as_df=False, include_warnings=False)
