import math
import pytest
from shapely.geometry import LineString
from app import parse_linestring, geojson_to_shapely, _geom_to_latlon_coords


@pytest.mark.parametrize("value", [
    "LINESTRING (0 0, 1 1)",
    '{"type": "LineString", "coordinates": [[0, 0], [1, 1]]}',
    "[(0, 0), (1, 1)]",
    "0 0 1 1",
])
def test_parse_linestring_formats(value):
    geom = parse_linestring(value)
    assert isinstance(geom, LineString)
    assert list(geom.coords) == [(0, 0), (1, 1)]


def test_parse_linestring_passthrough():
    line = LineString([(0, 0), (1, 1)])
    assert parse_linestring(line) is line


@pytest.mark.parametrize("value", [None, float("nan"), "", "   ", "1 2 3", "abc"])
def test_parse_linestring_invalid_returns_none(value):
    assert parse_linestring(value) is None


def test_parse_linestring_negative_decimals():
    geom = parse_linestring("-2.5 51.4, -2.6 51.5")
    assert list(geom.coords) == [(-2.5, 51.4), (-2.6, 51.5)]


def test_geojson_to_shapely_polygon():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    assert geojson_to_shapely(poly).geom_type == "Polygon"


@pytest.mark.parametrize("value", [None, {}, {"type": "Point", "coordinates": [0, 0]}])
def test_geojson_to_shapely_rejects_non_polygons(value):
    assert geojson_to_shapely(value) is None


def test_latlon_swap():
    line = LineString([(-2.6, 51.4), (-2.5, 51.5)])
    assert _geom_to_latlon_coords(line) == [(51.4, -2.6), (51.5, -2.5)]
    assert _geom_to_latlon_coords(None) is None
