import pytest
from app import load_geodataframe


def test_load_converts_bng_to_wgs84(bng_csv_bytes):
    gdf = load_geodataframe(bng_csv_bytes, "test.csv", "Coords", "EPSG:27700")
    assert gdf.crs.to_string() == "EPSG:4326"
    assert len(gdf) == 3
    minx, miny, maxx, maxy = gdf.total_bounds
    assert -2.7 < minx < -2.4 and 51.3 < miny < 51.6    # Bristol-ish
    assert list(gdf["__rowid__"]) == [0, 1, 2]


def test_load_missing_geometry_column(bng_csv_bytes):
    with pytest.raises(ValueError, match="not found"):
        load_geodataframe(bng_csv_bytes, "test.csv", "WrongCol", "EPSG:27700")


def test_load_no_valid_geometries():
    csv = b"Coords,v\nabc,1\n,2\n"
    with pytest.raises(ValueError, match="No valid LineString"):
        load_geodataframe(csv, "bad.csv", "Coords", "EPSG:27700")


def test_load_drops_invalid_rows_keeps_rowid():
    csv = b'Coords,v\n"LINESTRING (358000 173000, 359000 174000)",1\nabc,2\n'
    gdf = load_geodataframe(csv, "mixed.csv", "Coords", "EPSG:27700")
    assert len(gdf) == 1


def test_load_bad_crs_raises():
    csv = b'Coords\n"LINESTRING (0 0, 1 1)"\n'
    with pytest.raises(Exception):
        load_geodataframe(csv, "x.csv", "Coords", "EPSG:999999")
