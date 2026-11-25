import pytest
import geopandas as gpd
from shapely.geometry import LineString
import benefits_map_app


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Stop @st.cache_data from leaking results between tests."""
    app.load_geodataframe.clear()
    yield


@pytest.fixture
def bng_csv_bytes():
    """Small CSV in EPSG:27700 around Bristol."""
    csv = (
        'id,Coords,Flow DM,Flow DS,Total Cost DIFF\n'
        '1,"LINESTRING (358000 173000, 359000 174000)",100,120,15000\n'
        '2,"LINESTRING (359000 174000, 360000 175000)",200,150,-25000\n'
        '3,"LINESTRING (360000 175000, 361000 176000)",0,50,500\n'
    )
    return csv.encode("utf-8")


@pytest.fixture
def small_gdf():
    return gpd.GeoDataFrame(
        {
            "__rowid__": [0, 1, 2],
            "Total Cost DIFF": [15000.0, -25000.0, 500.0],
            "pct_change": [20.0, -25.0, 150.0],
        },
        geometry=[
            LineString([(-2.60, 51.45), (-2.59, 51.46)]),
            LineString([(-2.59, 51.46), (-2.58, 51.47)]),
            LineString([(-2.58, 51.47), (-2.57, 51.48)]),
        ],
        crs="EPSG:4326",
    )
