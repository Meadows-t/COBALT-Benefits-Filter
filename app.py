"""
COBALT Benefits Analysis Dashboard.

A three-tab Streamlit application for exploring COBALT transport appraisal
outputs spatially.

Inputs
------
- Flow CSV: link geometry plus Do-Minimum (DM) and Do-Something (DS) flows.
- Benefits CSV: link geometry plus a signed economic benefits column.

Both files must share the same geometry column name and source CRS.

Outputs
-------
- Interactive maps and summary metrics within the Streamlit app.
- Standalone HTML reports (downloaded from the Flow and Benefits tabs).

Assumptions and limitations
---------------------------
- Geometry is LineString only (WKT, GeoJSON string or flat coordinate list).
- Percentage change is undefined (NaN) where the DM flow is zero.
- Overlap offsetting only separates *exactly* identical geometries.
- HTML reports load basemap tiles and Leaflet from the internet when opened.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from io import BytesIO
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw, PolyLineOffset
from jinja2 import Environment
from shapely import wkt
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry
from streamlit_folium import st_folium

__version__ = "1.1.0"

logger = logging.getLogger(__name__)

StyleFunction = Callable[[pd.Series], dict[str, Any]]


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input defaults
DEFAULT_GEOM_COL = "Coords"
DEFAULT_SOURCE_CRS = "EPSG:27700"
DEFAULT_TARGET_CRS = "EPSG:4326"
DEFAULT_BENEFITS_COL = "Total Cost DIFF"
DEFAULT_BENEFITS_THRESHOLD = 10000.0
DEFAULT_DM_COL = "Flow DM"
DEFAULT_DS_COL = "Flow DS"
DEFAULT_PCT_THRESHOLD = 10

# Map appearance
DEFAULT_MAP_CENTRE = (51.4545, -2.5879)  # Bristol, used when there is no data
DEFAULT_ZOOM = 9
REPORT_ZOOM = 10
BASEMAP_TILES = "CartoDB positron"
LINE_WEIGHT = 3
NO_DATA_COLOUR = "#888888"
NO_DATA_OPACITY = 0.3
DEFAULT_OFFSET_STEP_PX = 6
PCT_CLIP_LIMIT = 100.0

COLORMAP_DIVERGING = ["#d7191c", "#ffffbf", "#2c7bb6"]
COLORMAP_BENEFITS = ["#2c7bb6", "#ffffbf", "#d7191c"]

MIN_OPACITY = 0.1
MAX_OPACITY = 1.0
FALLBACK_OPACITY = 0.25
OPACITY_GAMMA = 0.9

# Table limits
APP_TABLE_ROWS = 200
SELECTED_TABLE_ROWS = 500
REPORT_TABLE_ROWS = 100

# Internal column names
ROW_ID_COL = "__rowid__"
PCT_CHANGE_COL = "pct_change"
ABS_PCT_CHANGE_COL = "abs_pct_change"
OFFSET_COL = "_offset_px"

LOGO_PATH = "assets/Red Logo.png"


# =============================================================================
# GEOMETRY PARSING
# =============================================================================


def parse_linestring(value: Any) -> LineString | None:
    """
    Parse a LineString from WKT, a GeoJSON string or a coordinate list.

    Parameters
    ----------
    value : Any
        Raw cell value. May be a ``LineString``, WKT text
        (``"LINESTRING (0 0, 1 1)"``), a GeoJSON string or any text containing
        an even number (>= 4) of numbers, read as ``x1 y1 x2 y2 ...``.

    Returns
    -------
    LineString or None
        Parsed geometry, or ``None`` if the value is empty, unparseable or
        not a LineString.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, LineString):
        return value

    string_value = str(value).strip()
    if not string_value:
        return None

    # WKT
    try:
        geom = wkt.loads(string_value)
        return geom if isinstance(geom, LineString) else None
    except Exception:  # noqa: BLE001 - fall through to the next format
        pass

    # GeoJSON string
    if string_value.startswith("{") and '"coordinates"' in string_value:
        try:
            geom = shape(json.loads(string_value))
            return geom if isinstance(geom, LineString) else None
        except Exception:  # noqa: BLE001
            pass

    # Flat coordinate list
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string_value)
    if len(numbers) >= 4 and len(numbers) % 2 == 0:
        coords = [(float(numbers[i]), float(numbers[i + 1])) for i in range(0, len(numbers), 2)]
        return LineString(coords)

    return None


def geojson_to_shapely(geojson_dict: dict[str, Any] | None) -> BaseGeometry | None:
    """
    Convert a GeoJSON Polygon or MultiPolygon dictionary to a shapely geometry.

    Parameters
    ----------
    geojson_dict : dict or None
        GeoJSON geometry, e.g. from a ``st_folium`` drawing.

    Returns
    -------
    shapely geometry or None
        Polygon/MultiPolygon, or ``None`` for any other type or invalid input.
    """
    if not geojson_dict:
        return None
    if geojson_dict.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    try:
        return shape(geojson_dict)
    except Exception:  # noqa: BLE001
        logger.warning("Could not convert drawn polygon to geometry.")
        return None


def _geom_to_latlon_coords(geom: BaseGeometry | None) -> list[tuple[float, float]] | None:
    """
    Convert a LineString in EPSG:4326 to Folium ``[(lat, lon), ...]`` order.

    Parameters
    ----------
    geom : shapely geometry or None
        Geometry with ``(x=lon, y=lat)`` coordinates.

    Returns
    -------
    list of tuple or None
        Coordinates as ``(lat, lon)``, or ``None`` if they cannot be read.
    """
    if geom is None:
        return None
    try:
        return [(y, x) for x, y in geom.coords]
    except Exception:  # noqa: BLE001
        return None


# =============================================================================
# CALCULATIONS (pure functions - unit tested)
# =============================================================================


def calculate_pct_change(df: pd.DataFrame, dm_col: str, ds_col: str) -> pd.DataFrame:
    """
    Add percentage change in flow between DM and DS scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Not modified.
    dm_col : str
        Do-Minimum flow column.
    ds_col : str
        Do-Something flow column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with numeric DM/DS columns plus ``pct_change`` and
        ``abs_pct_change``. Rows where DM is zero or non-numeric are NaN.
    """
    out = df.copy()
    out[dm_col] = pd.to_numeric(out[dm_col], errors="coerce")
    out[ds_col] = pd.to_numeric(out[ds_col], errors="coerce")
    dm_safe = out[dm_col].replace(0, np.nan)
    out[PCT_CHANGE_COL] = ((out[ds_col] - out[dm_col]) / dm_safe * 100).astype(float)
    out[ABS_PCT_CHANGE_COL] = out[PCT_CHANGE_COL].abs()
    return out


def filter_by_abs_threshold(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """
    Keep rows whose absolute value in ``column`` meets a threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    column : str
        Column to test.
    threshold : float
        Inclusive minimum absolute value.

    Returns
    -------
    pandas.DataFrame
        Copy of the rows where ``|column| >= threshold``. NaN and non-numeric
        values are dropped.
    """
    abs_vals = pd.to_numeric(df[column], errors="coerce").abs()
    return df[abs_vals.notna() & (abs_vals >= threshold)].copy()


def compute_offsets(
    geoms: Iterable[BaseGeometry | None],
    step_px: float = DEFAULT_OFFSET_STEP_PX,
    base_px: float = 0,
    offset_overlaps: bool = True,
) -> pd.Series:
    """
    Calculate pixel offsets so identical geometries are drawn side by side.

    Parameters
    ----------
    geoms : iterable of shapely geometry
        Geometries in display order.
    step_px : float, default 6
        Spacing between overlapping lines, in pixels.
    base_px : float, default 0
        Offset applied to every line.
    offset_overlaps : bool, default True
        If False, every line gets ``base_px``.

    Returns
    -------
    pandas.Series
        Float offsets with a fresh 0..n-1 index. Duplicates are spread
        symmetrically: ``base_px + (i - (n - 1) / 2) * step_px``.
    """
    keys = pd.Series([getattr(g, "wkb_hex", None) for g in geoms])

    if not offset_overlaps or step_px == 0:
        return pd.Series([float(base_px)] * len(keys))

    idx = keys.groupby(keys).cumcount()
    n = keys.groupby(keys).transform("size")
    return (base_px + (idx - (n - 1) / 2.0) * step_px).astype(float)


def calculate_opacity(value: float | None, max_abs_value: float) -> float:
    """
    Scale line opacity by the magnitude of a value.

    Parameters
    ----------
    value : float or None
        Value to scale.
    max_abs_value : float
        Largest absolute value in the dataset.

    Returns
    -------
    float
        Opacity between ``MIN_OPACITY`` and ``MAX_OPACITY`` using a gamma
        curve, or ``FALLBACK_OPACITY`` for missing values or a zero maximum.
    """
    if value is None or pd.isna(value) or max_abs_value == 0:
        return FALLBACK_OPACITY
    normalised = min(abs(value) / max_abs_value, 1.0) ** OPACITY_GAMMA
    return MIN_OPACITY + normalised * (MAX_OPACITY - MIN_OPACITY)


def _to_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` if it is missing or invalid."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data(show_spinner=True)
def load_geodataframe(
    file_bytes: bytes, filename: str, geom_column: str, source_crs: str
) -> gpd.GeoDataFrame:
    """
    Load a CSV with a geometry column and reproject it to EPSG:4326.

    Parameters
    ----------
    file_bytes : bytes
        Raw CSV content.
    filename : str
        File name, used in error messages.
    geom_column : str
        Column holding LineString geometry.
    source_crs : str
        CRS of the input coordinates, e.g. ``"EPSG:27700"``.

    Returns
    -------
    geopandas.GeoDataFrame
        Valid rows only, with ``__rowid__`` holding the original row index.

    Raises
    ------
    ValueError
        If the geometry column is missing, no geometry is valid, or the CRS
        conversion fails.
    """
    df = pd.read_csv(BytesIO(file_bytes))

    if geom_column not in df.columns:
        raise ValueError(
            f"Geometry column '{geom_column}' not found in '{filename}'.\n"
            f"Available columns: {', '.join(df.columns)}"
        )

    df["geometry"] = df[geom_column].apply(parse_linestring)
    n_invalid = int(df["geometry"].isna().sum())
    df = df[df["geometry"].notna()].copy()

    if len(df) == 0:
        raise ValueError(f"No valid LineString geometries found in column '{geom_column}'")

    df[ROW_ID_COL] = df.index.astype(int)

    try:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=source_crs)
        gdf = gdf.to_crs(DEFAULT_TARGET_CRS)
    except Exception as exc:
        raise ValueError(
            f"CRS conversion failed from {source_crs} to {DEFAULT_TARGET_CRS}.\n"
            f"Check your source CRS setting. Error: {exc}"
        ) from exc

    logger.info(
        "Loaded %s: %d valid features, %d invalid geometries dropped (CRS %s).",
        filename,
        len(gdf),
        n_invalid,
        source_crs,
    )
    return gdf


# =============================================================================
# MAP STYLING
# =============================================================================


def make_flow_style(colormap: cm.LinearColormap, value_column: str) -> StyleFunction:
    """
    Build a row style function for percentage-change flow maps.

    Parameters
    ----------
    colormap : branca.colormap.LinearColormap
        Diverging colormap spanning +/- ``PCT_CLIP_LIMIT``.
    value_column : str
        Column holding percentage change.

    Returns
    -------
    callable
        Function mapping a row to a Folium style dictionary.
    """

    def style_row(row: pd.Series) -> dict[str, Any]:
        value = _to_float(row.get(value_column))
        if value is None:
            return {"color": NO_DATA_COLOUR, "weight": LINE_WEIGHT, "opacity": NO_DATA_OPACITY}
        clipped = max(-PCT_CLIP_LIMIT, min(PCT_CLIP_LIMIT, value))
        return {"color": colormap(clipped), "weight": LINE_WEIGHT, "opacity": MAX_OPACITY}

    return style_row


def make_benefits_style(
    colormap: cm.LinearColormap, value_column: str, max_abs: float
) -> StyleFunction:
    """
    Build a row style function for signed benefits maps.

    Parameters
    ----------
    colormap : branca.colormap.LinearColormap
        Colormap spanning the data range.
    value_column : str
        Column holding benefits.
    max_abs : float
        Largest absolute benefit, used to scale opacity.

    Returns
    -------
    callable
        Function mapping a row to a Folium style dictionary.
    """

    def style_row(row: pd.Series) -> dict[str, Any]:
        value = _to_float(row.get(value_column))
        colour = colormap(value) if value is not None else NO_DATA_COLOUR
        opacity = calculate_opacity(value, max_abs)
        return {"color": colour, "weight": LINE_WEIGHT, "opacity": float(opacity)}

    return style_row


def build_flow_colormap(value_column: str) -> cm.LinearColormap:
    """Return the clipped diverging colormap used for flow change."""
    colormap = cm.LinearColormap(
        colors=COLORMAP_DIVERGING, vmin=-PCT_CLIP_LIMIT, vmax=PCT_CLIP_LIMIT
    )
    colormap.caption = f"{value_column} (% clipped to ±{PCT_CLIP_LIMIT:.0f})"
    return colormap


def build_benefits_colormap(
    values: pd.Series, value_column: str
) -> tuple[cm.LinearColormap, float]:
    """
    Return the benefits colormap and the maximum absolute value.

    Parameters
    ----------
    values : pandas.Series
        Benefit values (coerced to numeric).
    value_column : str
        Column name, used in the legend caption.

    Returns
    -------
    tuple
        ``(colormap, max_abs)``. ``max_abs`` is never zero.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    has_data = bool(numeric.notna().any())
    vmin = float(numeric.min()) if has_data else 0.0
    vmax = float(numeric.max()) if has_data else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    colormap = cm.LinearColormap(colors=COLORMAP_BENEFITS, vmin=vmin, vmax=vmax)
    colormap.caption = f"{value_column} (Signed benefits)"

    max_abs = float(numeric.abs().max()) if has_data else 1.0
    return colormap, (max_abs or 1.0)


# =============================================================================
# MAPPING - INTERACTIVE (FOR STREAMLIT DISPLAY)
# =============================================================================


def _base_map(gdf: gpd.GeoDataFrame, zoom: int = DEFAULT_ZOOM) -> folium.Map:
    """Return a Folium map centred on ``gdf`` (or the default centre if empty)."""
    if len(gdf) == 0:
        return folium.Map(location=DEFAULT_MAP_CENTRE, zoom_start=zoom, tiles=BASEMAP_TILES)
    minx, miny, maxx, maxy = gdf.total_bounds
    centre = [(miny + maxy) / 2, (minx + maxx) / 2]
    return folium.Map(location=centre, zoom_start=zoom, tiles=BASEMAP_TILES)


def add_polygon_draw(folium_map: folium.Map) -> None:
    """
    Add a polygon-only drawing control to a map.

    Parameters
    ----------
    folium_map : folium.Map
        Map to modify in place.
    """
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(folium_map)


def _add_offset_lines_layer(
    folium_map: folium.Map,
    gdf_lines: gpd.GeoDataFrame,
    value_column: str,
    style_row_fn: StyleFunction,
    layer_name: str,
    tooltip_alias: str,
    offset_overlaps: bool = True,
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX,
    base_offset_px: float = 0,
) -> None:
    """
    Add LineStrings to a map as a layer of offset polylines.

    Parameters
    ----------
    folium_map : folium.Map
        Map to modify in place.
    gdf_lines : geopandas.GeoDataFrame
        Lines in EPSG:4326.
    value_column : str
        Column shown in the tooltip.
    style_row_fn : callable
        Function returning a style dictionary for each row.
    layer_name : str
        Name shown in the layer control.
    tooltip_alias : str
        Label for the value in the tooltip.
    offset_overlaps : bool, default True
        Spread identical geometries side by side.
    offset_step_px : float, default 6
        Spacing between overlapping lines, in pixels.
    base_offset_px : float, default 0
        Offset applied to every line.
    """
    feature_group = folium.FeatureGroup(name=layer_name, show=True)

    if len(gdf_lines) > 0:
        df = gdf_lines.copy()
        df[OFFSET_COL] = compute_offsets(
            df.geometry,
            step_px=offset_step_px,
            base_px=base_offset_px,
            offset_overlaps=offset_overlaps,
        ).values

        for _, row in df.iterrows():
            coords = _geom_to_latlon_coords(row.geometry)
            if not coords or len(coords) < 2:
                continue

            style = style_row_fn(row)
            tooltip_html = (
                f"Row ID: {html.escape(str(row.get(ROW_ID_COL, '')))}<br>"
                f"{html.escape(tooltip_alias)}: "
                f"{html.escape(str(row.get(value_column, '')))}"
            )

            PolyLineOffset(
                locations=coords,
                color=style.get("color", NO_DATA_COLOUR),
                weight=style.get("weight", LINE_WEIGHT),
                opacity=style.get("opacity", 0.8),
                offset=float(row[OFFSET_COL]),
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
            ).add_to(feature_group)

    feature_group.add_to(folium_map)


def create_benefits_map(gdf_display: gpd.GeoDataFrame, value_column: str) -> folium.Map:
    """
    Create the interactive benefits map with a polygon drawing tool.

    Parameters
    ----------
    gdf_display : geopandas.GeoDataFrame
        Links to display, in EPSG:4326.
    value_column : str
        Signed benefits column.

    Returns
    -------
    folium.Map
        Map with benefits layer, legend and layer control. Drawing tools are
        added by the caller where needed.
    """
    folium_map = _base_map(gdf_display)
    if len(gdf_display) == 0:
        return folium_map

    colormap, max_abs = build_benefits_colormap(gdf_display[value_column], value_column)
    colormap.add_to(folium_map)

    _add_offset_lines_layer(
        folium_map=folium_map,
        gdf_lines=gdf_display,
        value_column=value_column,
        style_row_fn=make_benefits_style(colormap, value_column, max_abs),
        layer_name="Benefits",
        tooltip_alias=value_column,
    )

    folium.LayerControl().add_to(folium_map)
    return folium_map


def create_flow_map(
    gdf: gpd.GeoDataFrame,
    pct_change_column: str,
    offset_overlaps: bool = True,
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX,
    base_offset_px: float = 0,
) -> folium.Map:
    """
    Create the flow comparison map with a clipped diverging colour scale.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Links to display, in EPSG:4326.
    pct_change_column : str
        Column holding percentage change.
    offset_overlaps : bool, default True
        Spread identical geometries side by side.
    offset_step_px : float, default 6
        Spacing between overlapping lines, in pixels.
    base_offset_px : float, default 0
        Offset applied to every line.

    Returns
    -------
    folium.Map
        Map with flow change layer, legend and layer control.
    """
    folium_map = _base_map(gdf)
    if len(gdf) == 0:
        return folium_map

    colormap = build_flow_colormap(pct_change_column)
    colormap.add_to(folium_map)

    _add_offset_lines_layer(
        folium_map=folium_map,
        gdf_lines=gdf,
        value_column=pct_change_column,
        style_row_fn=make_flow_style(colormap, pct_change_column),
        layer_name="Flow change",
        tooltip_alias="% Change",
        offset_overlaps=offset_overlaps,
        offset_step_px=offset_step_px,
        base_offset_px=base_offset_px,
    )

    folium.LayerControl().add_to(folium_map)
    return folium_map


# =============================================================================
# REPORT GENERATION
# =============================================================================


def create_report_map(
    gdf: gpd.GeoDataFrame, value_column: str, map_title: str, is_flow_map: bool = False
) -> str:
    """
    Create map HTML for the exported report (no drawing tools).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Links to display, in EPSG:4326.
    value_column : str
        Column used for colouring.
    map_title : str
        Layer name.
    is_flow_map : bool, default False
        Use flow (percentage change) styling instead of benefits styling.

    Returns
    -------
    str
        Map HTML, or a "No data" message if ``gdf`` is empty.
    """
    if len(gdf) == 0:
        return "<p style='text-align:center; padding:40px; color:#888;'>No data to display</p>"

    folium_map = _base_map(gdf, zoom=REPORT_ZOOM)
    minx, miny, maxx, maxy = gdf.total_bounds
    folium_map.fit_bounds([[miny, minx], [maxy, maxx]])

    if is_flow_map:
        colormap = build_flow_colormap(value_column)
        style_row = make_flow_style(colormap, value_column)
        tooltip_alias = "% Change"
    else:
        colormap, max_abs = build_benefits_colormap(gdf[value_column], value_column)
        style_row = make_benefits_style(colormap, value_column, max_abs)
        tooltip_alias = value_column

    colormap.add_to(folium_map)

    _add_offset_lines_layer(
        folium_map=folium_map,
        gdf_lines=gdf,
        value_column=value_column,
        style_row_fn=style_row,
        layer_name=map_title,
        tooltip_alias=tooltip_alias,
    )

    folium.LayerControl().add_to(folium_map)
    return folium_map._repr_html_()


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COBALT Analysis Report - {{ timestamp }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                Oxygen, Ubuntu, sans-serif;
            line-height: 1.6; max-width: 1400px; margin: 0 auto; padding: 20px;
            background: #f5f7fa; color: #1f2937;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0 0 10px 0; font-size: 2.5em; }
        .header p { margin: 5px 0; opacity: 0.9; }
        .section {
            background: white; padding: 30px; margin-bottom: 25px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: #667eea; border-bottom: 3px solid #667eea;
            padding-bottom: 10px; margin-top: 0;
        }
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
            padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;
        }
        .metric-label {
            font-size: 0.85em; color: #6b7280; margin-bottom: 5px;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #1f2937; }
        .map-container {
            margin: 25px 0; border-radius: 8px; overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .map-container iframe { width: 100%; height: 600px; border: none; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }
        th {
            background: #667eea; color: white; padding: 12px;
            text-align: left; font-weight: 600;
        }
        td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }
        tr:hover, tr:nth-child(even) { background: #f9fafb; }
        .config-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px; background: #f9fafb; padding: 20px; border-radius: 8px;
            margin: 20px 0;
        }
        .config-item {
            display: flex; justify-content: space-between; padding: 10px;
            background: white; border-radius: 5px; border-left: 3px solid #10b981;
        }
        .config-label { font-weight: 600; color: #6b7280; }
        .config-value { color: #1f2937; font-family: 'Courier New', monospace; }
        .footer {
            text-align: center; padding: 20px; color: #6b7280; font-size: 0.9em;
            margin-top: 40px; border-top: 2px solid #e5e7eb;
        }
        .info-box, .success-box {
            padding: 15px; margin: 15px 0; border-radius: 5px;
        }
        .info-box { background: #dbeafe; border-left: 4px solid #3b82f6; }
        .success-box { background: #d1fae5; border-left: 4px solid #10b981; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚦 COBALT Benefits Analysis Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Analysis Type:</strong> {{ analysis_type }}</p>
        <p><strong>Tool version:</strong> {{ version }}</p>
    </div>

    <div class="section">
        <h2>⚙️ Analysis Configuration</h2>
        <div class="config-grid">
            {% for key, value in config.items() %}
            <div class="config-item">
                <span class="config-label">{{ key }}</span>
                <span class="config-value">{{ value }}</span>
            </div>
            {% endfor %}
        </div>
    </div>

    {% if metrics %}
    <div class="section">
        <h2>📊 Key Metrics</h2>
        <div class="metrics-grid">
            {% for metric in metrics %}
            <div class="metric-card">
                <div class="metric-label">{{ metric.label }}</div>
                <div class="metric-value">{{ metric.value }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if map_html %}
    <div class="section">
        <h2>🗺️ Interactive Map</h2>
        {% if map_caption %}<div class="info-box">{{ map_caption }}</div>{% endif %}
        <div class="map-container">
            <iframe srcdoc="{{ map_html | e }}"></iframe>
        </div>
        <p style="color: #6b7280; font-size: 0.9em; margin-top: 10px;">
            <strong>Map Controls:</strong> Use mouse wheel to zoom, click and drag to
            pan. Click layer icon (top right) to toggle layers. Hover over features
            to see details.
        </p>
    </div>
    {% endif %}

    {% if selection_info %}
    <div class="section">
        <h2>✅ Polygon Selection Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Selected Features</div>
                <div class="metric-value">{{ selection_info.count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Selected Benefits</div>
                <div class="metric-value">{{ selection_info.total }}</div>
            </div>
        </div>
        {% if selection_info.polygon_source %}
        <div class="success-box">
            <strong>🎯 Selection source:</strong> {{ selection_info.polygon_source }}
        </div>
        {% endif %}
    </div>
    {% endif %}

    {% if table_data %}
    <div class="section">
        <h2>📋 Data Summary (Top Results)</h2>
        <div style="overflow-x: auto;">{{ table_data | safe }}</div>
        {% if table_note %}
        <p style="color: #6b7280; font-size: 0.9em; margin-top: 10px;">{{ table_note }}</p>
        {% endif %}
    </div>
    {% endif %}

    <div class="footer">
        <p><strong>COBALT Benefits Analysis Dashboard</strong> v{{ version }}</p>
        <p>Built on UK Department for Transport COBALT outputs (TAG appraisal)</p>
        <p style="font-size: 0.85em; margin-top: 10px;">
            Maps are interactive (pan, zoom, tooltips). Basemap tiles require an
            internet connection.
        </p>
    </div>
</body>
</html>
"""

_JINJA_ENV = Environment(autoescape=True)


def generate_html_report(report_data: dict[str, Any]) -> str:
    """
    Render the standalone HTML report.

    Parameters
    ----------
    report_data : dict
        Template values: ``timestamp``, ``analysis_type``, ``config``,
        ``metrics``, ``map_html``, ``map_caption``, ``table_data``,
        ``table_note`` and ``selection_info``. Missing optional keys hide the
        related section.

    Returns
    -------
    str
        Complete HTML document. User-supplied text is escaped; ``table_data``
        is inserted as trusted HTML.
    """
    template = _JINJA_ENV.from_string(REPORT_TEMPLATE)
    return template.render(version=__version__, **report_data)


def _timestamp() -> str:
    """Return the current time for report headers."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _file_stamp() -> str:
    """Return the current time for report file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _table_html(df: pd.DataFrame, sort_col: str, columns: list[str]) -> str:
    """Return the top report rows by ``sort_col`` (descending) as an HTML table."""
    return (
        df.sort_values(sort_col, ascending=False)[columns]
        .head(REPORT_TABLE_ROWS)
        .to_html(index=False, classes="table", float_format="%.2f")
    )


def _offer_download(html_report: str, prefix: str, key: str) -> None:
    """Show the download button and confirmation messages for a report."""
    st.download_button(
        label="⬇️ Download HTML Report",
        data=html_report,
        file_name=f"{prefix}_{_file_stamp()}.html",
        mime="text/html",
        key=key,
    )
    st.success("✅ Report generated! Click the download button above to save the HTML file.")
    st.info("💡 The downloaded HTML file includes an interactive map with pan/zoom.")


# =============================================================================
# TAB RENDERERS
# =============================================================================


def render_intro_tab() -> None:
    """Render the introduction tab."""
    st.header("What is COBALT?")

    col_main, col_sidebar = st.columns([2, 1])

    with col_main:
        st.markdown("""
        **COBALT** (COst and Benefit to Accidents – Light Touch) is the UK
        Department for Transport's tool for appraising the accident impacts of
        road schemes, comparing Do-Minimum (DM) and Do-Something (DS) networks.
        This dashboard maps COBALT and related appraisal outputs by link.

        ### This Dashboard
        This tool helps you:
        1. **Explore COBALT outputs** spatially on an interactive map
        2. **Compare DM vs DS flows** to identify where traffic patterns change
        3. **Filter and analyse benefits** using polygon selection and thresholds
        4. **Export HTML reports** with interactive maps and summary data

        ### Useful Links
        - [TAG](https://www.gov.uk/guidance/transport-analysis-guidance-tag)
        - [COBALT](https://www.gov.uk/government/publications/cobalt)
        - [Transport appraisal](https://www.gov.uk/transport/transport-appraisal)

        ---

        ### Data Requirements

        **Flow Comparison CSV** (Tab 2):
        - **Geometry column**: LineString coordinates
        - **DM flow column**: Do-Minimum scenario flows
        - **DS flow column**: Do-Something scenario flows

        **Benefits Analysis CSV** (Tab 3):
        - **Geometry column**: LineString coordinates (same structure as flow CSV)
        - **Benefits column**: Numeric benefits (positive or negative)
        """)

    with col_sidebar:
        st.info("""
        **Getting Started**

        1. Upload your COBALT CSV files using the sidebar:
           - Flow CSV for DM vs DS comparison
           - Benefits CSV for economic analysis
        2. Configure column names and coordinate system
        3. Explore the analysis tabs independently
        4. Generate HTML reports from each analysis tab
        """)
        st.markdown("---")
        st.caption(f"📊 Upload data files to begin analysis · v{__version__}")


def render_flow_tab(gdf: gpd.GeoDataFrame, sidebar_container: Any) -> None:
    """
    Render the DM vs DS flow comparison tab.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Flow data in EPSG:4326.
    sidebar_container : streamlit container
        Sidebar area for tab-specific controls.
    """
    st.header("DM vs DS Flow Comparison")
    st.caption("Identify links where traffic flow changes significantly between scenarios")

    with sidebar_container:
        st.markdown("---")
        st.subheader("🔄 Flow Comparison Settings")

        dm_col = st.text_input(
            "DM (Do-Minimum) flow column", value=DEFAULT_DM_COL, key="flow_dm_col"
        )
        ds_col = st.text_input(
            "DS (Do-Something) flow column", value=DEFAULT_DS_COL, key="flow_ds_col"
        )
        pct_threshold = st.slider(
            "Show links with |% change| ≥",
            min_value=0,
            max_value=100,
            value=DEFAULT_PCT_THRESHOLD,
            step=5,
            key="flow_pct_threshold",
        )

        st.markdown("### 🧷 Overlap Offsetting")
        offset_overlaps = st.checkbox(
            "Offset overlapping links", value=True, key="flow_offset_overlaps"
        )
        offset_step_px = st.slider(
            "Offset step (px)", 0, 20, DEFAULT_OFFSET_STEP_PX, 1, key="flow_offset_step_px"
        )
        base_offset_px = st.slider("Base offset (px)", -20, 20, 0, 1, key="flow_base_offset_px")

    missing = [c for c in (dm_col, ds_col) if c not in gdf.columns]
    if missing:
        st.error(
            f"❌ Column(s) not found: {', '.join(missing)}\n\n"
            f"Available: {', '.join(gdf.columns)}"
        )
        return

    gdf_calc = calculate_pct_change(gdf, dm_col, ds_col)
    gdf_filtered = filter_by_abs_threshold(gdf_calc, PCT_CHANGE_COL, pct_threshold)

    n_total = len(gdf_calc)
    n_valid = int(gdf_calc[PCT_CHANGE_COL].notna().sum())
    n_filtered = len(gdf_filtered)
    mean_change = f"{gdf_filtered[PCT_CHANGE_COL].mean():+.1f}%" if n_filtered > 0 else "—"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total links", f"{n_total:,}")
    col2.metric("Valid % change calculations", f"{n_valid:,}")
    col3.metric(f"Links with |change| ≥ {pct_threshold}%", f"{n_filtered:,}")
    col4.metric("Mean % change (filtered)", mean_change)

    if n_filtered == 0:
        st.warning(
            f"⚠️ No links found with |% change| ≥ {pct_threshold}%. " "Try lowering the threshold."
        )
        return

    flow_map = create_flow_map(
        gdf_filtered,
        PCT_CHANGE_COL,
        offset_overlaps=offset_overlaps,
        offset_step_px=offset_step_px,
        base_offset_px=base_offset_px,
    )
    add_polygon_draw(flow_map)

    st.subheader("Interactive Map")
    st.caption("Draw a polygon to select flow changes, then apply it to the Benefits tab")
    map_output = st_folium(flow_map, height=600, width=None, key="flow_map")

    drawn_polygon_geojson = _last_drawn_polygon(map_output)
    _render_polygon_controls(drawn_polygon_geojson)

    display_cols = [ROW_ID_COL, dm_col, ds_col, PCT_CHANGE_COL]
    with st.expander("📋 View filtered links data"):
        table_data = gdf_filtered.sort_values(ABS_PCT_CHANGE_COL, ascending=False)[
            display_cols
        ].head(APP_TABLE_ROWS)
        st.dataframe(table_data, use_container_width=True)
        st.caption(f"Showing top {APP_TABLE_ROWS} of {n_filtered:,} filtered links")

    # --- HTML report export ---------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Flow Comparison Report")
    st.caption("Generate a standalone HTML report with interactive map and analysis summary")

    if st.button("📄 Generate Flow Comparison Report", key="generate_flow_report", type="primary"):
        with st.spinner("Generating HTML report..."):
            report_data = {
                "timestamp": _timestamp(),
                "analysis_type": "DM vs DS Flow Comparison",
                "config": {
                    "DM Flow Column": dm_col,
                    "DS Flow Column": ds_col,
                    "% Change Threshold": f"≥ {pct_threshold}%",
                    "Source CRS": st.session_state.get("source_crs", DEFAULT_SOURCE_CRS),
                    "Target CRS": DEFAULT_TARGET_CRS,
                },
                "metrics": [
                    {"label": "Total Links", "value": f"{n_total:,}"},
                    {"label": "Valid Calculations", "value": f"{n_valid:,}"},
                    {"label": f"Links ≥ {pct_threshold}% Change", "value": f"{n_filtered:,}"},
                    {"label": "Mean % Change (Filtered)", "value": mean_change},
                ],
                "map_html": create_report_map(
                    gdf_filtered, PCT_CHANGE_COL, "Flow Changes", is_flow_map=True
                ),
                "map_caption": (
                    f"Showing {n_filtered:,} links with absolute % change ≥ {pct_threshold}%"
                ),
                "table_data": _table_html(gdf_filtered, ABS_PCT_CHANGE_COL, display_cols),
                "table_note": (
                    f"Showing top {REPORT_TABLE_ROWS} of {n_filtered:,} filtered links "
                    "(sorted by absolute % change)"
                    if n_filtered > REPORT_TABLE_ROWS
                    else None
                ),
                "selection_info": None,
            }
            logger.info(
                "Flow report generated: DM=%s DS=%s threshold=%s%% links=%d",
                dm_col,
                ds_col,
                pct_threshold,
                n_filtered,
            )
            _offer_download(
                generate_html_report(report_data), "COBALT_Flow_Report", "download_flow_report"
            )


def _last_drawn_polygon(map_output: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the most recent Polygon/MultiPolygon drawn on a ``st_folium`` map."""
    if not map_output or not map_output.get("all_drawings"):
        return None
    geometry = map_output["all_drawings"][-1].get("geometry") or {}
    if geometry.get("type") in ("Polygon", "MultiPolygon"):
        return geometry
    return None


def _clear_shared_polygon() -> None:
    """Clear the polygon shared between the Flow and Benefits tabs."""
    st.session_state.shared_polygon_geojson = None
    st.session_state.apply_polygon_to_benefits = False


def _render_polygon_controls(drawn_polygon_geojson: dict[str, Any] | None) -> None:
    """Show the status of the shared polygon and controls to apply or clear it."""
    st.markdown("---")
    col_info, col_action = st.columns([2, 1])

    with col_info:
        if drawn_polygon_geojson:
            st.session_state.shared_polygon_geojson = drawn_polygon_geojson
            st.success("✅ **Polygon drawn on map**")
            st.caption("You can edit or delete the polygon using the map controls")
        elif st.session_state.shared_polygon_geojson:
            st.info("ℹ️ **Previously drawn polygon stored** (not visible on this map)")
        else:
            st.info("ℹ️ No polygon drawn yet. Use the polygon tool on the map above.")

    with col_action:
        if not st.session_state.shared_polygon_geojson:
            return

        st.session_state.apply_polygon_to_benefits = st.checkbox(
            "**Apply polygon to Benefits tab**",
            value=st.session_state.apply_polygon_to_benefits,
            key="apply_polygon_checkbox",
            help="When checked, the drawn polygon filters features in the Benefits tab",
        )
        if st.session_state.apply_polygon_to_benefits:
            st.success("🔗 Polygon will be applied to Benefits tab")

        if st.button("🗑️ Clear polygon", key="clear_polygon_tab2"):
            _clear_shared_polygon()
            st.rerun()


def render_benefits_tab(gdf: gpd.GeoDataFrame, geom_col: str, sidebar_container: Any) -> None:
    """
    Render the benefits analysis tab with polygon selection.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Benefits data in EPSG:4326.
    geom_col : str
        Original geometry column name, shown in the selected rows table.
    sidebar_container : streamlit container
        Sidebar area for tab-specific controls.
    """
    st.header("Benefits Analysis")

    with sidebar_container:
        st.markdown("---")
        st.subheader("💰 Benefits Settings")

        value_col = st.text_input(
            "Benefits column", value=DEFAULT_BENEFITS_COL, key="benefits_value_col"
        )
        abs_threshold = st.number_input(
            "Absolute benefit threshold",
            min_value=0.0,
            value=DEFAULT_BENEFITS_THRESHOLD,
            step=1000.0,
            key="benefits_threshold",
        )
        show_table = st.checkbox("Show selected rows table", value=True, key="benefits_show_table")

    if value_col not in gdf.columns:
        st.error(f"❌ Column '{value_col}' not found.\n\nAvailable: {', '.join(gdf.columns)}")
        return

    # Prepare data
    gdf_prep = gdf.copy()
    gdf_prep[value_col] = pd.to_numeric(gdf_prep[value_col], errors="coerce")
    abs_col = f"ABS_{value_col}"
    gdf_prep[abs_col] = gdf_prep[value_col].abs()

    original_count = int(gdf_prep[value_col].notna().sum())
    original_total = float(gdf_prep[value_col].sum(skipna=True))

    gdf_filtered = filter_by_abs_threshold(gdf_prep, value_col, abs_threshold)
    map_count = len(gdf_filtered)
    map_total = float(gdf_filtered[value_col].sum(skipna=True))

    with sidebar_container:
        st.markdown("---")
        st.subheader("📊 Dataset Totals (Signed)")
        st.metric("Original links (valid benefits)", f"{original_count:,}")
        st.metric("Original total benefits", f"{original_total:,.2f}")
        st.metric(f"Map links (|benefit| ≥ {abs_threshold:,.0f})", f"{map_count:,}")
        st.metric("Map total benefits", f"{map_total:,.2f}")
        st.caption(f"Threshold filters by absolute value (≥ ±{abs_threshold:,.0f}).")

    using_shared_polygon = bool(
        st.session_state.apply_polygon_to_benefits
        and st.session_state.shared_polygon_geojson is not None
    )

    col_map, col_stats = st.columns([2.2, 1.0], gap="large")

    with col_map:
        st.subheader("Interactive Map")
        benefits_map = create_benefits_map(gdf_filtered, value_col)

        if using_shared_polygon:
            folium.GeoJson(
                st.session_state.shared_polygon_geojson,
                name="Selection from Flow Tab",
                style_function=lambda _: {
                    "fillColor": "#3388ff",
                    "color": "#0066cc",
                    "weight": LINE_WEIGHT,
                    "fillOpacity": 0.15,
                    "dashArray": "5, 5",
                },
                highlight_function=lambda _: {"fillOpacity": 0.25},
                show=True,
            ).add_to(benefits_map)
            st.info("🔗 **Using polygon from Flow Comparison tab**")
            st.caption("The blue dashed outline shows your selection from Tab 2")
        else:
            add_polygon_draw(benefits_map)
            st.caption("Draw a polygon to select and analyse specific features")

        map_output = st_folium(benefits_map, height=650, width=None, key="benefits_map")

    if using_shared_polygon:
        selection_geom = geojson_to_shapely(st.session_state.shared_polygon_geojson)
    else:
        selection_geom = geojson_to_shapely(_last_drawn_polygon(map_output))

    has_selection = selection_geom is not None
    selected = (
        gdf_filtered[gdf_filtered.intersects(selection_geom)].copy()
        if has_selection
        else gdf_filtered.iloc[0:0]
    )
    selected_count = len(selected)
    selected_total = float(selected[value_col].sum(skipna=True))

    with col_stats:
        st.subheader("Selection Summary")
        st.caption("**Original dataset** (all valid benefits):")
        st.metric("Total benefits", f"{original_total:,.2f}")
        st.caption(f"**On-map dataset** (|benefit| ≥ {abs_threshold:,.0f}):")
        st.metric("Total benefits", f"{map_total:,.2f}")
        st.markdown("---")

        if not has_selection:
            if using_shared_polygon:
                st.warning("⚠️ Applied polygon from Tab 2 is invalid or could not be loaded.")
            else:
                st.info("👆 Draw a polygon on the map to see selection totals.")
            st.metric("Selected features", 0)
            st.metric(f"Selected {value_col}", "—")
        else:
            if using_shared_polygon:
                st.success("✅ **Selection from Flow tab polygon**")
            st.metric("Selected features", f"{selected_count:,}")
            st.metric(f"Selected {value_col}", f"{selected_total:,.2f}")

            if show_table and selected_count > 0:
                st.markdown("---")
                st.markdown("### Selected Rows")
                display_cols = [c for c in (ROW_ID_COL, value_col, geom_col) if c in selected]
                extra_cols = [
                    c
                    for c in selected.columns
                    if c not in display_cols and c not in ("geometry", abs_col, ABS_PCT_CHANGE_COL)
                ]
                st.dataframe(
                    selected[display_cols + extra_cols].head(SELECTED_TABLE_ROWS),
                    use_container_width=True,
                )
                if selected_count > SELECTED_TABLE_ROWS:
                    st.caption(
                        f"Showing first {SELECTED_TABLE_ROWS} of {selected_count:,} selected rows"
                    )

        if using_shared_polygon:
            st.markdown("---")
            if st.button("🗑️ Clear applied polygon", key="clear_polygon_tab3"):
                _clear_shared_polygon()
                st.rerun()

    # --- HTML report export ---------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Benefits Analysis Report")
    st.caption(
        "Generate a standalone HTML report with interactive map, metrics and selection summary"
    )

    if st.button("📄 Generate Benefits Report", key="generate_benefits_report", type="primary"):
        with st.spinner("Generating HTML report..."):
            report_cols = [ROW_ID_COL, value_col]
            has_selected_rows = has_selection and selected_count > 0

            if has_selected_rows:
                selection_data = {
                    "count": f"{selected_count:,}",
                    "total": f"{selected_total:,.2f}",
                    "polygon_source": (
                        "Flow Comparison Tab (imported)"
                        if using_shared_polygon
                        else "Drawn directly on Benefits Map"
                    ),
                }
                table_source, table_count, label = selected, selected_count, "selected features"
            else:
                selection_data = None
                table_source, table_count, label = gdf_filtered, map_count, "features on map"

            table_note = (
                f"Showing top {REPORT_TABLE_ROWS} of {table_count:,} {label} "
                "(sorted by absolute benefit)"
                if table_count > REPORT_TABLE_ROWS
                else f"Showing all {table_count:,} {label}"
            )
            map_caption = (
                f"Displaying {map_count:,} links with |{value_col}| ≥ {abs_threshold:,.0f}"
            )
            if has_selected_rows:
                map_caption += f". Polygon selection contains {selected_count:,} features."

            report_data = {
                "timestamp": _timestamp(),
                "analysis_type": "Benefits Analysis"
                + (" (with polygon selection)" if has_selection else ""),
                "config": {
                    "Benefits Column": value_col,
                    "Absolute Threshold": f"≥ {abs_threshold:,.0f}",
                    "Source CRS": st.session_state.get("source_crs", DEFAULT_SOURCE_CRS),
                    "Target CRS": DEFAULT_TARGET_CRS,
                    "Geometry Column": geom_col,
                    "Features on Map": f"{map_count:,}",
                },
                "metrics": [
                    {"label": "Original Dataset Links", "value": f"{original_count:,}"},
                    {"label": "Original Total Benefits", "value": f"{original_total:,.2f}"},
                    {"label": "Map Dataset Links", "value": f"{map_count:,}"},
                    {"label": "Map Total Benefits", "value": f"{map_total:,.2f}"},
                ],
                "map_html": create_report_map(
                    gdf_filtered, value_col, "Benefits Analysis", is_flow_map=False
                ),
                "map_caption": map_caption,
                "table_data": (
                    _table_html(table_source, abs_col, report_cols) if table_count > 0 else None
                ),
                "table_note": table_note,
                "selection_info": selection_data,
            }
            logger.info(
                "Benefits report generated: column=%s threshold=%s map_links=%d selected=%d",
                value_col,
                abs_threshold,
                map_count,
                selected_count,
            )
            _offer_download(
                generate_html_report(report_data),
                "COBALT_Benefits_Report",
                "download_benefits_report",
            )


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def initialise_session_state() -> None:
    """Create session state keys used to share a polygon between tabs."""
    st.session_state.setdefault("shared_polygon_geojson", None)
    st.session_state.setdefault("apply_polygon_to_benefits", False)


def _load_upload(uploaded_file: Any, label: str, geom_col: str, source_crs: str):
    """
    Load an uploaded CSV and report the result in the sidebar.

    Returns
    -------
    geopandas.GeoDataFrame or None
        Loaded data, or ``None`` if no file was uploaded or loading failed.
    """
    if uploaded_file is None:
        return None
    try:
        gdf = load_geodataframe(uploaded_file.getvalue(), uploaded_file.name, geom_col, source_crs)
    except Exception as exc:  # noqa: BLE001 - show any load error to the user
        logger.exception("Failed to load %s file '%s'.", label, uploaded_file.name)
        st.sidebar.error(f"❌ {label} file error:\n\n{exc}")
        return None
    st.sidebar.success(f"✅ {label}: Loaded **{len(gdf):,}** features")
    return gdf


def main() -> None:
    """Run the Streamlit application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    st.set_page_config(page_title="COBALT Benefits Analysis", layout="wide")
    st.logo(LOGO_PATH)
    st.title("🚦 COBALT Benefits Analysis Dashboard")

    initialise_session_state()

    with st.sidebar:
        st.header("📁 Data Upload")

        st.subheader("Flow Comparison (Tab 2)")
        flow_file = st.file_uploader(
            "Upload Flow CSV",
            type=["csv"],
            help="CSV containing DM/DS flow data with LineString geometry",
            key="flow_uploader",
        )

        st.subheader("Benefits Analysis (Tab 3)")
        benefits_file = st.file_uploader(
            "Upload Benefits CSV",
            type=["csv"],
            help="CSV containing benefits data with LineString geometry",
            key="benefits_uploader",
        )

        st.markdown("---")
        st.header("⚙️ Global Configuration")

        geom_col = st.text_input(
            "Geometry column name",
            value=DEFAULT_GEOM_COL,
            help="Column containing LineString coordinates (must be same in both files)",
        )
        source_crs = st.text_input(
            "Source coordinate system",
            value=DEFAULT_SOURCE_CRS,
            help="CRS of geometry coordinates (e.g., EPSG:27700 for UK grid)",
        )
        st.session_state.source_crs = source_crs

        tab_sidebar = st.container()

    gdf_flow = _load_upload(flow_file, "Flow", geom_col, source_crs)
    gdf_benefits = _load_upload(benefits_file, "Benefits", geom_col, source_crs)

    tab_intro, tab_flow, tab_benefits = st.tabs(
        ["📘 Introduction", "🔄 DM vs DS Flow Comparison", "💰 Benefits Analysis"]
    )

    with tab_intro:
        render_intro_tab()
        if flow_file is None and benefits_file is None:
            st.warning("⬅️ Please upload CSV files using the sidebar to access analysis tabs")

    with tab_flow:
        if flow_file is None:
            st.info("⬅️ Upload a Flow CSV file in the sidebar to view flow comparison analysis")
        elif gdf_flow is None:
            st.error("❌ Flow data failed to load. Check error message in sidebar.")
        else:
            render_flow_tab(gdf_flow, tab_sidebar)

    with tab_benefits:
        if benefits_file is None:
            st.info("⬅️ Upload a Benefits CSV file in the sidebar to view benefits analysis")
        elif gdf_benefits is None:
            st.error("❌ Benefits data failed to load. Check error message in sidebar.")
        else:
            render_benefits_tab(gdf_benefits, geom_col, tab_sidebar)


if __name__ == "__main__":
    main()
