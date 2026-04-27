"""
COBALT Benefits Analysis Dashboard
A three-tab Streamlit application for exploring transport appraisal outputs.
"""

import json
import re
from io import BytesIO
from datetime import datetime
import base64

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, shape
from shapely import wkt
import folium
from folium.plugins import Draw, PolyLineOffset
from streamlit_folium import st_folium
import streamlit as st
import branca.colormap as cm
from jinja2 import Template


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_GEOM_COL = "Coords"
DEFAULT_SOURCE_CRS = "EPSG:27700"
DEFAULT_TARGET_CRS = "EPSG:4326"
DEFAULT_BENEFITS_COL = "Total Cost DIFF"
DEFAULT_BENEFITS_THRESHOLD = 10000.0
DEFAULT_DM_COL = "Flow DM"
DEFAULT_DS_COL = "Flow DS"
DEFAULT_PCT_THRESHOLD = 10

COLORMAP_DIVERGING = ["#d7191c", "#ffffbf", "#2c7bb6"]
COLORMAP_BENEFITS = ["#2c7bb6", "#ffffbf", "#d7191c"]

MIN_OPACITY = 0.1
MAX_OPACITY = 1.0
OPACITY_GAMMA = 0.9


# =============================================================================
# GEOMETRY PARSING
# =============================================================================

def parse_linestring(value):
    """Parse LineString from WKT, GeoJSON string, or coordinate list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, LineString):
        return value

    string_value = str(value).strip()
    if not string_value:
        return None

    # Try WKT
    try:
        return wkt.loads(string_value)
    except Exception:
        pass

    # Try GeoJSON string
    if string_value.startswith("{") and '"coordinates"' in string_value:
        try:
            return shape(json.loads(string_value))
        except Exception:
            pass

    # Try coordinate list
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string_value)
    if len(numbers) >= 4 and len(numbers) % 2 == 0:
        try:
            coords = [(float(numbers[i]), float(numbers[i + 1])) for i in range(0, len(numbers), 2)]
            return LineString(coords)
        except Exception:
            pass

    return None


def geojson_to_shapely(geojson_dict):
    """Convert GeoJSON Polygon/MultiPolygon dict to shapely geometry."""
    if not geojson_dict:
        return None
    if geojson_dict.get("type") in ("Polygon", "MultiPolygon"):
        try:
            return shape(geojson_dict)
        except Exception:
            pass
    return None


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=True)
def load_geodataframe(file_bytes: bytes, filename: str, geom_column: str, source_crs: str):
    """Load CSV with geometry column and convert to GeoDataFrame in EPSG:4326."""
    df = pd.read_csv(BytesIO(file_bytes))

    if geom_column not in df.columns:
        raise ValueError(
            f"Geometry column '{geom_column}' not found in '{filename}'.\n"
            f"Available columns: {', '.join(df.columns)}"
        )

    df["geometry"] = df[geom_column].apply(parse_linestring) # type: ignore
    df = df[df["geometry"].notna()].copy()

    if len(df) == 0:
        raise ValueError(f"No valid LineString geometries found in column '{geom_column}'")

    df["__rowid__"] = df.index.astype(int)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=source_crs)

    try:
        return gdf.to_crs(DEFAULT_TARGET_CRS)
    except Exception as e:
        raise ValueError(
            f"CRS conversion failed from {source_crs} to {DEFAULT_TARGET_CRS}.\n"
            f"Check your source CRS setting. Error: {e}"
        )


# =============================================================================
# MAPPING - INTERACTIVE (FOR STREAMLIT DISPLAY)
# =============================================================================

def add_polygon_draw(folium_map):
    """Add polygon drawing control to map."""
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


def calculate_opacity(value, max_abs_value):
    """Calculate line opacity based on absolute value magnitude."""
    if value is None or pd.isna(value) or max_abs_value == 0:
        return 0.25
    normalized = min(abs(value) / max_abs_value, 1.0) ** OPACITY_GAMMA
    return MIN_OPACITY + normalized * (MAX_OPACITY - MIN_OPACITY)

def _geom_to_latlon_coords(geom):
    """Convert a shapely LineString to folium [(lat, lon), ...]."""
    if geom is None:
        return None
    # Shapely coords are (x=lon, y=lat) once you're in EPSG:4326
    try:
        return [(y, x) for x, y in geom.coords]
    except Exception:
        return None


def _add_offset_lines_layer(
    folium_map,
    gdf_lines,
    value_column,
    style_row_fn,
    layer_name,
    tooltip_alias,
    offset_overlaps=True,
    offset_step_px=6,
    base_offset_px=0,
):
    """
    Add LineString features as PolyLineOffset layers.
    If offset_overlaps=True, identical geometries are stacked with symmetric offsets.
    """
    fg = folium.FeatureGroup(name=layer_name, show=True)

    if len(gdf_lines) == 0:
        fg.add_to(folium_map)
        return

    # Build per-row offsets
    df = gdf_lines.copy()

    if offset_overlaps and offset_step_px != 0:
        # Group identical geometries using WKB hex (fast + stable for exact duplicates)
        df["_geom_key"] = df.geometry.apply(lambda g: getattr(g, "wkb_hex", None))
        df["_idx"] = df.groupby("_geom_key").cumcount()
        df["_n"] = df.groupby("_geom_key")["_geom_key"].transform("size")
        # Symmetric offsets around 0: -(n-1)/2, ..., +(n-1)/2
        df["_offset_px"] = base_offset_px + (df["_idx"] - (df["_n"] - 1) / 2.0) * offset_step_px
    else:
        df["_offset_px"] = base_offset_px

    # Add polylines
    for _, row in df.iterrows():
        coords = _geom_to_latlon_coords(row.geometry)
        if not coords or len(coords) < 2:
            continue

        style = style_row_fn(row)
        tooltip_html = f"Row ID: {row.get('__rowid__', '')}<br>{tooltip_alias}: {row.get(value_column, '')}"
        tooltip = folium.Tooltip(tooltip_html, sticky=True)

        PolyLineOffset(
            locations=coords,
            color=style.get("color", "#888888"),
            weight=style.get("weight", 3),
            opacity=style.get("opacity", 0.8),
            offset=float(row["_offset_px"]), # type: ignore
            tooltip=tooltip,
        ).add_to(fg)

    fg.add_to(folium_map)


def create_benefits_map(gdf_display, value_column):
    """Create interactive benefits analysis map with polygon selection."""
    if len(gdf_display) == 0:
        return folium.Map(location=[51.4545, -2.5879], zoom_start=9, tiles="CartoDB positron")

    bounds = gdf_display.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    if len(gdf_display) == 0:
        add_polygon_draw(m)
        return m

    # Prepare colormap
    values = pd.to_numeric(gdf_display[value_column], errors="coerce")
    vmin = float(values.min()) if values.notna().any() else 0.0
    vmax = float(values.max()) if values.notna().any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    colormap = cm.LinearColormap(colors=COLORMAP_BENEFITS, vmin=vmin, vmax=vmax)
    colormap.caption = f"{value_column} (Signed benefits)"
    colormap.add_to(m)

    # Opacity scaling
    abs_values = values.abs()
    max_abs = float(abs_values.max()) if abs_values.notna().any() else 1.0
    if max_abs == 0:
        max_abs = 1.0

    def style_function(feature):
        raw_value = feature["properties"].get(value_column)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None

        color = colormap(numeric_value) if numeric_value is not None else "#888888"
        opacity = calculate_opacity(numeric_value, max_abs)

        return {"color": color, "weight": 3, "opacity": float(opacity)}

    def style_row(row):
        raw_value = row.get(value_column)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None

        color = colormap(numeric_value) if numeric_value is not None else "#888888"
        opacity = calculate_opacity(numeric_value, max_abs)
        return {"color": color, "weight": 3, "opacity": float(opacity)}

    _add_offset_lines_layer(
        folium_map=m,
        gdf_lines=gdf_display,
        value_column=value_column,
        style_row_fn=style_row,
        layer_name="Benefits",
        tooltip_alias=value_column,
        offset_overlaps=True,      # you can expose controls similarly if you want
        offset_step_px=6,
        base_offset_px=0,
    )

    folium.LayerControl().add_to(m)
    add_polygon_draw(m)

    return m


def create_flow_map(
    gdf,
    pct_change_column,
    offset_overlaps=True,
    offset_step_px=6,
    base_offset_px=0,
):
    """Create flow comparison map with fixed zoom and clipped color scale + optional polyline offsets."""
    if len(gdf) == 0:
        return folium.Map(location=[51.4545, -2.5879], zoom_start=9, tiles="CartoDB positron")

    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    # Clipped color scale
    colormap = cm.LinearColormap(colors=COLORMAP_DIVERGING, vmin=-100.0, vmax=100.0)
    colormap.caption = f"{pct_change_column} (% clipped to ±100)"
    colormap.add_to(m)

    def style_row(row):
        raw_value = row.get(pct_change_column)
        try:
            clipped_value = max(-100.0, min(100.0, float(raw_value)))
            color = colormap(clipped_value)
            opacity = 1.0
        except (TypeError, ValueError):
            color = "#888888"
            opacity = 0.3
        return {"color": color, "weight": 3, "opacity": opacity}

    # Draw as offset polylines (instead of GeoJson) so overlaps can be separated
    _add_offset_lines_layer(
        folium_map=m,
        gdf_lines=gdf,
        value_column=pct_change_column,
        style_row_fn=style_row,
        layer_name="Flow change",
        tooltip_alias="% Change",
        offset_overlaps=offset_overlaps,
        offset_step_px=offset_step_px,
        base_offset_px=base_offset_px,
    )

    folium.LayerControl().add_to(m)
    return m

# =============================================================================
# REPORT GENERATION - MAPS (NO DRAWING TOOLS)
# =============================================================================

def create_report_map(gdf, value_column, map_title, is_flow_map=False):
    """Create a Folium map for HTML report export (no drawing tools)."""
    if len(gdf) == 0:
        return "<p style='text-align:center; padding:40px; color:#888;'>No data to display</p>"
    
    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    if is_flow_map:
        # Flow map styling (diverging colormap)
        colormap = cm.LinearColormap(colors=COLORMAP_DIVERGING, vmin=-100.0, vmax=100.0)
        colormap.caption = f"{value_column} (% clipped to ±100)"
        
        def style_function(feature):
            raw_value = feature["properties"].get(value_column)
            try:
                clipped_value = max(-100.0, min(100.0, float(raw_value)))
                color = colormap(clipped_value)
                opacity = 1.0
            except (TypeError, ValueError):
                color = "#888888"
                opacity = 0.3
            return {"color": color, "weight": 3, "opacity": opacity}
    else:
        # Benefits map styling
        values = pd.to_numeric(gdf[value_column], errors="coerce")
        vmin = float(values.min()) if values.notna().any() else 0.0
        vmax = float(values.max()) if values.notna().any() else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        
        colormap = cm.LinearColormap(colors=COLORMAP_BENEFITS, vmin=vmin, vmax=vmax)
        colormap.caption = f"{value_column} (Signed benefits)"
        
        abs_values = values.abs()
        max_abs = float(abs_values.max()) if abs_values.notna().any() else 1.0
        if max_abs == 0:
            max_abs = 1.0
        
        def style_function(feature):
            raw_value = feature["properties"].get(value_column)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = None
            
            color = colormap(numeric_value) if numeric_value is not None else "#888888"
            opacity = calculate_opacity(numeric_value, max_abs)
            return {"color": color, "weight": 3, "opacity": float(opacity)}
    
    colormap.add_to(m)
    
    def style_row(row):
        raw_value = row.get(value_column)

        if is_flow_map:
            try:
                clipped_value = max(-100.0, min(100.0, float(raw_value)))
                color = colormap(clipped_value)
                opacity = 1.0
            except (TypeError, ValueError):
                color = "#888888"
                opacity = 0.3
            return {"color": color, "weight": 3, "opacity": opacity}

        # Benefits style
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None

        color = colormap(numeric_value) if numeric_value is not None else "#888888"
        opacity = calculate_opacity(numeric_value, max_abs)
        return {"color": color, "weight": 3, "opacity": float(opacity)}

    _add_offset_lines_layer(
        folium_map=m,
        gdf_lines=gdf,
        value_column=value_column,
        style_row_fn=style_row,
        layer_name=map_title,
        tooltip_alias=value_column if not is_flow_map else "% Change",
        offset_overlaps=True,
        offset_step_px=6,
        base_offset_px=0,
    )
    
    folium.LayerControl().add_to(m)
    
    # Return HTML string (NOT the map object)
    return m._repr_html_()


# =============================================================================
# HTML REPORT TEMPLATE & GENERATION
# =============================================================================

def generate_html_report(report_data):
    """Generate complete HTML report from captured data using Jinja2."""
    
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>COBALT Analysis Report - {{ timestamp }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                line-height: 1.6;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f7fa;
                color: #1f2937;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 5px 0;
                opacity: 0.9;
            }
            .section {
                background: white;
                padding: 30px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .section h2 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .metric-label {
                font-size: 0.85em;
                color: #6b7280;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #1f2937;
            }
            .map-container {
                margin: 25px 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .map-container iframe {
                width: 100%;
                height: 600px;
                border: none;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
            }
            th {
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 10px 12px;
                border-bottom: 1px solid #e5e7eb;
            }
            tr:hover {
                background: #f9fafb;
            }
            tr:nth-child(even) {
                background: #f9fafb;
            }
            .config-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                background: #f9fafb;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .config-item {
                display: flex;
                justify-content: space-between;
                padding: 10px;
                background: white;
                border-radius: 5px;
                border-left: 3px solid #10b981;
            }
            .config-label {
                font-weight: 600;
                color: #6b7280;
            }
            .config-value {
                color: #1f2937;
                font-family: 'Courier New', monospace;
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #6b7280;
                font-size: 0.9em;
                margin-top: 40px;
                border-top: 2px solid #e5e7eb;
            }
            .warning-box {
                background: #fef3c7;
                border-left: 4px solid #f59e0b;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .info-box {
                background: #dbeafe;
                border-left: 4px solid #3b82f6;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .success-box {
                background: #d1fae5;
                border-left: 4px solid #10b981;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚦 COBALT Benefits Analysis Report</h1>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            <p><strong>Analysis Type:</strong> {{ analysis_type }}</p>
        </div>

        <!-- Configuration Section -->
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

        <!-- Metrics Section -->
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

        <!-- Map Section -->
        {% if map_html %}
        <div class="section">
            <h2>🗺️ Interactive Map</h2>
            {% if map_caption %}
            <div class="info-box">{{ map_caption }}</div>
            {% endif %}
            <div class="map-container">
                <iframe srcdoc="{{ map_html | e }}"></iframe>
            </div>
            <p style="color: #6b7280; font-size: 0.9em; margin-top: 10px;">
                <strong>Map Controls:</strong> Use mouse wheel to zoom, click and drag to pan. 
                Click layer icon (top right) to toggle layers. Hover over features to see details.
            </p>
        </div>
        {% endif %}

        <!-- Selection Details (Benefits tab) -->
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

        <!-- Data Table -->
        {% if table_data %}
        <div class="section">
            <h2>📋 Data Summary (Top Results)</h2>
            <div style="overflow-x: auto;">
                {{ table_data | safe }}
            </div>
            {% if table_note %}
            <p style="color: #6b7280; font-size: 0.9em; margin-top: 10px;">{{ table_note }}</p>
            {% endif %}
        </div>
        {% endif %}

        <div class="footer">
            <p><strong>COBALT Benefits Analysis Dashboard</strong></p>
            <p>UK Department for Transport WebTAG-compliant appraisal tool</p>
            <p style="font-size: 0.85em; margin-top: 10px;">
                This is an offline-capable HTML report. All maps are fully interactive (pan, zoom, tooltips).
            </p>
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    return template.render(**report_data)


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_intro_tab():
    """Render introduction tab."""
    st.header("What is COBALT?")

    col_main, col_sidebar = st.columns([2, 1])

    with col_main:
        st.markdown("""
        **COBALT** (Cost Benefit Analysis Light Touch) is the UK Department for Transport's 
        appraisal tool for assessing the economic benefits of transport schemes.

        ### Key Features
        - **Economic Appraisal**: Calculates user benefits including travel time savings, 
          vehicle operating costs, and broader economic impacts
        - **Network Modelling**: Works with traffic models to compare Do-Minimum (DM) vs 
          Do-Something (DS) scenarios
        - **WebTAG Compliant**: Follows official UK transport appraisal guidance

        ### This Dashboard
        This tool helps you:
        1. **Explore COBALT outputs** spatially on an interactive map
        2. **Compare DM vs DS flows** to identify where traffic patterns change
        3. **Filter and analyse benefits** using polygon selection and thresholds
        4. **Export professional HTML reports** with interactive maps and summary data

        ### Useful Links
        - https://www.gov.uk/guidance/transport-analysis-guidance-tag
        - https://www.gov.uk/government/publications/cobalt
        - https://www.gov.uk/transport/transport-appraisal

        ---

        ### Data Requirements
        
        **Flow Comparison CSV** (Tab 2):
        - **Geometry column**: LineString coordinates
        - **DM flow column**: Do-Minimum scenario flows
        - **DS flow column**: Do-Something scenario flows
        
        **Benefits Analysis CSV** (Tab 3):
        - **Geometry column**: LineString coordinates (same structure as flow CSV)
        - **Benefits column**: Numeric economic benefits (positive or negative)
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
        st.caption("📊 Upload data files to begin analysis")

        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 5px;'>
            <p style='color: #666;'>🚦</p>
            <p style='color: #666; font-size: 0.9em;'>COBALT Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)


def render_flow_tab(gdf, sidebar_container):
    """Render DM vs DS flow comparison tab."""
    st.header("DM vs DS Flow Comparison")
    st.caption("Identify trips where traffic flow changes significantly between scenarios")

    # Sidebar controls
    with sidebar_container:
        st.markdown("---")
        st.subheader("🔄 Flow Comparison Settings")

        dm_col = st.text_input("DM (Do-Minimum) flow column", value=DEFAULT_DM_COL, key="flow_dm_col")
        ds_col = st.text_input("DS (Do-Something) flow column", value=DEFAULT_DS_COL, key="flow_ds_col")
        pct_threshold = st.slider(
            "Show trips with |% change| ≥",
            min_value=0,
            max_value=100,
            value=DEFAULT_PCT_THRESHOLD,
            step=5,
            key="flow_pct_threshold"
        )

        st.markdown("### 🧷 Overlap Offsetting")
        offset_overlaps = st.checkbox("Offset overlapping links", value=True, key="flow_offset_overlaps")
        offset_step_px = st.slider("Offset step (px)", 0, 20, 6, 1, key="flow_offset_step_px")
        base_offset_px = st.slider("Base offset (px)", -20, 20, 0, 1, key="flow_base_offset_px")

    # Validate columns
    missing = [c for c in [dm_col, ds_col] if c not in gdf.columns]
    if missing:
        st.error(f"❌ Column(s) not found: {', '.join(missing)}\n\nAvailable: {', '.join(gdf.columns)}")
        return

    # Calculate percentage change
    gdf_calc = gdf.copy()
    gdf_calc[dm_col] = pd.to_numeric(gdf_calc[dm_col], errors="coerce")
    gdf_calc[ds_col] = pd.to_numeric(gdf_calc[ds_col], errors="coerce")

    dm_safe = gdf_calc[dm_col].replace(0, pd.NA)
    gdf_calc["pct_change"] = ((gdf_calc[ds_col] - gdf_calc[dm_col]) / dm_safe * 100)
    gdf_calc["abs_pct_change"] = gdf_calc["pct_change"].abs()

    # Filter
    gdf_filtered = gdf_calc[
        gdf_calc["pct_change"].notna() & (gdf_calc["abs_pct_change"] >= pct_threshold)
    ].copy()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total trips", f"{len(gdf_calc):,}")
    with col2:
        st.metric("Valid % change calculations", f"{gdf_calc['pct_change'].notna().sum():,}")
    with col3:
        st.metric(f"Trips with |change| ≥ {pct_threshold}%", f"{len(gdf_filtered):,}")
    with col4:
        if len(gdf_filtered) > 0:
            st.metric("Mean % change (filtered)", f"{gdf_filtered['pct_change'].mean():+.1f}%")
        else:
            st.metric("Mean % change (filtered)", "—")

    if len(gdf_filtered) == 0:
        st.warning(f"⚠️ No trips found with |% change| ≥ {pct_threshold}%. Try lowering the threshold.")
        return

    # Create map with drawing enabled
    flow_map = create_flow_map(
        gdf_filtered,
        "pct_change",
        offset_overlaps=offset_overlaps,
        offset_step_px=offset_step_px,
        base_offset_px=base_offset_px,
    )
    
    # Add polygon drawing to Tab 2 map
    add_polygon_draw(flow_map)
    
    # Render map and capture output
    st.subheader("Interactive Map")
    st.caption("Draw a polygon to select flow changes, then apply it to the Benefits tab")
    
    map_output = st_folium(flow_map, height=600, width=None, key="flow_map")

    # Extract polygon from map drawings
    drawn_polygon_geojson = None
    if map_output and map_output.get("all_drawings"):
        last_drawing = map_output["all_drawings"][-1]
        if last_drawing.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            drawn_polygon_geojson = last_drawing["geometry"]

    # Polygon control section
    st.markdown("---")
    col_poly_info, col_poly_action = st.columns([2, 1])
    
    with col_poly_info:
        if drawn_polygon_geojson:
            st.success("✅ **Polygon drawn on map**")
            st.caption("You can edit or delete the polygon using the map controls")
            
            # Store in session state
            st.session_state.shared_polygon_geojson = drawn_polygon_geojson
        else:
            if st.session_state.shared_polygon_geojson:
                st.info("ℹ️ **Previously drawn polygon stored** (not visible on this map)")
            else:
                st.info("ℹ️ No polygon drawn yet. Use the polygon tool on the map above.")
    
    with col_poly_action:
        if st.session_state.shared_polygon_geojson:
            apply_checked = st.checkbox(
                "**Apply polygon to Benefits tab**",
                value=st.session_state.apply_polygon_to_benefits,
                key="apply_polygon_checkbox",
                help="When checked, the drawn polygon will filter features in the Benefits Analysis tab"
            )
            
            # Update session state
            st.session_state.apply_polygon_to_benefits = apply_checked
            
            if apply_checked:
                st.success("🔗 Polygon will be applied to Benefits tab")
            
            # Clear button
            if st.button("🗑️ Clear polygon", key="clear_polygon_tab2"):
                st.session_state.shared_polygon_geojson = None
                st.session_state.apply_polygon_to_benefits = False
                st.rerun()

    # Table
    with st.expander("📋 View filtered trips data"):
        display_cols = ["__rowid__", dm_col, ds_col, "pct_change"]
        table_data = gdf_filtered.sort_values("abs_pct_change", ascending=False)[display_cols].head(200)
        st.dataframe(table_data, use_container_width=True)
        st.caption(f"Showing top 200 of {len(gdf_filtered):,} filtered trips")

    # =========================================================================
    # HTML REPORT EXPORT SECTION - FLOW TAB
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 Export Flow Comparison Report")
    st.caption("Generate a standalone HTML report with interactive map and analysis summary")

    if st.button("📄 Generate Flow Comparison Report", key="generate_flow_report", type="primary"):
        with st.spinner("Generating HTML report..."):
            # Capture current state
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_type": "DM vs DS Flow Comparison",
                "config": {
                    "DM Flow Column": dm_col,
                    "DS Flow Column": ds_col,
                    "% Change Threshold": f"≥ {pct_threshold}%",
                    "Source CRS": st.session_state.get('source_crs', DEFAULT_SOURCE_CRS),
                    "Target CRS": DEFAULT_TARGET_CRS
                },
                "metrics": [
                    {"label": "Total Trips", "value": f"{len(gdf_calc):,}"},
                    {"label": "Valid Calculations", "value": f"{gdf_calc['pct_change'].notna().sum():,}"},
                    {"label": f"Trips ≥ {pct_threshold}% Change", "value": f"{len(gdf_filtered):,}"},
                    {"label": "Mean % Change (Filtered)", "value": f"{gdf_filtered['pct_change'].mean():+.1f}%" if len(gdf_filtered) > 0 else "—"}
                ],
                "map_html": create_report_map(gdf_filtered, "pct_change", "Flow Changes", is_flow_map=True),
                "map_caption": f"Showing {len(gdf_filtered):,} trips with absolute % change ≥ {pct_threshold}%",
                "table_data": gdf_filtered.sort_values("abs_pct_change", ascending=False)[["__rowid__", dm_col, ds_col, "pct_change"]].head(100).to_html(index=False, classes="table", float_format="%.2f") if len(gdf_filtered) > 0 else None,
                "table_note": f"Showing top 100 of {len(gdf_filtered):,} filtered trips (sorted by absolute % change)" if len(gdf_filtered) > 100 else None,
                "selection_info": None
            }
            
            html_report = generate_html_report(report_data)
            
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_report,
                file_name=f"COBALT_Flow_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_flow_report"
            )
            st.success("✅ Report generated! Click the download button above to save the HTML file.")
            st.info("💡 The downloaded HTML file works offline and includes an interactive map.")


def render_benefits_tab(gdf, geom_col, sidebar_container):
    """Render benefits analysis tab with polygon selection."""
    st.header("Benefits Analysis")

    # Sidebar controls
    with sidebar_container:
        st.markdown("---")
        st.subheader("💰 Benefits Settings")

        value_col = st.text_input("Benefits column", value=DEFAULT_BENEFITS_COL, key="benefits_value_col")
        abs_threshold = st.number_input(
            "Absolute benefit threshold",
            min_value=0.0,
            value=DEFAULT_BENEFITS_THRESHOLD,
            step=1000.0,
            key="benefits_threshold"
        )
        show_table = st.checkbox("Show selected rows table", value=True, key="benefits_show_table")

    # Validate column
    if value_col not in gdf.columns:
        st.error(f"❌ Column '{value_col}' not found.\n\nAvailable: {', '.join(gdf.columns)}")
        return

    # Prepare data
    gdf_prep = gdf.copy()
    gdf_prep[value_col] = pd.to_numeric(gdf_prep[value_col], errors="coerce")
    abs_col = f"ABS_{value_col}"
    gdf_prep[abs_col] = gdf_prep[value_col].abs()

    # Totals
    original_count = int(gdf_prep[value_col].notna().sum())
    original_total = float(gdf_prep[value_col].sum(skipna=True))

    # Filter for map
    gdf_filtered = gdf_prep[gdf_prep[abs_col].notna() & (gdf_prep[abs_col] >= abs_threshold)].copy()
    map_count = len(gdf_filtered)
    map_total = float(gdf_filtered[value_col].sum(skipna=True))

    # Sidebar summary
    with sidebar_container:
        st.markdown("---")
        st.subheader("📊 Dataset Totals (Signed)")
        st.metric("Original links (valid benefits)", f"{original_count:,}")
        st.metric("Original total benefits", f"{original_total:,.2f}")
        st.metric(f"Map links (|benefit| ≥ {abs_threshold:,.0f})", f"{map_count:,}")
        st.metric("Map total benefits", f"{map_total:,.2f}")
        st.caption(f"Threshold filters by absolute value (≥ ±{abs_threshold:,.0f}).")

    # Layout
    col_map, col_stats = st.columns([2.2, 1.0], gap="large")

    with col_map:
        st.subheader("Interactive Map")
        
        # Create base map
        benefits_map = create_benefits_map(gdf_filtered, value_col)
        
        # Check if polygon should be applied from Tab 2
        using_shared_polygon = (
            st.session_state.apply_polygon_to_benefits 
            and st.session_state.shared_polygon_geojson is not None
        )
        
        if using_shared_polygon:
            # Visualize the shared polygon
            folium.GeoJson(
                st.session_state.shared_polygon_geojson,
                name="Selection from Flow Tab",
                style_function=lambda x: {
                    "fillColor": "#3388ff",
                    "color": "#0066cc",
                    "weight": 3,
                    "fillOpacity": 0.15,
                    "dashArray": "5, 5"
                },
                highlight_function=lambda x: {
                    "fillOpacity": 0.25
                },
                show=True
            ).add_to(benefits_map)
            
            st.info("🔗 **Using polygon from Flow Comparison tab**")
            st.caption("The blue dashed outline shows your selection from Tab 2")
        else:
            add_polygon_draw(benefits_map)
            st.caption("Draw a polygon to select and analyse specific features")
        
        # Render map
        map_output = st_folium(benefits_map, height=650, width=None, key="benefits_map")

    # Determine which polygon to use for filtering
    selection_geom = None
    
    if using_shared_polygon:
        selection_geom = geojson_to_shapely(st.session_state.shared_polygon_geojson)
    else:
        # Use locally drawn polygon
        if map_output and map_output.get("all_drawings"):
            selection_geom = geojson_to_shapely(
                map_output["all_drawings"][-1].get("geometry", {})
            )

    # Initialize selected DataFrame for report use
    selected = pd.DataFrame()

    with col_stats:
        st.subheader("Selection Summary")

        st.caption("**Original dataset** (all valid benefits):")
        st.metric("Total benefits", f"{original_total:,.2f}")

        st.caption(f"**On-map dataset** (|benefit| ≥ {abs_threshold:,.0f}):")
        st.metric("Total benefits", f"{map_total:,.2f}")

        st.markdown("---")

        if selection_geom is None:
            if using_shared_polygon:
                st.warning("⚠️ Applied polygon from Tab 2 is invalid or could not be loaded.")
            else:
                st.info("👆 Draw a polygon on the map to see selection totals.")
            st.metric("Selected features", 0)
            st.metric(f"Selected {value_col}", "—")
        else:
            # Show source of polygon
            if using_shared_polygon:
                st.success("✅ **Selection from Flow tab polygon**")
            
            selected = gdf_filtered[gdf_filtered.intersects(selection_geom)].copy()
            selected_count = len(selected)
            selected_total = float(selected[value_col].sum(skipna=True))

            st.metric("Selected features", f"{selected_count:,}")
            st.metric(f"Selected {value_col}", f"{selected_total:,.2f}")

            if show_table and selected_count > 0:
                st.markdown("---")
                st.markdown("### Selected Rows")

                display_cols = [c for c in ["__rowid__", value_col, geom_col] if c in selected.columns]
                extra_cols = [
                    c for c in selected.columns
                    if c not in display_cols and c not in ["geometry", abs_col, "abs_pct_change"]
                ]

                st.dataframe(selected[display_cols + extra_cols].head(500), use_container_width=True)

                if selected_count > 500:
                    st.caption(f"Showing first 500 of {selected_count:,} selected rows")
        
        # Option to clear shared polygon from Tab 3
        if using_shared_polygon:
            st.markdown("---")
            if st.button("🗑️ Clear applied polygon", key="clear_polygon_tab3"):
                st.session_state.shared_polygon_geojson = None
                st.session_state.apply_polygon_to_benefits = False
                st.rerun()

    # =========================================================================
    # HTML REPORT EXPORT SECTION - BENEFITS TAB
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 Export Benefits Analysis Report")
    st.caption("Generate a standalone HTML report with interactive map, metrics, and selection summary")

    if st.button("📄 Generate Benefits Report", key="generate_benefits_report", type="primary"):
        with st.spinner("Generating HTML report..."):
            # Determine selection data
            selection_data = None
            polygon_source_text = None
            
            if selection_geom is not None and len(selected) > 0:
                selected_count_report = len(selected)
                selected_total_report = float(selected[value_col].sum(skipna=True))
                polygon_source_text = "Flow Comparison Tab (imported)" if using_shared_polygon else "Drawn directly on Benefits Map"
                
                selection_data = {
                    "count": f"{selected_count_report:,}",
                    "total": f"{selected_total_report:,.2f}",
                    "polygon_source": polygon_source_text
                }
            
            # Prepare table
            if selection_geom is not None and len(selected) > 0:
                display_cols = [c for c in ["__rowid__", value_col] if c in selected.columns]
                table_html = selected.sort_values(abs_col, ascending=False)[display_cols].head(100).to_html(index=False, classes="table", float_format="%.2f") # type: ignore
                table_note_text = f"Showing top 100 of {len(selected):,} selected features (sorted by absolute benefit)" if len(selected) > 100 else f"Showing all {len(selected):,} selected features"
            else:
                display_cols = [c for c in ["__rowid__", value_col] if c in gdf_filtered.columns]
                table_html = gdf_filtered.sort_values(abs_col, ascending=False)[display_cols].head(100).to_html(index=False, classes="table", float_format="%.2f")
                table_note_text = f"Showing top 100 of {map_count:,} features on map (sorted by absolute benefit)" if map_count > 100 else f"Showing all {map_count:,} features on map"
            
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_type": "Benefits Analysis" + (" (with polygon selection)" if selection_geom else ""),
                "config": {
                    "Benefits Column": value_col,
                    "Absolute Threshold": f"≥ {abs_threshold:,.0f}",
                    "Source CRS": st.session_state.get('source_crs', DEFAULT_SOURCE_CRS),
                    "Target CRS": DEFAULT_TARGET_CRS,
                    "Geometry Column": geom_col,
                    "Features on Map": f"{map_count:,}"
                },
                "metrics": [
                    {"label": "Original Dataset Links", "value": f"{original_count:,}"},
                    {"label": "Original Total Benefits", "value": f"{original_total:,.2f}"},
                    {"label": "Map Dataset Links", "value": f"{map_count:,}"},
                    {"label": "Map Total Benefits", "value": f"{map_total:,.2f}"}
                ],
                "map_html": create_report_map(gdf_filtered, value_col, "Benefits Analysis", is_flow_map=False),
                "map_caption": f"Displaying {map_count:,} links with |{value_col}| ≥ {abs_threshold:,.0f}" + (f". Polygon selection contains {len(selected):,} features." if selection_geom and len(selected) > 0 else ""),
                "table_data": table_html,
                "table_note": table_note_text,
                "selection_info": selection_data
            }
            
            html_report = generate_html_report(report_data)
            
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_report,
                file_name=f"COBALT_Benefits_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_benefits_report"
            )
            st.success("✅ Report generated! Click the download button above to save the HTML file.")
            st.info("💡 The downloaded HTML file works offline and includes an interactive map with pan/zoom capabilities.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables for polygon sharing."""
    if "shared_polygon_geojson" not in st.session_state:
        st.session_state.shared_polygon_geojson = None
    if "apply_polygon_to_benefits" not in st.session_state:
        st.session_state.apply_polygon_to_benefits = False


def main():
    """Main application entry point."""
    st.set_page_config(page_title="COBALT Benefits Analysis", layout="wide")
    st.title("🚦 COBALT Benefits Analysis Dashboard")

    # Initialize state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")

        st.subheader("Flow Comparison (Tab 2)")
        flow_file = st.file_uploader(
            "Upload Flow CSV",
            type=["csv"],
            help="CSV containing DM/DS flow data with LineString geometry",
            key="flow_uploader"
        )

        st.subheader("Benefits Analysis (Tab 3)")
        benefits_file = st.file_uploader(
            "Upload Benefits CSV",
            type=["csv"],
            help="CSV containing benefits data with LineString geometry",
            key="benefits_uploader"
        )

        st.markdown("---")
        st.header("⚙️ Global Configuration")

        geom_col = st.text_input(
            "Geometry column name",
            value=DEFAULT_GEOM_COL,
            help="Column containing LineString coordinates (must be same in both files)"
        )

        source_crs = st.text_input(
            "Source coordinate system",
            value=DEFAULT_SOURCE_CRS,
            help="CRS of geometry coordinates (e.g., EPSG:27700 for UK grid)"
        )
        
        # Store source_crs in session state for report access
        st.session_state.source_crs = source_crs

        tab_sidebar = st.container()

    # Load data
    gdf_flow = None
    gdf_benefits = None

    if flow_file is not None:
        try:
            gdf_flow = load_geodataframe(flow_file.getvalue(), flow_file.name, geom_col, source_crs)
            st.sidebar.success(f"✅ Flow: Loaded **{len(gdf_flow):,}** features")
        except Exception as e:
            st.sidebar.error(f"❌ Flow file error:\n\n{e}")

    if benefits_file is not None:
        try:
            gdf_benefits = load_geodataframe(benefits_file.getvalue(), benefits_file.name, geom_col, source_crs)
            st.sidebar.success(f"✅ Benefits: Loaded **{len(gdf_benefits):,}** features")
        except Exception as e:
            st.sidebar.error(f"❌ Benefits file error:\n\n{e}")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📘 Introduction",
        "🔄 DM vs DS Flow Comparison",
        "💰 Benefits Analysis"
    ])

    with tab1:
        render_intro_tab()
        if flow_file is None and benefits_file is None:
            st.warning("⬅️ Please upload CSV files using the sidebar to access analysis tabs")

    with tab2:
        if flow_file is None:
            st.info("⬅️ Upload a Flow CSV file in the sidebar to view flow comparison analysis")
        elif gdf_flow is None:
            st.error("❌ Flow data failed to load. Check error message in sidebar.")
        else:
            render_flow_tab(gdf_flow, tab_sidebar)

    with tab3:
        if benefits_file is None:
            st.info("⬅️ Upload a Benefits CSV file in the sidebar to view benefits analysis")
        elif gdf_benefits is None:
            st.error("❌ Benefits data failed to load. Check error message in sidebar.")
        else:
            render_benefits_tab(gdf_benefits, geom_col, tab_sidebar)


if __name__ == "__main__":
    main()