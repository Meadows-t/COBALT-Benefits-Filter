import json
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, shape
from shapely import wkt
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import streamlit as st
import branca.colormap as cm


# -----------------------------
# Helpers
# -----------------------------
def parse_linestring(val):
    """
    Attempts to parse a LineString from several common CSV encodings:
      1) WKT: "LINESTRING (x y, x y, ...)"
      2) GeoJSON-like string: {"type":"LineString","coordinates":[[x,y],[x,y]]}
      3) Coordinate list-like string: "[[x,y],[x,y]]" or "[(x,y),(x,y)]"
    Returns shapely geometry or None.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    if isinstance(val, LineString):
        return val

    s = str(val).strip()
    if not s:
        return None

    # 1) WKT
    try:
        return wkt.loads(s)
    except Exception:
        pass

    # 2) GeoJSON string
    if s.startswith("{") and '"coordinates"' in s:
        try:
            obj = json.loads(s)
            return shape(obj)
        except Exception:
            pass

    # 3) List-ish coordinates (extract number pairs)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) >= 4 and len(nums) % 2 == 0:
        coords = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]
        try:
            return LineString(coords)
        except Exception:
            return None

    return None


def geojson_to_shapely(geom_dict):
    """Convert a GeoJSON Polygon/MultiPolygon dict (from Leaflet.Draw) to shapely geometry."""
    if not geom_dict:
        return None
    gtype = geom_dict.get("type", None)
    if gtype in ("Polygon", "MultiPolygon"):
        try:
            return shape(geom_dict)
        except Exception:
            return None
    return None


def make_map(gdf_4326, value_col, selected_idx=None):
    """
    Create a Folium map with line styling by *signed* value_col.
    If selected_idx provided, highlight those features in a thicker line.
    """
    if len(gdf_4326) == 0:
        m = folium.Map(location=[51.4545, -2.5879], zoom_start=12, tiles="CartoDB positron")
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
        ).add_to(m)
        return m

    bounds = gdf_4326.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    vals = pd.to_numeric(gdf_4326[value_col], errors="coerce")
    vmin = float(vals.min()) if vals.notna().any() else 0.0
    vmax = float(vals.max()) if vals.notna().any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    colormap = cm.LinearColormap(colors=["#2c7bb6", "#ffffbf", "#d7191c"], vmin=vmin, vmax=vmax)
    colormap.caption = f"{value_col} (Signed benefits)"
    colormap.add_to(m)

    selected_idx = set(selected_idx or [])

    def style_fn(feat):
        v = feat["properties"].get(value_col, None)
        try:
            v = float(v)
        except Exception:
            v = None

        is_sel = feat["properties"].get("__rowid__") in selected_idx
        color = colormap(v) if v is not None else "#888888"
        weight = 6 if is_sel else 3
        opacity = 0.9 if is_sel else 0.6
        return {"color": color, "weight": weight, "opacity": opacity}

    folium.GeoJson(
        data=gdf_4326.to_json(),
        name="Benefits lines",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["__rowid__", value_col],
            aliases=["Row", value_col],
            localize=True
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)

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
    ).add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Benefits Polygon Filter", layout="wide")
st.title("LineString Benefits Map (Polygon Filter)")

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

    st.markdown("---")
    st.header("Columns / CRS")
    geom_col = st.text_input("Geometry column (LineString coords)", value="Coords")
    value_col = st.text_input("Benefits column", value="Total Cost DIFF")

    # ✅ Updated default CRS
    st.caption("If coords are lon/lat, use EPSG:4326. If UK grid, often EPSG:27700.")
    source_crs = st.text_input("Source CRS (e.g., EPSG:4326 or EPSG:27700)", value="EPSG:27700")

    st.markdown("---")
    # ✅ Updated default threshold
    abs_threshold = st.number_input(
        "Min |benefit| to include on the map",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )
    show_table = st.checkbox("Show selected rows table", value=True)


@st.cache_data(show_spinner=True)
def load_data_from_upload(file_bytes: bytes, filename: str, geom_col: str, value_col: str, source_crs: str):
    """
    Cache keyed on file content + parameters.
    Streamlit file_uploader gives us bytes; we read via BytesIO.
    """
    from io import BytesIO

    df = pd.read_csv(BytesIO(file_bytes))

    if geom_col not in df.columns:
        raise ValueError(f"Geometry column '{geom_col}' not found. Columns: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"Benefits column '{value_col}' not found. Columns: {list(df.columns)}")

    df["geometry"] = df[geom_col].apply(parse_linestring)
    df = df[df["geometry"].notna()].copy()

    # Stable row id for highlighting
    df["__rowid__"] = df.index.astype(int)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=source_crs)

    # Convert to WGS84 for web mapping
    try:
        gdf_4326 = gdf.to_crs("EPSG:4326")
    except Exception as e:
        raise ValueError(
            f"Could not convert CRS from {source_crs} to EPSG:4326. "
            f"Check your source CRS. Underlying error: {e}"
        )

    # Signed benefits numeric
    gdf_4326[value_col] = pd.to_numeric(gdf_4326[value_col], errors="coerce")

    # Absolute benefits column for threshold filter only
    abs_col = f"ABS_{value_col}"
    gdf_4326[abs_col] = gdf_4326[value_col].abs()

    return gdf_4326, abs_col


if uploaded_file is None:
    st.info("⬅️ Upload a CSV file using the sidebar to begin.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    gdf_4326, abs_col = load_data_from_upload(file_bytes, uploaded_file.name, geom_col, value_col, source_crs)
except Exception as e:
    st.error(str(e))
    st.stop()

st.write(f"Loaded **{len(gdf_4326):,}** LineStrings with valid geometry from **{uploaded_file.name}**.")

# Original totals (signed only)
orig_total_signed = float(gdf_4326[value_col].sum(skipna=True))
orig_count = int(gdf_4326[value_col].notna().sum())

# Filter for on-map dataset using ABS threshold (keeps large negatives too)
gdf_plot = gdf_4326[gdf_4326[abs_col].notna() & (gdf_4326[abs_col] >= abs_threshold)].copy()
plot_total_signed = float(gdf_plot[value_col].sum(skipna=True))
plot_count = len(gdf_plot)

# Sidebar summary (signed totals only)
st.sidebar.markdown("---")
st.sidebar.subheader("Totals (Signed)")
st.sidebar.metric("Original links (valid benefits)", f"{orig_count:,}")
st.sidebar.metric("Original total benefits", f"{orig_total_signed:,.2f}")
st.sidebar.metric("Map links (after |benefit| threshold)", f"{plot_count:,}")
st.sidebar.metric("Map total benefits", f"{plot_total_signed:,.2f}")

st.sidebar.caption(
    "Threshold uses absolute value: keeps both large positives and large negatives "
    f"(e.g., ±{abs_threshold:,.0f} and above)."
)

# Layout
col_map, col_stats = st.columns([2.2, 1.0], gap="large")

with col_map:
    st.subheader("Map (Draw a polygon to filter)")
    m = make_map(gdf_plot, value_col)
    map_state = st_folium(m, height=650, width=None)

# Determine selection polygon from drawn polygon
selection_geom = None
if map_state and map_state.get("all_drawings"):
    last = map_state["all_drawings"][-1]
    geom = last.get("geometry", {})
    selection_geom = geojson_to_shapely(geom)

with col_stats:
    st.subheader("Selection Summary")

    st.caption("Original total (whole dataset, signed):")
    st.metric("Total benefits", f"{orig_total_signed:,.2f}")

    st.caption("On-map total (after |benefit| threshold, signed):")
    st.metric("Total benefits", f"{plot_total_signed:,.2f}")

    st.markdown("---")

    if selection_geom is None:
        st.info("Draw a polygon on the map to see selection totals.")
        st.metric("Selected features (on-map)", 0)
        st.metric(f"Selected sum {value_col} (signed)", "—")
    else:
        sel = gdf_plot[gdf_plot.intersects(selection_geom)].copy()
        sel_count = len(sel)

        # Signed selection sum
        sel_sum_signed = float(sel[value_col].sum(skipna=True))

        st.metric("Selected features (on-map)", f"{sel_count:,}")
        st.metric(f"Selected sum {value_col} (signed)", f"{sel_sum_signed:,.2f}")

        if show_table and sel_count > 0:
            st.markdown("### Selected rows")
            cols_to_show = [c for c in ["__rowid__", value_col, geom_col] if c in sel.columns]
            extra = [c for c in sel.columns if c not in cols_to_show and c not in ["geometry"]]
            st.dataframe(sel[cols_to_show + extra].head(500))

# Re-draw map with selection highlighted
if selection_geom is not None:
    selected_idx = set(gdf_plot[gdf_plot.intersects(selection_geom)]["__rowid__"].tolist())
    with col_map:
        st.subheader("Map (Selection highlighted)")
        m2 = make_map(gdf_plot, value_col, selected_idx=selected_idx)
        st_folium(m2, height=650, width=None)