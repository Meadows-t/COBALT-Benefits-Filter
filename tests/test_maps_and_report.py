import folium
from app import create_flow_map, create_benefits_map, create_report_map, generate_html_report


def test_flow_map_returns_folium_map(small_gdf):
    m = create_flow_map(small_gdf, "pct_change")
    assert isinstance(m, folium.Map)
    assert "Flow change" in m.get_root().render()


def test_benefits_map_empty_gdf(small_gdf):
    m = create_benefits_map(small_gdf.iloc[0:0], "Total Cost DIFF")
    assert isinstance(m, folium.Map)


def test_report_map_empty_message(small_gdf):
    assert "No data to display" in create_report_map(small_gdf.iloc[0:0], "pct_change", "x")


def test_report_map_returns_html(small_gdf):
    html = create_report_map(small_gdf, "Total Cost DIFF", "Benefits", is_flow_map=False)
    assert isinstance(html, str) and "<iframe" in html or "<div" in html


def test_generate_html_report_renders_sections():
    data = {
        "timestamp": "2026-11-25 12:00:00",
        "analysis_type": "Benefits Analysis",
        "config": {"Benefits Column": "Total Cost DIFF"},
        "metrics": [{"label": "Map Total Benefits", "value": "1,234.00"}],
        "map_html": "<div>map</div>",
        "map_caption": "caption here",
        "table_data": "<table><tr><td>42</td></tr></table>",
        "table_note": None,
        "selection_info": {"count": "3", "total": "999.00", "polygon_source": "Drawn"},
    }
    html = generate_html_report(data)
    assert "2026-11-25 12:00:00" in html
    assert "Total Cost DIFF" in html
    assert "<td>42</td>" in html                 # table rendered unescaped (| safe)
    assert "&lt;div&gt;map&lt;/div&gt;" in html  # map escaped inside srcdoc
    assert "Polygon Selection Summary" in html


def test_generate_html_report_hides_optional_sections():
    html = generate_html_report({"timestamp": "t", "analysis_type": "a", "config": {},
                                 "metrics": [], "map_html": None, "table_data": None,
                                 "selection_info": None})
    assert "Interactive Map" not in html
    assert "Polygon Selection Summary" not in html
