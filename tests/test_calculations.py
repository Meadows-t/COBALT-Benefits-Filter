import math
import pandas as pd
import pytest
from shapely.geometry import LineString
from app import (calculate_opacity, calculate_pct_change, filter_by_abs_threshold,
                 compute_offsets, MIN_OPACITY, MAX_OPACITY, OPACITY_GAMMA)


# --- opacity ---
def test_opacity_bounds():
    assert calculate_opacity(0, 10) == pytest.approx(MIN_OPACITY)
    assert calculate_opacity(10, 10) == pytest.approx(MAX_OPACITY)
    assert calculate_opacity(-10, 10) == pytest.approx(MAX_OPACITY)   # uses abs
    assert calculate_opacity(50, 10) == pytest.approx(MAX_OPACITY)    # clipped


def test_opacity_midpoint():
    expected = MIN_OPACITY + 0.5 ** OPACITY_GAMMA * (MAX_OPACITY - MIN_OPACITY)
    assert calculate_opacity(5, 10) == pytest.approx(expected)


@pytest.mark.parametrize("value, max_abs", [(None, 10), (float("nan"), 10), (5, 0)])
def test_opacity_fallback(value, max_abs):
    assert calculate_opacity(value, max_abs) == 0.25


# --- % change ---
def test_pct_change_known_values():
    df = pd.DataFrame({"DM": [100, 200, 0, "x"], "DS": [120, 150, 50, 10]})
    out = calculate_pct_change(df, "DM", "DS")
    assert out["pct_change"].iloc[0] == pytest.approx(20.0)
    assert out["pct_change"].iloc[1] == pytest.approx(-25.0)
    assert math.isnan(out["pct_change"].iloc[2])   # DM = 0
    assert math.isnan(out["pct_change"].iloc[3])   # non-numeric
    assert out["abs_pct_change"].iloc[1] == pytest.approx(25.0)
    assert out["pct_change"].dtype == float


def test_pct_change_does_not_mutate_input():
    df = pd.DataFrame({"DM": ["100"], "DS": ["110"]})
    calculate_pct_change(df, "DM", "DS")
    assert df["DM"].iloc[0] == "100"


# --- threshold filter ---
def test_filter_keeps_positive_and_negative():
    df = pd.DataFrame({"v": [15000, -25000, 500, None]})
    out = filter_by_abs_threshold(df, "v", 10000)
    assert sorted(out["v"]) == [-25000, 15000]


def test_filter_threshold_is_inclusive():
    df = pd.DataFrame({"v": [10000, 9999.99]})
    assert len(filter_by_abs_threshold(df, "v", 10000)) == 1


# --- overlap offsets ---
def test_offsets_symmetric_for_duplicates():
    a = LineString([(0, 0), (1, 1)])
    b = LineString([(2, 2), (3, 3)])
    offsets = compute_offsets([a, a, b], step_px=6)
    assert list(offsets) == [-3.0, 3.0, 0.0]


def test_offsets_three_duplicates_and_base():
    a = LineString([(0, 0), (1, 1)])
    assert list(compute_offsets([a, a, a], step_px=6, base_px=2)) == [-4.0, 2.0, 8.0]


def test_offsets_disabled():
    a = LineString([(0, 0), (1, 1)])
    assert list(compute_offsets([a, a], step_px=6, base_px=1, offset_overlaps=False)) == [1.0, 1.0]
