"""Tests for aggregating by plain string columns (``by``/``count_cat``).

Plain (non-categorical) string columns should be accepted directly, with their
categories sorted lexicographically.  Missing values are aggregated into a
trailing ``unclassified`` bucket rather than being merged into a real category.
"""
import numpy as np
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf


def _categorical_dataframe():
    return pd.DataFrame({
        "x": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        "ch": ["web", "app", "web", None, "app", "web"],
    })


CANVAS = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))


def test_count_cat_string_column_with_missing():
    df = _categorical_dataframe()
    agg = CANVAS.points(df, "x", "y", ds.count_cat("ch"))

    assert agg.shape == (2, 2, 3)
    assert [str(v) for v in agg.coords["ch"].values] == ["app", "web", "unclassified"]

    expected = np.array([
        [[1, 2, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1]],
    ], dtype=np.uint32)
    np.testing.assert_array_equal(agg.values, expected)

    # The sum of all category buckets equals the plain count.
    count = CANVAS.points(df, "x", "y", ds.count())
    np.testing.assert_array_equal(agg.sum(dim="ch").values, count.values.astype(np.uint64))


def test_by_string_column_with_missing():
    df = _categorical_dataframe()
    agg = CANVAS.points(df, "x", "y", ds.by("ch"))

    assert agg.shape == (2, 2, 3)
    assert [str(v) for v in agg.coords["ch"].values] == ["app", "web", "unclassified"]

    expected = np.array([
        [[1, 2, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1]],
    ], dtype=np.uint32)
    np.testing.assert_array_equal(agg.values, expected)


def test_string_column_without_missing_has_no_unclassified():
    df = pd.DataFrame({
        "x": [0.0, 0.0, 1.0, 1.0],
        "y": [0.0, 1.0, 0.0, 1.0],
        "ch": ["b", "a", "b", "a"],
    })
    agg = CANVAS.points(df, "x", "y", ds.by("ch"))

    assert agg.shape == (2, 2, 2)
    assert [str(v) for v in agg.coords["ch"].values] == ["a", "b"]
    np.testing.assert_array_equal(agg.values, np.array([
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
    ], dtype=np.uint32))


def test_categorical_without_missing_is_unchanged():
    df = _categorical_dataframe().dropna(subset=["ch"])
    df["ch"] = df["ch"].astype("category")

    agg = CANVAS.points(df, "x", "y", ds.by("ch"))

    # No trailing unclassified bucket for data with no missing values.
    assert agg.shape == (2, 2, 2)
    assert [str(v) for v in agg.coords["ch"].values] == ["app", "web"]


def test_shade_unclassified_default_color():
    df = _categorical_dataframe()
    agg = CANVAS.points(df, "x", "y", ds.by("ch"))

    # A color_key covering only the real categories must be accepted; the
    # unclassified bucket gets a stable default color.
    img = tf.shade(agg, color_key={"app": "red", "web": "blue"})
    rgba = img.data.view(np.uint8).reshape((*img.shape, 4))

    expected = np.array([
        [[85, 0, 170, 255], [0, 0, 255, 40]],
        [[255, 0, 0, 40], [128, 128, 128, 40]],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(rgba, expected)


def test_shade_unclassified_explicit_color():
    df = _categorical_dataframe()
    agg = CANVAS.points(df, "x", "y", ds.by("ch"))

    img = tf.shade(agg, color_key={"app": "red", "web": "blue", "unclassified": "green"})
    rgba = img.data.view(np.uint8).reshape((*img.shape, 4))

    expected = np.array([
        [[85, 0, 170, 255], [0, 0, 255, 40]],
        [[255, 0, 0, 40], [0, 128, 0, 40]],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(rgba, expected)
