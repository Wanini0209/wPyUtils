# -*- coding: utf-8 -*-
"""Comprehensive test suite for `MovingWindow` class using pytest.

This module contains unit tests for the `MovingWindow` class and its associated
utility functions. The tests cover initialization, window creation, various
statistical operations, edge cases, and error handling.

Tests are organized into classes for better structure and readability.

Created on Wed Jun 25 14:27:19 2025

@author: WaNiNi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from wutils.array import MovingWindow


def _create_window_view(values, **kwargs):
    return MovingWindow(values, **kwargs).windows


class TestCreateWindowView:
    """Test cases for the _create_window_view utility function."""

    def test_basic_window_creation(self):
        """Test basic sliding window creation with default parameters."""
        values = np.array([1, 2, 3, 4, 5])
        result = _create_window_view(values, window_size=3)

        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ])
        assert_array_equal(result, expected)

    def test_window_with_step(self):
        """Test sliding window creation with step > 1 (dilated window)."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = _create_window_view(values, window_size=3, step=2)

        expected = np.array([
            [1, 3, 5],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8]
        ])
        assert_array_equal(result, expected)

    def test_window_with_align_right(self):
        """Test sliding window creation with align_right=True."""
        values = np.array([1, 2, 3, 4, 5, 6])
        result = _create_window_view(
            values, window_size=2, step=2, align_right=True
        )

        # With align_right=True and step=2, we slice values[1:]
        # So we work with [2, 3, 4, 5, 6]
        expected = np.array([
            [2, 4],
            [3, 5],
            [4, 6]
        ])
        assert_array_equal(result, expected)

    def test_insufficient_data(self):
        """Test window creation when there's insufficient data."""
        values = np.array([1, 2])
        result = _create_window_view(values, window_size=5)

        # Should return empty array with correct shape
        assert result.shape == (0, 5)
        assert result.dtype == values.dtype

    def test_multidimensional_array(self):
        """Test window creation with multidimensional input arrays."""
        values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = _create_window_view(values, window_size=2)

        expected = np.array([
            [[1, 2], [3, 4]],
            [[3, 4], [5, 6]],
            [[5, 6], [7, 8]]
        ])
        assert_array_equal(result, expected)

    def test_single_element_window(self):
        """Test window creation with window_size=1."""
        values = np.array([1, 2, 3, 4])
        result = _create_window_view(values, window_size=1)

        expected = np.array([[1], [2], [3], [4]])
        assert_array_equal(result, expected)


class TestMovingWindowInitialization:
    """Test cases for MovingWindow class initialization."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        assert mw.window_size == 3
        assert mw.step == 1
        assert mw.align_right is False
        assert mw._original_values is values

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=2, step=2, align_right=True)

        assert mw.window_size == 2
        assert mw.step == 2
        assert mw.align_right is True

    def test_invalid_values_type(self):
        """Test initialization with non-numpy array values."""
        expected_msg = "Parameter 'values' must be a NumPy array"
        with pytest.raises(TypeError, match=expected_msg):
            MovingWindow([1, 2, 3, 4], window_size=2)

    def test_invalid_window_size(self):
        """Test initialization with invalid window_size."""
        values = np.array([1, 2, 3, 4])
        expected_msg = "Parameter 'window_size' must be a positive integer"

        with pytest.raises(ValueError, match=expected_msg):
            MovingWindow(values, window_size=0)

        with pytest.raises(ValueError, match=expected_msg):
            MovingWindow(values, window_size=-1)

    def test_invalid_step(self):
        """Test initialization with invalid step."""
        values = np.array([1, 2, 3, 4])
        expected_msg = "Parameter 'step' must be a positive integer"

        with pytest.raises(ValueError, match=expected_msg):
            MovingWindow(values, window_size=2, step=0)

        with pytest.raises(ValueError, match=expected_msg):
            MovingWindow(values, window_size=2, step=-1)


class TestMovingWindowProperties:
    """Test cases for MovingWindow class properties."""

    def test_windows_property_lazy_creation(self):
        """Test that windows property creates windows lazily."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        # Initially, _windows should be None
        assert mw._windows is None

        # Accessing windows should create them
        windows = mw.windows
        assert mw._windows is not None
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        assert_array_equal(windows, expected)

    def test_windows_property_caching(self):
        """Test that windows property caches the result."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        windows1 = mw.windows
        windows2 = mw.windows

        # Should return the same object (cached)
        assert windows1 is windows2

    def test_shape_property(self):
        """Test the shape property."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        assert mw.shape == (3, 3)

    def test_n_windows_property(self):
        """Test the n_windows property."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        assert mw.n_windows == 3

    def test_properties_with_multidimensional_data(self):
        """Test properties with multidimensional input data."""
        values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mw = MovingWindow(values, window_size=2)

        assert mw.shape == (3, 2, 2)
        assert mw.n_windows == 3


class TestMovingWindowReduction:
    """Test cases for the reduce method and reduction operations."""

    def test_reduce_basic(self):
        """Test basic reduce functionality."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.reduce(np.sum)
        expected = np.array([6, 9, 12])  # [1+2+3, 2+3+4, 3+4+5]
        assert_array_equal(result, expected)

    def test_reduce_with_kwargs(self):
        """Test reduce with additional keyword arguments."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.reduce(np.std, ddof=1)
        # Standard deviation with ddof=1 for each window
        expected_windows = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        expected = np.std(expected_windows, axis=1, ddof=1)
        assert_array_almost_equal(result, expected)

    def test_reduce_keep_length(self):
        """Test reduce with keep_length=True."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.reduce(np.sum, keep_length=True)
        expected = np.array([np.nan, np.nan, 6, 9, 12])
        assert_array_almost_equal(result, expected)

    def test_reduce_keep_length_custom_fill(self):
        """Test reduce with keep_length=True and custom fill_value."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.reduce(np.sum, keep_length=True, fill_value=0)
        expected = np.array([0, 0, 6, 9, 12])
        assert_array_equal(result, expected)

    def test_reduce_empty_windows(self):
        """Test reduce when no windows can be formed."""
        values = np.array([1, 2])
        mw = MovingWindow(values, window_size=5)

        result = mw.reduce(np.sum)
        assert result.shape == (0,)
        # default `fill_value` is `np.nan`, which dtype is `np.float64`
        assert result.dtype == np.float64

    def test_reduce_empty_windows_keep_length(self):
        """Test reduce with empty windows and keep_length=True."""
        values = np.array([1, 2])
        mw = MovingWindow(values, window_size=5)

        result = mw.reduce(np.sum, keep_length=True, fill_value=-1)
        expected = np.array([-1, -1])
        assert_array_equal(result, expected)

    def test_reduce_multidimensional(self):
        """Test reduce with multidimensional data."""
        values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mw = MovingWindow(values, window_size=2)

        result = mw.reduce(np.sum)
        expected = np.array([[4, 6], [8, 10], [12, 14]])
        assert_array_equal(result, expected)


class TestMovingWindowStatistics:
    """Test cases for statistical methods."""

    def test_mean(self):
        """Test the mean method."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.mean()
        expected = np.array([2.0, 3.0, 4.0])
        assert_array_almost_equal(result, expected)

    def test_mean_keep_length(self):
        """Test the mean method with keep_length=True."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.mean(keep_length=True)
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
        assert_array_almost_equal(result, expected)

    def test_average_with_weights(self):
        """Test the average method with weights."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)
        weights = np.array([0.1, 0.2, 0.7])

        result = mw.average(weights=weights)
        expected = np.array([
            np.average([1, 2, 3], weights=weights),
            np.average([2, 3, 4], weights=weights),
            np.average([3, 4, 5], weights=weights)
        ])
        assert_array_almost_equal(result, expected)

    def test_max(self):
        """Test the max method."""
        values = np.array([1, 5, 2, 8, 3])
        mw = MovingWindow(values, window_size=3)

        result = mw.max()
        expected = np.array([5, 8, 8])
        assert_array_equal(result, expected)

    def test_min(self):
        """Test the min method."""
        values = np.array([1, 5, 2, 8, 3])
        mw = MovingWindow(values, window_size=3)

        result = mw.min()
        expected = np.array([1, 2, 2])
        assert_array_equal(result, expected)

    def test_std(self):
        """Test the std method."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.std()
        expected = np.array([
            np.std([1, 2, 3]),
            np.std([2, 3, 4]),
            np.std([3, 4, 5])
        ])
        assert_array_almost_equal(result, expected)

    def test_std_with_ddof(self):
        """Test the std method with ddof parameter."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.std(ddof=1)
        expected = np.array([
            np.std([1, 2, 3], ddof=1),
            np.std([2, 3, 4], ddof=1),
            np.std([3, 4, 5], ddof=1)
        ])
        assert_array_almost_equal(result, expected)

    def test_sum(self):
        """Test the sum method."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        result = mw.sum()
        expected = np.array([6, 9, 12])
        assert_array_equal(result, expected)

    def test_all_boolean(self):
        """Test the all method with boolean data."""
        values = np.array([True, True, False, True, True])
        mw = MovingWindow(values, window_size=3)

        result = mw.all()
        expected = np.array([False, False, False])
        assert_array_equal(result, expected)

    def test_all_with_allow_nan(self):
        """Test the all method with allow_nan=True."""
        values = np.array([True, True, False, True, True])
        mw = MovingWindow(values, window_size=3)

        result = mw.all(keep_length=True, allow_nan=True)
        expected = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_any_boolean(self):
        """Test the any method with boolean data."""
        values = np.array([False, False, True, False, False])
        mw = MovingWindow(values, window_size=3)

        result = mw.any()
        expected = np.array([True, True, True])
        assert_array_equal(result, expected)

    def test_any_with_allow_nan(self):
        """Test the any method with allow_nan=True."""
        values = np.array([False, False, True, False, False])
        mw = MovingWindow(values, window_size=3)

        result = mw.any(keep_length=True, allow_nan=True)
        expected = np.array([np.nan, np.nan, 1.0, 1.0, 1.0])
        assert_array_almost_equal(result, expected)


class TestMovingWindowEdgeCases:
    """Test cases for edge cases and special scenarios."""

    def test_single_element_array(self):
        """Test with single element array."""
        values = np.array([42])
        mw = MovingWindow(values, window_size=1)

        assert mw.n_windows == 1
        assert_array_equal(mw.windows, np.array([[42]]))
        assert_array_equal(mw.mean(), np.array([42.0]))

    def test_window_size_equals_array_length(self):
        """Test when window_size equals array length."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=5)

        assert mw.n_windows == 1
        assert_array_equal(mw.windows, np.array([[1, 2, 3, 4, 5]]))
        assert_array_equal(mw.mean(), np.array([3.0]))

    def test_large_step_size(self):
        """Test with large step size creating dilated windows."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mw = MovingWindow(values, window_size=3, step=3)

        result = mw.sum()
        # Windows: [1,4,7], [2,5,8], [3,6,9], [4,7,10]
        expected = np.array([12, 15, 18, 21])
        assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test with empty array."""
        values = np.array([])
        mw = MovingWindow(values, window_size=1)

        assert mw.n_windows == 0
        assert mw.windows.shape == (0, 1)
        assert mw.sum().shape == (0,)

    def test_float_array(self):
        """Test with floating point array."""
        values = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        mw = MovingWindow(values, window_size=3)

        result = mw.mean()
        expected = np.array([2.2, 3.3, 4.4])
        assert_array_almost_equal(result, expected)

    def test_integer_array(self):
        """Test with integer array."""
        values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        mw = MovingWindow(values, window_size=3)

        result = mw.sum()
        expected = np.array([6, 9, 12])
        assert_array_equal(result, expected)
        # `np.sum` over array with dtype, `np.int32` always return array with
        # dtype, `np.int64`
        assert result.dtype == np.int64


class TestMovingWindowChaining:
    """Test cases for method chaining and complex operations."""

    def test_method_chaining_concept(self):
        """Test that methods can be used in sequence (not actual chaining)."""
        values = np.array([1, 2, 3, 4, 5, 6])
        mw = MovingWindow(values, window_size=3)

        # Test multiple operations on the same MovingWindow instance
        mean_result = mw.mean()
        sum_result = mw.sum()
        std_result = mw.std()

        expected_mean = np.array([2.0, 3.0, 4.0, 5.0])
        expected_sum = np.array([6, 9, 12, 15])
        expected_std = np.array([
            np.std([1, 2, 3]), np.std([2, 3, 4]),
            np.std([3, 4, 5]), np.std([4, 5, 6])
        ])

        assert_array_almost_equal(mean_result, expected_mean)
        assert_array_equal(sum_result, expected_sum)
        assert_array_almost_equal(std_result, expected_std)

    def test_custom_reduction_function(self):
        """Test using custom reduction function with reduce method."""
        def custom_range(arr, axis=None):
            """Custom function that returns max - min."""
            return np.max(arr, axis=axis) - np.min(arr, axis=axis)

        values = np.array([1, 5, 2, 8, 3])
        mw = MovingWindow(values, window_size=3)

        result = mw.reduce(custom_range)
        expected = np.array([4, 6, 6])  # [5-1, 8-2, 8-2]
        assert_array_equal(result, expected)


class TestMovingWindowPerformance:
    """Test cases for performance-related aspects."""

    def test_memory_efficiency_strided_view(self):
        """Test that windows use strided views (memory efficient)."""
        values = np.array([1, 2, 3, 4, 5])
        mw = MovingWindow(values, window_size=3)

        # Check that it's a view, not a copy
        # When using np.lib.stride_tricks.as_strided, NumPy creates an
        # intermediate view object, so windows.base.base points to the original
        # array, `values`
        assert mw.windows.base.base is values

        # The windows should be based on the original array
        assert np.shares_memory(values, mw.windows)

    @pytest.mark.parametrize("array_size", [100, 1000, 5000])
    def test_large_arrays(self, array_size):
        """Test MovingWindow with large arrays."""
        values = np.arange(array_size, dtype=np.float64)
        mw = MovingWindow(values, window_size=10)

        result = mw.mean()

        # Basic sanity checks
        assert len(result) == array_size - 9
        assert not np.isnan(result).any()
        assert result[0] == np.mean(values[:10])
        assert result[-1] == np.mean(values[-10:])
