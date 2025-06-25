# -*- coding: utf-8 -*-
"""Test Suite for Functions in `moving` Module in `array` package.

This module provides comprehensive test coverage for functions in
`array.moving` module, which are processing moving operations over NumPy arrays.

The test suite covers the following aspects:
- Basic functionality and edge cases
- Parameter validation and error handling
- Different data types (int, float) and special values (NaN)
- Array length handling (empty, single element, shorter than offset)
- Integration tests with real-world scenarios

Functions Under Test:
    change: Calculates moving absolute changes between array elements
    pct_change: Calculates moving percentage changes between array elements

Test Classes:
    TestChangeFunction: Comprehensive tests for the change function
    TestPctChangeFunction: Comprehensive tests for the pct_change function
    TestIntegrationTests: Integration tests for both functions

Created on Wed Jun 25 16:08:33 2025

@author: WaNiNi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from wutils.array.moving import change, pct_change


class TestChangeFunction:
    """Test cases for the change function"""

    def test_basic_functionality(self):
        """Test basic functionality with example from docstring"""
        data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
        result = change(data, 2)
        expected = np.array([1., 2., 3., 5., 8., 13., 21., 34.])
        assert_array_equal(result, expected)

    def test_offset_one_equivalent_to_diff(self):
        """Test that offset=1 is equivalent to np.diff"""
        data = np.array([1, 3, 6, 10, 15])
        result = change(data, 1)
        expected = np.diff(data)
        assert_array_equal(result, expected)

    def test_keep_length_true(self):
        """Test keep_length=True functionality"""
        data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
        result = change(data, 2, keep_length=True)
        expected = np.array([
            np.nan, np.nan, 1., 2., 3., 5., 8., 13., 21., 34.
        ])

        # Check first two values are NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Check remaining values
        assert_array_equal(result[2:], expected[2:])

    def test_different_dtypes(self):
        """Test with different data types"""
        # Integer type
        data_int = np.array([1, 2, 3, 4, 5], dtype=int)
        result_int = change(data_int, 1)
        expected_int = np.array([1, 1, 1, 1])
        assert_array_equal(result_int, expected_int)

        # Float type
        data_float = np.array([1.5, 2.7, 4.1, 6.8], dtype=float)
        result_float = change(data_float, 1)
        expected_float = np.array([1.2, 1.4, 2.7])
        assert_array_almost_equal(result_float, expected_float)

    def test_large_offset(self):
        """Test with larger offset values"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = change(data, 5)
        # [6-1, 7-2, 8-3, 9-4, 10-5]
        expected = np.array([5, 5, 5, 5, 5])
        assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test with empty array"""
        data = np.array([])
        result = change(data, 1)
        assert len(result) == 0

        # Test keep_length=True case
        result_keep = change(data, 1, keep_length=True)
        assert len(result_keep) == 0

    def test_single_element_array(self):
        """Test with single element array"""
        data = np.array([5])
        result = change(data, 1)
        assert len(result) == 0

        # Test keep_length=True case
        result_keep = change(data, 1, keep_length=True)
        assert len(result_keep) == 1
        assert np.isnan(result_keep[0])

    def test_array_shorter_than_offset(self):
        """Test when array length is less than offset"""
        data = np.array([1, 2])
        result = change(data, 5)
        assert len(result) == 0

        # Test keep_length=True case
        result_keep = change(data, 5, keep_length=True)
        assert len(result_keep) == 2
        assert np.isnan(result_keep[0])
        assert np.isnan(result_keep[1])

    def test_invalid_offset_zero(self):
        """Test invalid offset=0"""
        data = np.array([1, 2, 3])
        with pytest.raises(
            ValueError,
            match="Parameter 'offset' must be a positive integer"
        ):
            change(data, 0)

    def test_invalid_offset_negative(self):
        """Test invalid negative offset"""
        data = np.array([1, 2, 3])
        with pytest.raises(
            ValueError,
            match="Parameter 'offset' must be a positive integer"
        ):
            change(data, -1)

    def test_invalid_offset_float(self):
        """Test invalid float offset"""
        data = np.array([1, 2, 3])
        with pytest.raises(
            ValueError,
            match="Parameter 'offset' must be a positive integer"
        ):
            change(data, 1.5)

    def test_with_nan_values(self):
        """Test array containing NaN values"""
        data = np.array([1, np.nan, 3, 4, 5])
        result = change(data, 1)

        # Result should contain NaN
        assert np.isnan(result[0])  # nan - 1
        assert np.isnan(result[1])  # 3 - nan
        assert_array_equal(result[2:], [1, 1])


class TestPctChangeFunction:
    """Test cases for the pct_change function"""

    def test_basic_functionality(self):
        """Test basic functionality with example from docstring"""
        data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
        result = pct_change(data, 2)
        expected = np.array([1., 2., 1.5, 1.66666667,
                             1.6, 1.625, 1.61538462, 1.61904762])
        assert_array_almost_equal(result, expected)

    def test_keep_length_true(self):
        """Test keep_length=True functionality"""
        data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
        result = pct_change(data, 2, keep_length=True)
        expected = np.array([np.nan, np.nan, 1., 2., 1.5,
                             1.66666667, 1.6, 1.625, 1.61538462, 1.61904762])

        # Check first two values are NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Check remaining values
        assert_array_almost_equal(result[2:], expected[2:])

    def test_simple_percentage_change(self):
        """Test simple percentage change calculation"""
        data = np.array([100, 110, 121, 133.1], dtype=float)
        result = pct_change(data, 1)
        expected = np.array([0.1, 0.1, 0.1])  # 10% growth
        assert_array_almost_equal(result, expected)

    def test_division_by_zero(self):
        """Test division by zero handling"""
        data = np.array([0, 5, 0, 10], dtype=float)
        result = pct_change(data, 1)

        # 5/0 - 1 should be NaN
        assert np.isnan(result[0])
        # 0/5 - 1 = -1
        assert result[1] == -1
        # 10/0 - 1 should be NaN
        assert np.isnan(result[2])

    def test_with_negative_values(self):
        """Test with negative values"""
        data = np.array([-2, 4, -1, 2], dtype=float)
        result = pct_change(data, 1)
        # expected: (4/-2)-1->np.nan, (-1/-4)-1=-1.25, (2/-1)-1->np.nan
        expected = np.array([np.nan, -1.25, np.nan])
        assert_array_almost_equal(result, expected)

    def test_offset_one(self):
        """Test with offset=1"""
        data = np.array([1, 2, 4, 8], dtype=float)
        result = pct_change(data, 1)
        expected = np.array([1., 1., 1.])  # 100% growth
        assert_array_almost_equal(result, expected)

    def test_empty_array(self):
        """Test with empty array"""
        data = np.array([])
        result = pct_change(data, 1)
        assert len(result) == 0

        # Test keep_length=True case
        result_keep = pct_change(data, 1, keep_length=True)
        assert len(result_keep) == 0

    def test_single_element_array(self):
        """Test with single element array"""
        data = np.array([5.0])
        result = pct_change(data, 1)
        assert len(result) == 0

        # Test keep_length=True case
        result_keep = pct_change(data, 1, keep_length=True)
        assert len(result_keep) == 1
        assert np.isnan(result_keep[0])

    def test_array_shorter_than_offset(self):
        """Test when array length is less than offset"""
        data = np.array([1.0, 2.0])
        result = pct_change(data, 5)
        assert len(result) == 0
        assert result.dtype == float

        # Test keep_length=True case
        result_keep = pct_change(data, 5, keep_length=True)
        assert len(result_keep) == 2
        assert np.isnan(result_keep[0])
        assert np.isnan(result_keep[1])

    def test_invalid_offset_zero(self):
        """Test invalid offset=0"""
        data = np.array([1.0, 2.0, 3.0])
        expected_msg = "Parameter 'offset' must be a positive integer"
        with pytest.raises(ValueError, match=expected_msg):
            pct_change(data, 0)

    def test_invalid_offset_negative(self):
        """Test invalid negative offset"""
        data = np.array([1.0, 2.0, 3.0])
        expected_msg = "Parameter 'offset' must be a positive integer"
        with pytest.raises(ValueError, match=expected_msg):
            pct_change(data, -1)

    def test_invalid_offset_float(self):
        """Test invalid float offset"""
        data = np.array([1.0, 2.0, 3.0])
        expected_msg = "Parameter 'offset' must be a positive integer"
        with pytest.raises(ValueError, match=expected_msg):
            pct_change(data, 1.5)

    def test_with_nan_values(self):
        """Test array containing NaN values"""
        data = np.array([1, np.nan, 3, 4], dtype=float)
        result = pct_change(data, 1)

        # Result should contain NaN
        assert np.isnan(result[0])  # (nan/1) - 1
        assert np.isnan(result[1])  # (3/nan) - 1
        assert_array_almost_equal(result[2:], [1 / 3])  # (4/3) - 1

    def test_large_offset(self):
        """Test with larger offset values"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = pct_change(data, 5)
        # expected: [6/1-1, 7/2-1, 8/3-1, 9/4-1, 10/5-1]
        expected = np.array([5., 2.5, 8 / 3 - 1, 1.25, 1.])
        assert_array_almost_equal(result, expected)

    def test_zero_to_positive(self):
        """Test change from zero to positive value"""
        data = np.array([0, 5], dtype=float)
        result = pct_change(data, 1)
        # 5/0 - 1 should be NaN (due to division by zero)
        assert np.isnan(result[0])

    def test_return_dtype_consistency(self):
        """Test return data type consistency"""
        data_int = np.array([1, 2, 3, 4])
        result = pct_change(data_int, 1)
        assert result.dtype == float

        data_float = np.array([1.0, 2.0, 3.0, 4.0])
        result = pct_change(data_float, 1)
        assert result.dtype == float


class TestIntegrationTests:
    """Integration tests for both functions"""

    def test_change_and_pct_change_consistency(self):
        """Test consistency between change and pct_change functions"""
        data = np.array([10, 15, 18, 20], dtype=float)

        # change(data, 1) should relate to data[:-1] * pct_change(data, 1)
        change_result = change(data, 1)
        pct_change_result = pct_change(data, 1)

        # Manual calculation for verification
        expected_change = np.array([5, 3, 2], dtype=float)
        expected_pct_change = np.array([0.5, 0.2, 1 / 9], dtype=float)

        assert_array_almost_equal(change_result, expected_change)
        assert_array_almost_equal(pct_change_result, expected_pct_change)

    def test_financial_data_example(self):
        """Test with financial-like data"""
        # Simulate stock price data
        prices = np.array([100, 105, 103, 108, 112, 110, 115], dtype=float)

        # Daily returns (1-day percentage change)
        daily_returns = pct_change(prices, 1)
        expected_returns = np.array([0.05, -0.019047619, 0.048543689,
                                     0.037037037, -0.017857143, 0.045454545])
        assert_array_almost_equal(daily_returns, expected_returns)

        # Absolute changes
        daily_changes = change(prices, 1)
        expected_changes = np.array([5, -2, 5, 4, -2, 5], dtype=float)
        assert_array_almost_equal(daily_changes, expected_changes)
