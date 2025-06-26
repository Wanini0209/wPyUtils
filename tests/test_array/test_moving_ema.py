# -*- coding: utf-8 -*-
"""Test Suite for EMA Functions in `moving` Module in `array` package.

This test module provides comprehensive coverage for the `exponential_smoothing`
and `ema` functions in `array.moving` module. The tests are designed to verify
correctness, validate parameter handling, and ensure robustness across various
scenarios.

The test suite covers the following aspects:
- Basic functionality and edge cases
- Parameter validation and error handling
- Different data types (int, float) and special values (NaN)
- Different dimension arrays (1D, 2D, and 3D)
- Array length handling (empty, single element, shorter than offset)
- Integration tests with real-world scenarios

Functions Under Test:
    exponential_smoothing: Calculates moving absolute changes between array elements
    ema: Calculates moving percentage changes between array elements

Test Classes:
    TestExponentialSmoothing: Tests for the `exponential_smoothing` function
    TestEMA: Tests for the `ema` function
    TestIntegrationTests: Integration tests for both functions

All major functionalities, including docstring examples, are tested to
ensure the functions perform as documented.

Created on Thu Jun 26 16:48:32 2025

@author: WaNiNi
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wutils.array.moving import ema, exponential_smoothing


class TestExponentialSmoothing:
    """Test cases for exponential_smoothing function."""

    def test_basic_1d_array(self):
        """Test exponential smoothing with basic 1D array."""
        values = np.array([1, 2, 3, 4, 5])
        alpha = 0.5
        result = exponential_smoothing(values, alpha)

        # Manual calculation for verification
        expected = np.array([
            1.0,  # alpha * 1 + (1-alpha) * 1 = 1
            1.5,  # alpha * 2 + (1-alpha) * 1 = 1.5
            2.25,  # alpha * 3 + (1-alpha) * 1.5 = 2.25
            3.125,  # alpha * 4 + (1-alpha) * 2.25 = 3.125
            4.0625  # alpha * 5 + (1-alpha) * 3.125 = 4.0625
        ])

        assert_allclose(result, expected, rtol=1e-10)

    def test_1d_with_custom_initial_value(self):
        """Test 1D array with custom initial value."""
        values = np.array([1, 2, 3, 4, 5])
        alpha = 0.5
        iv = 0.0
        result = exponential_smoothing(values, alpha, iv)

        # Manual calculation
        expected = np.array([
            0.5,    # alpha * 1 + (1-alpha) * 0 = 0.5
            1.25,   # alpha * 2 + (1-alpha) * 0.5 = 1.25
            2.125,  # alpha * 3 + (1-alpha) * 1.25 = 2.125
            3.0625,  # alpha * 4 + (1-alpha) * 2.125 = 3.0625
            4.03125  # alpha * 5 + (1-alpha) * 3.0625 = 4.03125
        ])

        assert_allclose(result, expected, rtol=1e-10)

    def test_2d_array(self):
        """Test exponential smoothing with 2D array."""
        values = np.array([[1, 2], [3, 4], [5, 6]])
        alpha = 0.5
        result = exponential_smoothing(values, alpha)

        # Expected: EMA computed along first axis
        # First row: [1, 2] (initial values)
        # Second row: alpha * [3, 4] + (1-alpha) * [1, 2] = [2, 3]
        # Third row: alpha * [5, 6] + (1-alpha) * [2, 3] = [3.5, 4.5]
        expected = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.5, 4.5]
        ])

        assert_allclose(result, expected, rtol=1e-10)

    def test_2d_with_custom_initial_value(self):
        """Test 2D array with custom initial value."""
        values = np.array([[1, 2], [3, 4], [5, 6]])
        alpha = 0.5
        iv = np.array([0, 1])
        result = exponential_smoothing(values, alpha, iv)

        # First row: alpha * [1, 2] + (1-alpha) * [0, 1] = [0.5, 1.5]
        # Second row: alpha * [3, 4] + (1-alpha) * [0.5, 1.5] = [1.75, 2.75]
        # Third row: alpha * [5, 6] + (1-alpha) * [1.75, 2.75] = [3.375, 4.375]
        expected = np.array([
            [0.5, 1.5],
            [1.75, 2.75],
            [3.375, 4.375]
        ])

        assert_allclose(result, expected, rtol=1e-10)

    def test_2d_with_scalar_initial_value(self):
        """Test 2D array with scalar initial value (broadcasted)."""
        values = np.array([[1, 2], [3, 4]])
        alpha = 0.5
        iv = 0.0
        result = exponential_smoothing(values, alpha, iv)

        # Scalar iv should be broadcasted to [0, 0]
        expected = np.array([
            [0.5, 1.0],  # alpha * [1, 2] + (1-alpha) * [0, 0]
            [1.75, 2.5]  # alpha * [3, 4] + (1-alpha) * [0.5, 1.0]
        ])

        assert_allclose(result, expected, rtol=1e-10)

    def test_3d_array(self):
        """Test exponential smoothing with 3D array."""
        values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        alpha = 0.5
        result = exponential_smoothing(values, alpha)

        # Shape should be preserved
        assert result.shape == values.shape

        # First "slice" should be the initial values
        assert_allclose(result[0], values[0], rtol=1e-10)

    def test_alpha_edge_cases(self):
        """Test alpha parameter edge cases."""
        values = np.array([1, 2, 3])

        # Alpha = 1 (no smoothing, just copy values)
        result = exponential_smoothing(values, 1.0)
        assert_allclose(result, values, rtol=1e-10)

        # Alpha close to 0 (heavy smoothing)
        alpha = 0.01
        result = exponential_smoothing(values, alpha)
        # Result should be very close to initial value
        assert abs(result[-1] - values[0]) < abs(values[-1] - values[0])

    def test_empty_array(self):
        """Test with empty array."""
        values = np.array([])
        result = exponential_smoothing(values, 0.5)
        assert result.size == 0
        assert result.shape == values.shape

    def test_single_element_array(self):
        """Test with single element array."""
        values = np.array([5.0])
        result = exponential_smoothing(values, 0.5)
        expected = np.array([5.0])
        assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_alpha_values(self):
        """Test invalid alpha parameter values."""
        values = np.array([1, 2, 3])

        # Alpha <= 0
        error_msg = "Parameter `alpha` must be in range"
        with pytest.raises(ValueError, match=error_msg):
            exponential_smoothing(values, 0.0)

        with pytest.raises(ValueError, match=error_msg):
            exponential_smoothing(values, -0.1)

        # Alpha > 1
        with pytest.raises(ValueError, match=error_msg):
            exponential_smoothing(values, 1.1)

    def test_docstring_examples(self):
        """Test examples from the docstring."""
        # Basic 1D example
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = exponential_smoothing(values, 1 / 3)
        expected = np.array([
            1., 1.33333333, 1.88888889, 2.59259259, 3.39506173,
            4.26337449, 5.17558299, 6.11705533, 7.07803688, 8.05202459])
        assert_allclose(result, expected, rtol=1e-7)

        # Multi-dimensional example
        values = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        result = exponential_smoothing(values, 1 / 3)
        expected = np.array([[1., 6.],
                             [1.33333333, 6.33333333],
                             [1.88888889, 6.88888889],
                             [2.59259259, 7.59259259],
                             [3.39506173, 8.39506173]])
        assert_allclose(result, expected, rtol=1e-7)

        # With specified initial value
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = exponential_smoothing(values, 1 / 3, iv=0)
        expected = np.array(
            [0.33333333, 0.88888889, 1.59259259, 2.39506173, 3.26337449,
             4.17558299, 5.11705533, 6.07803688, 7.05202459, 8.03468306])
        assert_allclose(result, expected, rtol=1e-7)


class TestEMA:
    """Test cases for ema function."""

    def test_basic_ema(self):
        """Test basic EMA calculation."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = ema(values, 3)

        # With period=3, factor=2: alpha = 2/(3+2-1) = 0.5
        # Initial value: mean of first 3 values = (1+2+3)/3 = 2
        # EMA starts from index 2 (period-1)
        expected_ema = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
        assert_allclose(result, expected_ema, rtol=1e-7)

    def test_ema_with_keep_length(self):
        """Test EMA with keep_length=True."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = ema(values, 3, keep_length=True)

        # First 2 values should be NaN, rest should be EMA values
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        expected_ema = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
        assert_allclose(result[2:], expected_ema, rtol=1e-7)

    def test_ema_with_custom_initial_value(self):
        """Test EMA with custom initial value."""
        values = np.array([1, 2, 3, 4, 5])
        result = ema(values, 3, iv=0)

        # With iv=0, calculation starts from index 0
        # alpha = 2/4 = 0.5
        alpha = 0.5
        expected = exponential_smoothing(values, alpha, iv=0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_ema_different_factor(self):
        """Test EMA with different factor values."""
        values = np.array([1, 2, 3, 4, 5, 6])

        # Test factor=1
        result = ema(values, 3, factor=1)
        # alpha = 1 / (3 + 1 - 1)  # = 1/3
        # iv = np.mean(values[:3])  # = 2
        # (2 * 2 + 4) / 3 = 8/3
        # (8/3 * 2 + 5) / 3 = 31/9
        # (31/9 * 2 + 6) / 3 = 116/27
        expected = np.array([2., 8 / 3, 31 / 9, 116 / 27])
        assert_allclose(result, expected, rtol=1e-10)

    def test_ema_2d_array(self):
        """Test EMA with 2D array."""
        values = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        result = ema(values, 2)

        # Should work along first axis
        assert result.shape[1] == values.shape[1]
        assert result.shape[0] == values.shape[0] - 1  # period - 1 elements removed

    def test_ema_insufficient_data(self):
        """Test EMA when data length < period."""
        values = np.array([1, 2])

        # Without keep_length
        result = ema(values, 3)
        assert result.size == 0

        # With keep_length
        result = ema(values, 3, keep_length=True)
        assert result.shape == values.shape
        assert np.all(np.isnan(result))

    def test_ema_exact_period_length(self):
        """Test EMA when data length equals period."""
        values = np.array([1, 2, 3])
        result = ema(values, 3)

        # Should return array with single element (the mean)
        expected = np.array([2.0])  # mean of [1, 2, 3]
        assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_period(self):
        """Test invalid period parameter."""
        values = np.array([1, 2, 3, 4, 5])
        error_msg = "Parameter `period` must be an integer > 1"
        # Period <= 1
        with pytest.raises(ValueError, match=error_msg):
            ema(values, 1)

        with pytest.raises(ValueError, match=error_msg):
            ema(values, 0)

        # Non-integer period
        with pytest.raises(ValueError, match=error_msg):
            ema(values, 2.5)

    def test_invalid_factor(self):
        """Test invalid factor parameter."""
        values = np.array([1, 2, 3, 4, 5])
        error_msg = "Parameter `factor` must be an integer >= 1"
        # Factor < 1
        with pytest.raises(ValueError, match=error_msg):
            ema(values, 3, factor=0)

        with pytest.raises(ValueError, match=error_msg):
            ema(values, 3, factor=-1)

        # Non-integer factor
        with pytest.raises(ValueError, match=error_msg):
            ema(values, 3, factor=1.5)

    def test_docstring_examples(self):
        """Test examples from the docstring."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Basic example
        result = ema(values, 3)
        expected = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
        assert_allclose(result, expected, rtol=1e-7)

        # With keep_length=True
        result = ema(values, 3, keep_length=True)
        expected_with_nan = np.array([np.nan, np.nan,
                                      2., 3., 4., 5., 6., 7., 8., 9.])
        assert_allclose(result[2:], expected_with_nan[2:], rtol=1e-7)
        assert np.isnan(result[0]) and np.isnan(result[1])

        # With specified initial value
        result = ema(values, 5, factor=1, iv=0)
        expected = np.array([0.2, 0.56, 1.048, 1.6384, 2.31072, 3.048576,
                             3.8388608, 4.67108864, 5.53687091, 6.42949673])
        assert_allclose(result, expected, rtol=1e-7)

    def test_consistency_with_exponential_smoothing(self):
        """
        Test that `ema` with iv parameter must be consistent with
        `exponential_smoothing`.
        """
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        period = 4
        factor = 3
        iv = 1.5

        alpha = factor / (period + factor - 1)

        # Both should give same result when iv is provided
        ema_result = ema(values, period, factor, iv=iv)
        exp_result = exponential_smoothing(values, alpha, iv=iv)

        assert_allclose(ema_result, exp_result, rtol=1e-10)


class TestIntegration:
    """Integration tests for both functions."""

    def test_numerical_stability(self):
        """Test numerical stability with large arrays."""
        np.random.seed(42)
        values = np.random.randn(10000)

        # Should not raise any errors
        result1 = exponential_smoothing(values, 0.1)
        result2 = ema(values, 20)

        # Results should be finite
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))

    def test_dtype_preservation(self):
        """Test that appropriate dtypes are returned."""
        # Float32 input
        values_f32 = np.array([1, 2, 3, 4], dtype=np.float32)
        result1 = exponential_smoothing(values_f32, 0.5)
        result2 = ema(values_f32, 2)

        # Should return float64 (as specified in the functions)
        assert result1.dtype == np.float64
        assert result2.dtype == np.float64

    def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # This is more of a smoke test - if there are memory leaks,
        # this test might fail in a memory-constrained environment
        for _ in range(100):
            values = np.random.randn(1000)
            _ = exponential_smoothing(values, 0.1)
            _ = ema(values, 10)
