# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:01:23 2025

@author: WaNiNi
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from wutils.array import (
    moving_all,
    moving_any,
    moving_average,
    moving_change,
    moving_change_rate,
    moving_max,
    moving_min,
    moving_reduction,
    moving_sampling,
    moving_std,
    moving_sum,
)


class TestMovingSampling:
    """Test suite for the `moving_sampling` function."""

    def test_basic_sliding_window(self):
        """Test basic sliding window functionality with step=1."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = moving_sampling(values, 3)
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10]
        ])
        assert_equal(result, expected)

    def test_dilated_window_step_2(self):
        """Test dilated window with step=2."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = moving_sampling(values, 3, step=2)
        expected = np.array([
            [1, 3, 5],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8],
            [5, 7, 9],
            [6, 8, 10]
        ])
        assert_equal(result, expected)

    def test_multidimensional_array(self):
        """Test with multidimensional input array."""
        values = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]).T
        result = moving_sampling(values, 3, step=2)
        expected = np.array([
            [[1, 5], [3, 7], [5, 9]],
            [[2, 6], [4, 8], [6, 10]]
        ])
        assert_equal(result, expected)

    def test_left_open_sampling(self):
        """Test left-open sampling functionality."""
        values = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]).T
        result = moving_sampling(values, 3, step=2, left_open_sampling=True)
        expected = np.array([
            [[2, 6], [4, 8], [6, 10]]
        ])
        assert_equal(result, expected)

    def test_left_open_sampling_step_1(self):
        """Test left-open sampling with step=1 (should have no effect)."""
        values = np.array([1, 2, 3, 4, 5])
        result_normal = moving_sampling(values, 3, step=1,
                                        left_open_sampling=False)
        result_left_open = moving_sampling(values, 3, step=1,
                                           left_open_sampling=True)
        assert_equal(result_normal, result_left_open)

    def test_insufficient_data(self):
        """Test behavior when there's insufficient data to form a window."""
        values = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
        result = moving_sampling(values, 3, step=3)
        expected_shape = (0, 3, 2)
        assert result.shape == expected_shape
        assert result.dtype == values.dtype
        assert result.size == 0

    def test_single_element_array(self):
        """Test with single element array."""
        values = np.array([42])
        result = moving_sampling(values, 1)
        expected = np.array([[42]])
        assert_equal(result, expected)

    def test_single_window_possible(self):
        """Test when only one window can be formed."""
        values = np.array([1, 2, 3])
        result = moving_sampling(values, 3)
        expected = np.array([[1, 2, 3]])
        assert_equal(result, expected)

    def test_large_step_size(self):
        """Test with large step size."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = moving_sampling(values, 2, step=5)
        expected = np.array([
            [1, 6],
            [2, 7],
            [3, 8],
            [4, 9],
            [5, 10]
        ])
        assert_equal(result, expected)

    def test_samples_equal_to_array_length(self):
        """Test when samples equals the length of the input array."""
        values = np.array([1, 2, 3, 4, 5])
        result = moving_sampling(values, 5)
        expected = np.array([[1, 2, 3, 4, 5]])
        assert_equal(result, expected)

    def test_different_dtypes(self):
        """Test with different numpy data types."""
        # Test with float
        values_float = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result_float = moving_sampling(values_float, 3)
        expected_float = np.array([
            [1.1, 2.2, 3.3],
            [2.2, 3.3, 4.4],
            [3.3, 4.4, 5.5]
        ])
        assert_equal(result_float, expected_float)
        assert result_float.dtype == values_float.dtype

        # Test with complex numbers
        values_complex = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
        result_complex = moving_sampling(values_complex, 2)
        expected_complex = np.array([
            [1 + 1j, 2 + 2j],
            [2 + 2j, 3 + 3j],
            [3 + 3j, 4 + 4j]
        ])
        assert_equal(result_complex, expected_complex)
        assert result_complex.dtype == values_complex.dtype

    def test_3d_input_array(self):
        """Test with 3D input array."""
        values = np.random.rand(5, 3, 2)
        result = moving_sampling(values, 3, step=1)

        # Check shape
        # The expected shape is (n_windows, samples, original_dim1, original_dim2)
        expected_shape = (3, 3, 3, 2)
        assert result.shape == expected_shape

        # Check first window manually
        assert_equal(result[0], values[0:3])

        # Check last window manually
        assert_equal(result[2], values[2:5])

    def test_return_type_is_view(self):
        """Test that the function returns a view of the original array."""
        values = np.array([1, 2, 3, 4, 5, 6])
        result = moving_sampling(values, 3)

        # Modify original array
        values[0] = 999

        # Check if the view reflects the change
        assert result[0, 0] == 999

    # Parameter validation tests
    def test_invalid_samples_parameter(self):
        """Test error handling for invalid samples parameter."""
        values = np.array([1, 2, 3, 4, 5])
        error_msg = "Parameter 'samples' must be a positive integer."
        with pytest.raises(ValueError, match=error_msg):
            moving_sampling(values, 0)

        with pytest.raises(ValueError, match=error_msg):
            moving_sampling(values, -1)

    def test_invalid_step_parameter(self):
        """Test error handling for invalid step parameter."""
        values = np.array([1, 2, 3, 4, 5])
        error_msg = "Parameter 'step' must be a positive integer."
        with pytest.raises(ValueError, match=error_msg):
            moving_sampling(values, 3, step=0)

        with pytest.raises(ValueError, match=error_msg):
            moving_sampling(values, 3, step=-1)

    def test_empty_input_array(self):
        """Test behavior with empty input array."""
        values = np.array([])
        result = moving_sampling(values, 1)

        # Should return empty array with correct shape
        expected_shape = (0, 1)
        assert result.shape == expected_shape
        assert result.size == 0

    def test_edge_case_window_span_equals_array_length(self):
        """Test edge case where window span exactly equals array length."""
        values = np.array([1, 2, 3, 4, 5])
        # With samples=3 and step=2, window_span = 2*(3-1)+1 = 5
        result = moving_sampling(values, 3, step=2)
        expected = np.array([[1, 3, 5]])
        assert_equal(result, expected)

    def test_consistency_with_numpy_indexing(self):
        """Test that results are consistent with manual numpy indexing."""
        values = np.array([10, 20, 30, 40, 50, 60, 70])
        result = moving_sampling(values, 4, step=2)

        # Manually create expected result
        expected = []
        window_span = 2 * (4 - 1)  # step * (samples - 1)
        for i in range(len(values) - window_span):
            start, stop, step = i, i + 2 * 4, 2  # step=2, samples=4
            window = values[start:stop:step]
            expected.append(window)
        expected = np.array(expected)

        assert_equal(result, expected)

    def test_memory_efficiency(self):
        """Test that the function creates views rather than copies."""
        values = np.arange(1000)
        result = moving_sampling(values, 10)

        # When using np.lib.stride_tricks.as_strided, NumPy creates an
        # intermediate view object, so result.base.base points to the original
        # array
        assert result.base.base is values

        # Also verify it's truly a view by checking memory sharing
        assert np.shares_memory(result, values)

        # Test that modifying original affects the view
        original_value = values[5]
        values[5] = 9999
        # The change should be reflected in windows that include index 5
        assert 9999 in result[5]  # Window starting at index 5 includes values[5]

        # Restore original value
        values[5] = original_value


class TestMovingReduction:
    """Test suite for the `moving_reduction` function."""

    def test_basic_moving_mean(self):
        """Test basic moving average functionality."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_reduction(values, 3, np.mean)
        expected = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
        assert_almost_equal(result, expected)

    def test_moving_sum(self):
        """Test moving sum operation."""
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_reduction(values, 3, np.sum)
        expected = np.array([6., 9., 12.])  # [1+2+3, 2+3+4, 3+4+5]
        assert_almost_equal(result, expected)

    def test_moving_std(self):
        """Test moving standard deviation operation."""
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_reduction(values, 3, np.std)
        expected = np.array([np.std([1, 2, 3]),
                             np.std([2, 3, 4]),
                             np.std([3, 4, 5])])
        assert_almost_equal(result, expected)

    def test_moving_min_max(self):
        """Test moving min and max operations."""
        values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)

        # Test min
        result_min = moving_reduction(values, 3, np.min)
        expected_min = np.array([1., 1., 1., 1., 2., 2.])
        assert_almost_equal(result_min, expected_min)

        # Test max
        result_max = moving_reduction(values, 3, np.max)
        expected_max = np.array([4., 4., 5., 9., 9., 9.])
        assert_almost_equal(result_max, expected_max)

    def test_with_weights_average(self):
        """Test moving operation with additional kwargs (weighted average)."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        weights = np.array([0.2, 0.3, 0.5])
        result = moving_reduction(values, 3, np.average, weights=weights)
        expected = np.array([2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3])
        assert_almost_equal(result, expected)

    def test_with_step_parameter(self):
        """Test moving operation with step > 1."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_reduction(values, 3, np.mean, step=2)
        # Windows: [1,3,5], [2,4,6], [3,5,7], [4,6,8], [5,7,9], [6,8,10]
        expected = np.array([3., 4., 5., 6., 7., 8.])
        assert_almost_equal(result, expected)

    def test_left_open_sampling(self):
        """Test moving operation with left_open_sampling."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_reduction(values, 3, np.mean, step=2,
                                  left_open_sampling=True)
        expected = np.array([4., 5., 6., 7., 8.])
        assert_almost_equal(result, expected)

    def test_keep_size_true(self):
        """Test moving operation with keep_size=True."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_reduction(values, 3, np.mean, keep_size=True)
        expected = np.array([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.])

        # Check that the result has the same length as input
        assert len(result) == len(values)

        # Check non-NaN values
        assert_almost_equal(result[2:], expected[2:])

        # Check NaN values
        assert (~(np.isnan(result) ^ np.isnan(expected))).all()

    def test_keep_size_with_step(self):
        """Test keep_size=True with step > 1."""
        values = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        result = moving_reduction(values, 2, np.mean, step=2, keep_size=True)
        # With step=2, samples=2: windows are [1,3], [2,4], [3,5], [4,6]
        # Expected means: [2, 3, 4, 5]
        # With keep_size=True, should pad with 2 NaNs: [nan, nan, 2, 3, 4, 5]
        expected = np.array([np.nan, np.nan, 2., 3., 4., 5.])

        # Check that the result has the same length as input
        assert len(result) == len(values)

        # Check non-NaN values
        assert_almost_equal(result[2:], expected[2:])

        # Check NaN values
        assert (~(np.isnan(result) ^ np.isnan(expected))).all()

    def test_empty_input_array(self):
        """Test behavior with empty input array."""
        values = np.array([], dtype=float)
        result = moving_reduction(values, 3, np.mean)
        expected = np.array([], dtype=float)
        assert_equal(result, expected)
        assert result.dtype == float

    def test_insufficient_data_no_keep_size(self):
        """Test behavior when insufficient data to form windows."""
        values = np.array([1, 2], dtype=float)
        result = moving_reduction(values, 5, np.mean)
        expected = np.array([], dtype=float)
        assert_equal(result, expected)

    def test_insufficient_data_with_keep_size(self):
        """
        Test behavior when insufficient data to form windows
        with keep_size=`True`.
        """
        values = np.array([1, 2], dtype=float)
        result = moving_reduction(values, 5, np.mean, keep_size=True)
        expected = np.array([np.nan, np.nan])

        assert len(result) == len(values)
        assert (~(np.isnan(result) ^ np.isnan(expected))).all()

    def test_single_window_case(self):
        """Test when only one window can be formed."""
        values = np.array([1, 2, 3], dtype=float)
        result = moving_reduction(values, 3, np.mean)
        expected = np.array([2.])
        assert_almost_equal(result, expected)

    def test_single_window_with_keep_size(self):
        """Test single window case with keep_size=`True`."""
        values = np.array([1, 2, 3], dtype=float)
        result = moving_reduction(values, 3, np.mean, keep_size=True)
        expected = np.array([np.nan, np.nan, 2.])

        # Check that the result has the same length as input
        assert len(result) == len(values)

        # Check non-NaN values
        assert_almost_equal(result[2:], expected[2:])

        # Check NaN values
        assert (~(np.isnan(result) ^ np.isnan(expected))).all()

    def test_integer_input_with_keep_size(self):
        """
        Test that integer input is properly converted to float when
        keep_size=`True`.
        """
        values = np.array([1, 2, 3, 4, 5])  # Integer array
        result = moving_reduction(values, 3, np.mean, keep_size=True)
        expected = np.array([np.nan, np.nan, 2., 3., 4.])

        # Check that the result has the same length as input
        assert len(result) == len(values)

        # Check non-NaN values
        assert_almost_equal(result[2:], expected[2:])

        # Check NaN values
        assert (~(np.isnan(result) ^ np.isnan(expected))).all()

    def test_multidimensional_input(self):
        """Test with multidimensional input array."""
        # Create a 2D array where each row represents a different signal
        values = np.array([[1, 2, 3, 4, 5],
                          [2, 4, 6, 8, 10]], dtype=float).T  # Shape: (5, 2)

        result = moving_reduction(values, 3, np.mean)

        # Expected shape: (3, 2) - 3 windows, 2 signals
        expected_shape = (3, 2)
        assert result.shape == expected_shape

        # Check first signal (column 0):
        # [1,2,3] -> 2, [2,3,4] -> 3, [3,4,5] -> 4
        assert_almost_equal(result[:, 0], [2., 3., 4.])

        # Check second signal (column 1):
        # [2,4,6] -> 4, [4,6,8] -> 6, [6,8,10] -> 8
        assert_almost_equal(result[:, 1], [4., 6., 8.])

    def test_multidimensional_with_keep_size(self):
        """Test multidimensional input with keep_size=True."""
        values = np.array([[1, 2, 3, 4],
                          [2, 4, 6, 8]], dtype=float).T  # Shape: (4, 2)

        result = moving_reduction(values, 3, np.mean, keep_size=True)

        # Should maintain original shape
        assert result.shape == values.shape

        # First two rows should be NaN
        assert np.all(np.isnan(result[0, :]))
        assert np.all(np.isnan(result[1, :]))

        # Check computed values
        # Mean([1,2,3])=2 and Mean([2,4,6])=4
        assert_almost_equal(result[2, :], [2., 4.])
        # Mean([2,3,4])=3 and Mean([4,6,8])=6
        assert_almost_equal(result[3, :], [3., 6.])

    def test_custom_function_with_additional_params(self):
        """Test with custom function that accepts additional parameters."""
        values = np.array([1, 2, 3, 4, 5, 6], dtype=float)

        # Test with np.percentile which requires additional parameter 'q'
        # 50th percentile = median
        # Median of [1,2,3] is 2, Median of [2,3,4] is 3,
        # Median of [3,4,5] is 4, and Medain of [4,5,6] is 5
        result = moving_reduction(values, 3, np.percentile, q=50)
        expected = np.array([2., 3., 4., 5.])
        assert_almost_equal(result, expected)

    def test_function_parameter_validation(self):
        """
        Test that invalid parameters for `moving_sampling` are properly
        propagated.
        """
        values = np.array([1, 2, 3, 4, 5], dtype=float)

        # Test invalid samples parameter
        error_msg = "Parameter 'samples' must be a positive integer."
        with pytest.raises(ValueError, match=error_msg):
            moving_reduction(values, 0, np.mean)

        # Test invalid step parameter
        error_msg = "Parameter 'step' must be a positive integer."
        with pytest.raises(ValueError, match=error_msg):
            moving_reduction(values, 3, np.mean, step=0)

    def test_dtype_preservation_without_keep_size(self):
        """Test that dtype is preserved when keep_size=False."""
        values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = moving_reduction(values, 3, np.sum)

        # Sum of integers should remain integer when keep_size=False
        expected_dtype = np.int32 if np.sum(np.array([1, 2, 3], dtype=np.int32)
                                            ).dtype == np.int32 else int
        assert np.issubdtype(result.dtype, expected_dtype)

    def test_edge_case_single_element_window(self):
        """Test with window size of 1."""
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_reduction(values, 1, np.mean)

        # Each window contains only one element, so `mean` should equal
        # the element
        expected = values.copy()
        assert_almost_equal(result, expected)

    def test_function_that_changes_dtype(self):
        """
        Test with function that might change dtype (e.g., std of integers).
        """
        values = np.array([1, 2, 3, 4, 5, 6], dtype=int)
        result = moving_reduction(values, 3, np.std)

        # Standard deviation typically returns float even for integer input
        assert np.issubdtype(result.dtype, np.floating)

    def test_large_window_size(self):
        """Test with window size equal to array length."""
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_reduction(values, 5, np.mean)

        # Only one window possible: the entire array
        expected = np.array([3.])  # Mean of [1,2,3,4,5]
        assert_almost_equal(result, expected)

    def test_consistency_with_manual_calculation(self):
        """
        Test that results are consistent with manual sliding window calculations.
        """
        values = np.array([10, 20, 30, 40, 50], dtype=float)
        result = moving_reduction(values, 3, np.mean)

        # Manual calculation
        manual_result = []
        for i in range(len(values) - 2):  # 3-element windows
            window_mean = np.mean(values[i:i + 3])
            manual_result.append(window_mean)

        manual_result = np.array(manual_result)
        assert_almost_equal(result, manual_result)


class TestMovingAverage:
    """Tests for the `moving_average` function."""

    def test_simple_average(self):
        """Tests a basic moving average (unweighted)."""
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_average(sample_data, samples=3)
        expected = np.array([
            (1 + 2 + 6) / 3, (2 + 6 + 3) / 3,
            (6 + 3 + 4) / 3, (3 + 4 + 8) / 3])
        assert_allclose(result, expected)

    def test_weighted_average(self):
        """Tests a weighted moving average."""
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        weights = np.array([1, 2, 3])
        result = moving_average(sample_data, samples=3, weights=weights)
        expected = np.array([
            np.average([1, 2, 6], weights=weights),
            np.average([2, 6, 3], weights=weights),
            np.average([6, 3, 4], weights=weights),
            np.average([3, 4, 8], weights=weights),
        ])
        assert_allclose(result, expected)

    def test_average_with_keep_size(self):
        """Tests the keep_size=True case."""
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_average(sample_data, samples=3, keep_size=True)
        expected = np.array([np.nan, np.nan, 3.0, 11 / 3, 13 / 3, 5.0])
        assert_equal(result, expected)


class TestMovingMaxMin:
    """Tests for the `moving_max` and `moving_min` functions."""

    def test_moving_max(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_max(sample_data, samples=4)
        expected = np.array([6., 6., 8.])
        assert_equal(result, expected)

    def test_moving_min(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_min(sample_data, samples=4)
        expected = np.array([1., 2., 3.])
        assert_equal(result, expected)

    def test_max_with_step(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        """Ensures the 'step' parameter is passed correctly."""
        result = moving_max(sample_data, samples=3, step=2)
        # Windows are: [1, 6, 4], [2, 3, 8]
        expected = np.array([6., 8.])
        assert_equal(result, expected)


class TestMovingSum:
    """Tests for the `moving_sum` function."""

    def test_simple_sum(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_sum(sample_data, samples=3)
        expected = np.array([9., 11., 13., 15.])
        assert_allclose(result, expected)

    def test_sum_with_keep_size(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        result = moving_sum(sample_data, samples=2, keep_size=True)
        expected = np.array([np.nan, 3., 8., 9., 7., 12.])
        assert_equal(result, expected)


class TestMovingStd:
    """Tests for the `moving_std` function."""

    def test_population_std(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        """Tests population standard deviation (ddof=0)."""
        result = moving_std(sample_data, samples=4, ddof=0)
        expected = np.array([
            np.std([1, 2, 6, 3], ddof=0),
            np.std([2, 6, 3, 4], ddof=0),
            np.std([6, 3, 4, 8], ddof=0),
        ])
        assert_allclose(result, expected)

    def test_sample_std(self):
        sample_data = np.array([1., 2., 6., 3., 4., 8.])
        """Tests sample standard deviation (ddof=1)."""
        result = moving_std(sample_data, samples=4, ddof=1)
        expected = np.array([
            np.std([1, 2, 6, 3], ddof=1),
            np.std([2, 6, 3, 4], ddof=1),
            np.std([6, 3, 4, 8], ddof=1),
        ])
        assert_allclose(result, expected)


class TestMovingLogical:
    """
    Test suite for the `moving_all` and `moving_any` functions.
    """

    def test_moving_all_basic(self):
        """
        Tests the basic functionality of moving_all.
        It should return True only when all elements in the window are True.
        """
        # This array is designed to have only one window of all Trues
        data = np.array([True, True, False, True, True, True, False])

        # Windows: [T,T,F], [T,F,T], [F,T,T], [T,T,T], [T,T,F]
        result = moving_all(data, samples=3)
        expected = np.array([False, False, False, True, False])

        assert_equal(result, expected)

    def test_moving_all_with_keep_size(self):
        """
        Tests moving_all with the keep_size=True parameter.
        The output should be padded with False at the beginning.
        """
        data = np.array([True, True, False, True, True, True, False])

        result = moving_all(data, samples=3, keep_size=True)

        # Expected result from the basic test, prepended with two False values
        # (since a window of size 3 loses 2 elements at the start)
        expected = np.array([np.nan, np.nan, 0., 0., 0., 1., 0.])

        assert_equal(result, expected)

    def test_moving_any_basic(self):
        """
        Tests the basic functionality of moving_any.
        It should return False only when all elements in the window are False.
        """
        # This array is designed to have only one window of all Falses
        data = np.array([False, False, True, False, False, False, True])

        # Windows: [F,F,T], [F,T,F], [T,F,F], [F,F,F], [F,F,T]
        result = moving_any(data, samples=3)
        expected = np.array([True, True, True, False, True])

        assert_equal(result, expected)

    def test_moving_any_with_keep_size(self):
        """
        Tests moving_any with the keep_size=True parameter.
        The output should be padded with False at the beginning.
        """
        data = np.array([False, False, True, False, False, False, True])

        result = moving_any(data, samples=3, keep_size=True)

        # Expected result from the basic test, prepended with two False values
        expected = np.array([np.nan, np.nan, 1., 1., 1., 0., 1.])

        assert_equal(result, expected)

    def test_empty_input(self):
        """
        Ensures both functions return an empty array for empty input.
        """
        data = np.array([], dtype=bool)

        result_all = moving_all(data, samples=3)
        result_any = moving_any(data, samples=3)

        assert_equal(result_all, np.array([], dtype=bool))
        assert_equal(result_any, np.array([], dtype=bool))


class TestMovingChange:
    """
    Test suite for the `moving_change` and `moving_change_rate` functions.
    """

    def test_moving_change_offset_1(self):
        """
        Tests moving_change with offset=1, which should behave like np.diff.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        result = moving_change(sample_data, offset=1)
        expected = np.array([1., 2., 1., 2., 4.])  # [3-2, 5-3, 6-5, 8-6, 12-8]
        assert_allclose(result, expected)

    def test_moving_change_offset_2(self):
        """
        Tests moving_change with a larger offset.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        result = moving_change(sample_data, offset=2)
        expected = np.array([3., 3., 3., 6.])  # [5-2, 6-3, 8-5, 12-6]
        assert_allclose(result, expected)

    def test_moving_change_with_keep_size(self):
        """
        Tests moving_change with the keep_size=True parameter.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        result = moving_change(sample_data, offset=2, keep_size=True)
        # Expected result from offset=2 test, prepended with two NaNs
        expected = np.array([np.nan, np.nan, 3., 3., 3., 6.])
        assert_equal(result, expected)  # Use assert_equal for NaN comparison

    def test_moving_change_rate(self):
        """
        Tests the basic functionality of moving_change_rate.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        result = moving_change_rate(sample_data, offset=1)
        expected = np.array([
            (3 / 2) - 1,  # 0.5
            (5 / 3) - 1,  # 0.666...
            (6 / 5) - 1,  # 0.2
            (8 / 6) - 1,  # 0.333...
            (12 / 8) - 1  # 0.5
        ])
        assert_allclose(result, expected)

    def test_moving_change_rate_division_by_zero(self):
        """
        Tests that moving_change_rate correctly returns NaN on division by zero.
        """
        data = np.array([4., 2., 0., 6.])
        result = moving_change_rate(data, offset=2)
        # Expected: [(0/4)-1, (6/2)-1] -> This is wrong.
        # Windows are [4, 0] and [2, 6]
        # Calculations are (0/4)-1 = -1.0 and (6/2)-1 = 2.0
        # Let's use a better data array:
        data_with_zero = np.array([2., 4., 0., 3., 6.])
        result = moving_change_rate(data_with_zero, offset=2)
        # Calculations:
        # (0/2)-1 = -1.0
        # (3/4)-1 = -0.25
        # (6/0)-1 -> nan
        expected = np.array([-1.0, -0.25, np.nan])
        assert_equal(result, expected)

    def test_moving_change_rate_with_keep_size(self):
        """
        Tests moving_change_rate with the keep_size=True parameter.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        result = moving_change_rate(sample_data, offset=2, keep_size=True)
        # Expected: [nan, nan, (5/2)-1, (6/3)-1, (8/5)-1, (12/6)-1]
        expected = np.array([np.nan, np.nan, 1.5, 1.0, 0.6, 1.0])
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("func", [moving_change, moving_change_rate])
    def test_invalid_offset(self, func):
        """
        Ensures both functions raise ValueError for non-positive offsets.
        """
        sample_data = np.array([2., 3., 5., 6., 8., 12.])
        error_msg = "Parameter 'offset' must be positive."
        with pytest.raises(ValueError, match=error_msg):
            func(sample_data, offset=0)

        with pytest.raises(ValueError, match=error_msg):
            func(sample_data, offset=-1)
