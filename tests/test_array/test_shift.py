# -*- coding: utf-8 -*-
"""
Test suite for the `_shift` module.

This module contains comprehensive tests for the `roll` and `shift` functions,
covering various scenarios including edge cases, data type handling,
and boundary conditions.

Test Classes
------------
TestRoll : Tests for the `roll` function
TestShift : Tests for the `shift` function

Created on Fri Jun 27 10:05:59 2025

@author: WaNiNi
"""

import numpy as np
import pytest

from wutils.array import roll, shift


class TestRoll:
    """Test cases for the roll function."""

    def test_roll_positive_shift(self):
        """Test rolling array elements with positive shift values."""
        values = np.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = roll(values, 2)
        expected = np.array([4, 5, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_roll_negative_shift(self):
        """Test rolling array elements with negative shift values."""
        values = np.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = roll(values, -2)
        expected = np.array([3, 4, 5, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_roll_zero_shift(self):
        """Test rolling with zero shift returns a copy."""
        values = np.arange(1, 6)
        result = roll(values, 0)

        # Should be equal but not the same object
        np.testing.assert_array_equal(result, values)
        assert result is not values  # Ensure it's a copy

    def test_roll_shift_greater_than_length(self):
        """Test rolling with shift value greater than array length."""
        values = np.arange(1, 4)  # [1, 2, 3]
        result = roll(values, 5)  # 5 % 3 = 2
        expected = np.array([2, 3, 1])
        np.testing.assert_array_equal(result, expected)

    def test_roll_shift_negative_greater_than_length(self):
        """Test rolling with negative shift greater than array length."""
        values = np.arange(1, 4)  # [1, 2, 3]
        result = roll(values, -5)  # -5 % 3 = 1
        expected = np.array([3, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_roll_empty_array(self):
        """Test rolling an empty array."""
        values = np.array([])
        result = roll(values, 3)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_roll_single_element(self):
        """Test rolling array with single element."""
        values = np.array([42])
        result = roll(values, 5)
        expected = np.array([42])
        np.testing.assert_array_equal(result, expected)

    def test_roll_2d_array(self):
        """Test rolling 2D array along first axis."""
        values = np.array([[1, 2], [3, 4], [5, 6]])
        result = roll(values, 1)
        expected = np.array([[5, 6], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_roll_preserves_dtype(self):
        """Test that rolling preserves the original data type."""
        values = np.array([1, 2, 3], dtype=np.int32)
        result = roll(values, 1)
        assert result.dtype == np.int32

    def test_roll_float_array(self):
        """Test rolling array with float values."""
        values = np.array([1.1, 2.2, 3.3, 4.4])
        result = roll(values, 2)
        expected = np.array([3.3, 4.4, 1.1, 2.2])
        np.testing.assert_array_almost_equal(result, expected)


class TestShift:
    """Test cases for the shift function."""

    def test_shift_positive_basic(self):
        """Test basic positive shift with default fill value."""
        values = np.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = shift(values, 2)
        expected = np.array([np.nan, np.nan, 1, 2, 3], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_shift_negative_basic(self):
        """Test basic negative shift with default fill value."""
        values = np.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = shift(values, -2)
        expected = np.array([3, 4, 5, np.nan, np.nan], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_shift_zero_returns_copy(self):
        """Test that zero shift returns a copy of the original array."""
        values = np.arange(1, 6)
        result = shift(values, 0)

        # Should be equal but not the same object
        np.testing.assert_array_equal(result, values)
        assert result is not values  # Ensure it's a copy

    def test_shift_custom_fill_value(self):
        """Test shift with custom fill value."""
        values = np.arange(1, 6)
        result = shift(values, 2, fill_value=-999)
        expected = np.array([-999, -999, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_shift_negative_custom_fill_value(self):
        """Test negative shift with custom fill value."""
        values = np.arange(1, 6)
        result = shift(values, -2, fill_value=0)
        expected = np.array([3, 4, 5, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_entire_array_positive(self):
        """Test shifting entire array (shift >= array length)."""
        values = np.arange(1, 4)  # [1, 2, 3]
        result = shift(values, 3, fill_value=0)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_entire_array_negative(self):
        """Test negative shift of entire array (abs(shift) >= length)."""
        values = np.arange(1, 4)  # [1, 2, 3]
        result = shift(values, -3, fill_value=0)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_greater_than_length(self):
        """Test shift value greater than array length."""
        values = np.arange(1, 4)  # [1, 2, 3]
        result = shift(values, 5, fill_value=-1)
        expected = np.array([-1, -1, -1])
        np.testing.assert_array_equal(result, expected)

    def test_shift_empty_array(self):
        """Test shifting an empty array."""
        values = np.array([])
        result = shift(values, 3, fill_value=0)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_shift_single_element(self):
        """Test shifting array with single element."""
        values = np.array([42])

        # Positive shift
        result_pos = shift(values, 1, fill_value=0)
        expected_pos = np.array([0])
        np.testing.assert_array_equal(result_pos, expected_pos)

        # Negative shift
        result_neg = shift(values, -1, fill_value=0)
        expected_neg = np.array([0])
        np.testing.assert_array_equal(result_neg, expected_neg)

        # Zero shift
        result_zero = shift(values, 0)
        np.testing.assert_array_equal(result_zero, values)

    def test_shift_dtype_promotion(self):
        """Test that dtype is promoted to accommodate fill_value."""
        values = np.array([1, 2, 3], dtype=np.int32)
        result = shift(values, 1, fill_value=1.5)

        # Should be promoted to float to accommodate the fill_value
        assert np.issubdtype(result.dtype, np.floating)
        expected = np.array([1.5, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shift_preserves_dtype_when_possible(self):
        """Test dtype preservation when fill_value is compatible."""
        values = np.array([1, 2, 3], dtype=np.int32)
        result = shift(values, 1, fill_value=0)

        # Should preserve int32 since fill_value is compatible
        assert result.dtype == np.int32
        expected = np.array([0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_shift_2d_array(self):
        """Test shifting 2D array along first axis."""
        values = np.array([[1, 2], [3, 4], [5, 6]])
        result = shift(values, 1, fill_value=0)
        expected = np.array([[0, 0], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_shift_string_array(self):
        """Test shifting array with string values."""
        values = np.array(['a', 'b', 'c'])
        result = shift(values, 1, fill_value='X')
        expected = np.array(['X', 'a', 'b'])
        np.testing.assert_array_equal(result, expected)

    def test_shift_boolean_array(self):
        """Test shifting boolean array."""
        values = np.array([True, False, True])
        result = shift(values, 1, fill_value=False)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_shift_complex_dtype(self):
        """Test shifting array with complex numbers."""
        values = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = shift(values, 1, fill_value=0 + 0j)
        expected = np.array([0 + 0j, 1 + 2j, 3 + 4j])
        np.testing.assert_array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases for both roll and shift functions."""

    def test_very_large_arrays(self):
        """Test functions with large arrays."""
        large_array = np.arange(10000)

        # Test roll
        rolled = roll(large_array, 100)
        assert len(rolled) == 10000
        assert rolled[0] == 9900  # First element after rolling

        # Test shift
        shifted = shift(large_array, 100, fill_value=-1)
        assert len(shifted) == 10000
        assert shifted[0] == -1  # Fill value
        assert shifted[100] == 0  # First original value

    def test_input_validation_types(self):
        """Test that functions handle various NumPy array types."""
        # Test with different dtypes
        dtypes_to_test = [np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64,
                          np.float32, np.float64]

        for dtype in dtypes_to_test:
            values = np.array([1, 2, 3, 4, 5], dtype=dtype)

            # Test roll
            rolled = roll(values, 1)
            assert rolled.dtype == dtype

            # Test shift with compatible fill_value
            if np.issubdtype(dtype, np.integer):
                fill_val = 0
            else:
                fill_val = 0.0

            shifted = shift(values, 1, fill_value=fill_val)
            assert shifted.dtype == dtype


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive scenario coverage."""

    @pytest.mark.parametrize("shift_amount", [-10, -3, -1, 0, 1, 3, 10])
    def test_roll_various_shifts(self, shift_amount):
        """Test roll function with various shift amounts."""
        values = np.arange(5)  # [0, 1, 2, 3, 4]
        result = roll(values, shift_amount)

        # Verify the result has the same length and elements
        assert len(result) == len(values)
        assert set(result) == set(values)

    @pytest.mark.parametrize("shift_amount", [-10, -3, -1, 0, 1, 3, 10])
    @pytest.mark.parametrize("fill_value", [0, -1, np.nan])
    def test_shift_various_parameters(self, shift_amount, fill_value):
        """Test shift function with various shift amounts and fill values."""
        values = np.arange(5)  # [0, 1, 2, 3, 4]
        result = shift(values, shift_amount, fill_value=fill_value)

        # Verify the result has the same length
        assert len(result) == len(values)

        # Check that appropriate positions contain the fill_value
        if shift_amount > 0:
            # For positive shift, first shift_amount elements should be fill
            fill_positions = min(shift_amount, len(values))
            if not np.isnan(fill_value):
                assert all(result[:fill_positions] == fill_value)
            else:
                assert all(np.isnan(result[:fill_positions]))
        elif shift_amount < 0:
            # For negative shift, last abs(shift_amount) elements should be fill
            fill_positions = min(abs(shift_amount), len(values))
            if not np.isnan(fill_value):
                assert all(result[-fill_positions:] == fill_value)
            else:
                assert all(np.isnan(result[-fill_positions:]))

    @pytest.mark.parametrize("array_size", [1, 2, 5, 10, 100])
    def test_functions_with_different_sizes(self, array_size):
        """Test both functions with arrays of different sizes."""
        values = np.arange(array_size)

        # Test roll
        rolled = roll(values, 1)
        assert len(rolled) == array_size

        # Test shift
        shifted = shift(values, 1, fill_value=-1)
        assert len(shifted) == array_size
