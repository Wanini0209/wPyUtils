# -*- coding: utf-8 -*-
"""Tests for the _tunit module.

This module contains comprehensive tests for the time unit functionality,
including predefined time unit instances and symbol-based lookups.

Created on Fri Jun 27 16:34:14 2025

@author: WaNiNi
"""

import datetime

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from wutils.tseries import tunit

_EPOCH_YEAR = 1970


class TestTimeUnitInstances:
    """Tests for predefined TimeUnit instances."""

    def test_day_properties(self):
        """Test DAY unit properties."""
        assert tunit.DAY.noun == 'Day'
        assert tunit.DAY.adj == 'Daily'
        assert tunit.DAY.symbol == 'D'
        assert tunit.DAY.dtype == np.dtype('datetime64[D]')
        assert tunit.DAY.encoder is not None
        assert tunit.DAY.decoder is None

    def test_week_properties(self):
        """Test WEEK unit properties."""
        assert tunit.WEEK.noun == 'Week'
        assert tunit.WEEK.adj == 'Weekly'
        assert tunit.WEEK.symbol == 'W'
        assert tunit.WEEK.dtype == np.dtype('datetime64[D]')
        assert tunit.WEEK.encoder is not None
        assert tunit.WEEK.decoder is not None

    def test_month_properties(self):
        """Test MONTH unit properties."""
        assert tunit.MONTH.noun == 'Month'
        assert tunit.MONTH.adj == 'Monthly'
        assert tunit.MONTH.symbol == 'Mo'
        assert tunit.MONTH.dtype == np.dtype('datetime64[D]')
        assert tunit.MONTH.encoder is not None
        assert tunit.MONTH.decoder is not None

    def test_quarter_properties(self):
        """Test QUARTER unit properties."""
        assert tunit.QUARTER.noun == 'Quarter'
        assert tunit.QUARTER.adj == 'Quarterly'
        assert tunit.QUARTER.symbol == 'Q'
        assert tunit.QUARTER.dtype == np.dtype('datetime64[D]')
        assert tunit.QUARTER.encoder is not None
        assert tunit.QUARTER.decoder is not None

    def test_year_properties(self):
        """Test YEAR unit properties."""
        assert tunit.YEAR.noun == 'Year'
        assert tunit.YEAR.adj == 'Annual'
        assert tunit.YEAR.symbol == 'Y'
        assert tunit.YEAR.dtype == np.dtype('datetime64[D]')
        assert tunit.YEAR.encoder is not None
        assert tunit.YEAR.decoder is not None


class TestDayUnit:
    """Tests for DAY time unit encoding and decoding."""

    def test_day_encode_single_date(self):
        """Test encoding a single date to days since epoch."""
        date = np.array(['1970-01-01'], dtype='datetime64[D]')
        encoded = tunit.DAY.encode(date)
        assert encoded[0] == 0  # Unix epoch should be day 0

    def test_day_encode_multiple_dates(self):
        """Test encoding multiple dates."""
        dates = np.array(['1970-01-01', '1970-01-02', '1970-01-03'],
                         dtype='datetime64[D]')
        encoded = tunit.DAY.encode(dates)
        expected = np.array([0, 1, 2])
        assert_array_equal(encoded, expected)

    def test_day_encode_decode_roundtrip(self):
        """Test that encoding and decoding preserves original dates."""
        original_dates = np.array(['2023-01-01', '2023-06-15', '2023-12-31'],
                                  dtype='datetime64[D]')
        encoded = tunit.DAY.encode(original_dates)
        decoded = tunit.DAY.decode(encoded)
        assert_array_equal(decoded, original_dates)

    def test_day_decode_zero(self):
        """Test decoding day zero returns Unix epoch."""
        decoded = tunit.DAY.decode(np.array([0]))
        expected = np.array(['1970-01-01'], dtype='datetime64[D]')
        assert_array_equal(decoded, expected)


class TestWeekUnit:
    """Tests for WEEK time unit encoding and decoding."""

    def test_week_encode_epoch_week(self):
        """Test week encoding around Unix epoch."""
        # 1970-01-01 was a Thursday, so week 0 should start on
        # Monday 1969-12-29
        epoch_date = np.array(['1970-01-01'], dtype='datetime64[D]')
        encoded = tunit.WEEK.encode(epoch_date)
        assert encoded[0] == 0  # Should be in week 0

    def test_week_encode_monday_alignment(self):
        """Test that weeks are aligned to Monday."""
        # Test a known Monday to Sunday in the same week
        monday = datetime.date(1970, 1, 5)  # Monday of week 1
        monday2sunday = np.array(
            [monday + datetime.timedelta(i) for i in range(7)],
            dtype='datetime64[D]')  # From Monday to Sunday

        # Both should be in the same week
        assert (tunit.WEEK.encode(monday2sunday) == 1).all()

    def test_week_decode_to_monday(self):
        """Test that decoding returns the Monday of each week."""
        week_nums = np.array([0, 1, 2])
        decoded = tunit.WEEK.decode(week_nums)

        # Check that all decoded dates are Mondays
        # Monday is weekday 0 in ISO format
        iso_weekdays = [each.weekday() for each in decoded.tolist()]
        expected_weekdays = np.array([0, 0, 0])  # All Mondays
        assert_array_equal(iso_weekdays, expected_weekdays)

    def test_week_encode_decode_roundtrip(self):
        """Test week encoding/decoding preserves week boundaries."""
        # Use Mondays to ensure exact roundtrip
        mondays = np.array(['2023-01-02', '2023-01-09', '2023-01-16'],
                           dtype='datetime64[D]')
        encoded = tunit.WEEK.encode(mondays)
        decoded = tunit.WEEK.decode(encoded)
        assert_array_equal(decoded, mondays)


class TestMonthUnit:
    """Tests for MONTH time unit encoding and decoding."""

    def test_month_encode_epoch(self):
        """Test month encoding for Unix epoch."""
        epoch_date = np.array(['1970-01-01'], dtype='datetime64[D]')
        encoded = tunit.MONTH.encode(epoch_date)
        assert encoded[0] == 0  # January 1970 should be month 0

    def test_month_encode_sequence(self):
        """Test encoding consecutive months."""
        dates = np.array(['1970-01-15', '1970-02-15', '1970-03-15'],
                         dtype='datetime64[D]')
        encoded = tunit.MONTH.encode(dates)
        expected = np.array([0, 1, 2])
        assert_array_equal(encoded, expected)

    def test_month_decode_to_first_day(self):
        """Test that decoding returns first day of each month."""
        month_nums = np.array([0, 1, 2])
        decoded = tunit.MONTH.decode(month_nums)
        expected = np.array(['1970-01-01', '1970-02-01', '1970-03-01'],
                            dtype='datetime64[D]')
        assert_array_equal(decoded, expected)

    def test_month_same_month_encoding(self):
        """Test that different days in same month encode equally."""
        same_month_dates = np.array(['2023-06-01', '2023-06-15',
                                     '2023-06-30'], dtype='datetime64[D]')
        encoded = tunit.MONTH.encode(same_month_dates)
        # All should have the same month encoding
        assert len(np.unique(encoded)) == 1


class TestQuarterUnit:
    """Tests for QUARTER time unit encoding and decoding."""

    def test_quarter_encode_by_month(self):
        """Test quarter encoding for different months."""
        # Test one date from each quarter of 1970
        dates = np.array(['1970-01-15',  # Q1
                          '1970-04-15',  # Q2
                          '1970-07-15',  # Q3
                          '1970-10-15'],  # Q4
                         dtype='datetime64[D]')
        encoded = tunit.QUARTER.encode(dates)
        expected = np.array([0, 1, 2, 3])
        assert_array_equal(encoded, expected)

    def test_quarter_decode_to_quarter_start(self):
        """Test that decoding returns first month of each quarter."""
        quarter_nums = np.array([0, 1, 2, 3])
        decoded = tunit.QUARTER.decode(quarter_nums)
        expected = np.array(['1970-01-01',  # Q1 starts January
                             '1970-04-01',  # Q2 starts April
                             '1970-07-01',  # Q3 starts July
                             '1970-10-01'],  # Q4 starts October
                            dtype='datetime64[D]')
        assert_array_equal(decoded, expected)

    def test_quarter_same_quarter_encoding(self):
        """Test that dates in same quarter encode equally."""
        q2_dates = np.array(['2023-04-01', '2023-05-15', '2023-06-30'],
                            dtype='datetime64[D]')
        encoded = tunit.QUARTER.encode(q2_dates)
        # All should have the same quarter encoding
        assert len(np.unique(encoded)) == 1


class TestYearUnit:
    """Tests for YEAR time unit encoding and decoding."""

    def test_year_encode_epoch(self):
        """Test year encoding for Unix epoch."""
        epoch_date = np.array(['1970-01-01'], dtype='datetime64[D]')
        encoded = tunit.YEAR.encode(epoch_date)
        assert encoded[0] == 0

    def test_year_encode_sequence(self):
        """Test encoding consecutive years."""
        dates = np.array(['1970-06-15', '1971-06-15', '1972-06-15'],
                         dtype='datetime64[D]')
        encoded = tunit.YEAR.encode(dates)
        expected = np.array([1970, 1971, 1972]) - _EPOCH_YEAR
        assert_array_equal(encoded, expected)

    def test_year_decode_to_january_first(self):
        """Test that decoding returns January 1st of each year."""
        year_nums = np.array([1970, 1971, 1972]) - _EPOCH_YEAR
        decoded = tunit.YEAR.decode(year_nums)
        expected = np.array(['1970-01-01', '1971-01-01', '1972-01-01'],
                            dtype='datetime64[D]')
        assert_array_equal(decoded, expected)

    def test_year_same_year_encoding(self):
        """Test that different dates in same year encode equally."""
        same_year_dates = np.array(['2023-01-01', '2023-06-15',
                                    '2023-12-31'], dtype='datetime64[D]')
        encoded = tunit.YEAR.encode(same_year_dates)
        # All should have the same year encoding
        assert len(np.unique(encoded)) == 1

    def test_year_leap_year_handling(self):
        """Test year encoding handles leap years correctly."""
        leap_year_dates = np.array(['2020-02-29', '2020-12-31'],
                                   dtype='datetime64[D]')
        encoded = tunit.YEAR.encode(leap_year_dates)
        expected = np.array([2020, 2020]) - _EPOCH_YEAR
        assert_array_equal(encoded, expected)


class TestGetBySymbol:
    """Tests for the get_by_symbol function."""

    def test_get_valid_symbols(self):
        """Test retrieving TimeUnits by valid symbols."""
        assert tunit.get_by_symbol('D') is tunit.DAY
        assert tunit.get_by_symbol('W') is tunit.WEEK
        assert tunit.get_by_symbol('Mo') is tunit.MONTH
        assert tunit.get_by_symbol('Q') is tunit.QUARTER
        assert tunit.get_by_symbol('Y') is tunit.YEAR

    def test_get_invalid_symbol(self):
        """Test that invalid symbols raise ValueError."""
        with pytest.raises(ValueError, match="Invalid time unit symbol: 'X'"):
            tunit.get_by_symbol('X')

    def test_get_case_sensitive(self):
        """Test that symbol lookup is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid time unit symbol: 'd'"):
            tunit.get_by_symbol('d')

        with pytest.raises(ValueError, match="Invalid time unit symbol: 'w'"):
            tunit.get_by_symbol('w')

    def test_get_empty_symbol(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Invalid time unit symbol: ''"):
            tunit.get_by_symbol('')

    def test_get_none_symbol(self):
        """Test that None symbol raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            tunit.get_by_symbol(None)

    def test_get_symbol_returns_correct_type(self):
        """Test that get_by_symbol returns TimeUnit instances."""
        for symbol in ['D', 'W', 'Mo', 'Q', 'Y']:
            unit = tunit.get_by_symbol(symbol)
            assert isinstance(unit, tunit.TimeUnit)


class TestSymbolMapping:
    """Tests for the internal symbol mapping."""

    def test_all_units_in_mapping(self):
        """Test that all predefined units are in the symbol mapping."""
        predefined_units = [tunit.DAY, tunit.WEEK, tunit.MONTH,
                            tunit.QUARTER, tunit.YEAR]

        for unit in predefined_units:
            retrieved_unit = tunit.get_by_symbol(unit.symbol)
            assert retrieved_unit is unit

    def test_mapping_completeness(self):
        """Test that mapping contains exactly the expected symbols."""
        expected_symbols = {'D', 'W', 'Mo', 'Q', 'Y'}
        # Access private mapping for testing
        actual_symbols = set(tunit._SYMBOL2UNIT.keys())
        assert actual_symbols == expected_symbols

    def test_no_duplicate_symbols(self):
        """Test that each symbol maps to exactly one unit."""
        symbols = [tunit.DAY.symbol, tunit.WEEK.symbol, tunit.MONTH.symbol,
                   tunit.QUARTER.symbol, tunit.YEAR.symbol]
        assert len(symbols) == len(set(symbols))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_array_encoding(self):
        """Test encoding empty arrays."""
        empty_dates = np.array([], dtype='datetime64[D]')

        for unit in [tunit.DAY, tunit.WEEK, tunit.MONTH,
                     tunit.QUARTER, tunit.YEAR]:
            encoded = unit.encode(empty_dates)
            assert len(encoded) == 0

    def test_empty_array_decoding(self):
        """Test decoding empty arrays."""
        empty_encoded = np.array([])

        for unit in [tunit.DAY, tunit.WEEK, tunit.MONTH,
                     tunit.QUARTER, tunit.YEAR]:
            decoded = unit.decode(empty_encoded)
            assert len(decoded) == 0
            assert decoded.dtype == unit.dtype

    def test_single_element_arrays(self):
        """Test encoding/decoding single-element arrays."""
        single_date = np.array(['2023-01-01'], dtype='datetime64[D]')

        for unit in [tunit.DAY, tunit.WEEK, tunit.MONTH,
                     tunit.QUARTER, tunit.YEAR]:
            encoded = unit.encode(single_date)
            decoded = unit.decode(encoded)
            assert len(encoded) == 1
            assert len(decoded) == 1
            assert decoded.dtype == unit.dtype

    def test_large_date_ranges(self):
        """Test handling of large date ranges."""
        # Test dates spanning multiple decades
        dates = np.array(['1950-01-01', '2000-01-01', '2050-01-01'],
                         dtype='datetime64[D]')

        for unit in [tunit.DAY, tunit.WEEK, tunit.MONTH,
                     tunit.QUARTER, tunit.YEAR]:
            encoded = unit.encode(dates)
            decoded = unit.decode(encoded)
            assert len(encoded) == 3
            assert len(decoded) == 3
            assert decoded.dtype == unit.dtype
