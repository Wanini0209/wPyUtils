# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Defines time unit constants and functions for time series manipulation.

This module provides a structured way to define different time units (including
Day, Week, Month, Quarter, Year) and the logic to encode/decode them from a
base `datetime64` representation. It serves as the foundation for time-based
aggregation and indexing operations in time series data.

The module implements a flexible time unit system where each unit can define
custom encoding and decoding logic to convert between datetime representations
and their corresponding numerical indices. This allows for efficient storage
and manipulation of time-based data while maintaining semantic meaning.

Key Components
--------------
- TimeUnit: A class representing the properties and conversion logic for a
  specific time unit.
- Module-level constants: DAY, WEEK, MONTH, QUARTER, YEAR provide direct
  access to predefined TimeUnit instances.
- get_by_symbol(): A function to retrieve a TimeUnit instance by its
  symbol (e.g., 'D', 'W').

Examples
--------
Basic usage of time units for encoding and decoding dates:

>>> import numpy as np
>>> from wutils.tseries import tunit
>>>
>>> # Create some sample dates
>>> dates = np.array(['2023-01-01', '2023-01-08'], dtype='datetime64[D]')
>>>
>>> # Access TimeUnit constants directly from the module
>>> day_encoded = tunit.DAY.encode(dates)
>>> week_encoded = tunit.WEEK.encode(dates)
>>>
>>> # Use the module-level function to get a unit by symbol
>>> month_unit = tunit.get_by_symbol('Mo')
>>>
>>> # Decode back to datetime representation
>>> decoded_days = tunit.DAY.decode(day_encoded)
>>> decoded_weeks = tunit.WEEK.decode(week_encoded)

Created on Fri Jun 27 13:37:01 2025

@author: WaNiNi
"""

from typing import Callable, Dict, NamedTuple, Optional

import numpy as np


class TimeUnit(NamedTuple):
    """Represents a unit of time and its conversion logic.

    This class defines a time unit's properties and provides methods to
    convert between datetime objects and their numerical representations.
    The conversion is typically bidirectional, allowing for efficient
    storage and retrieval of time-based data.

    The encoding process converts datetime values to numerical indices,
    while decoding converts numerical indices back to datetime values.
    This allows for compact representation and fast operations on time
    series data.

    Attributes
    ----------
    noun : str
        The singular noun for the time unit (e.g., 'Day', 'Week').
        Used for display and documentation purposes.
    adj : str
        The adjective form for the time unit (e.g., 'Daily', 'Weekly').
        Used when describing frequency or intervals.
    symbol : str
        A short symbol for the unit (e.g., 'D', 'W', 'Mo').
        Used as a compact identifier and for parsing time unit specifications.
    dtype : np.dtype
        The base numpy data type, typically datetime64[D]. This defines the
        fundamental time resolution and storage format.
    encoder : Optional[Callable[[np.ndarray], np.ndarray]]
        A function that takes a datetime array and returns its numerical
        representation. If None, no encoding transformation is applied.
    decoder : Optional[Callable[[np.ndarray], np.ndarray]]
        A function that takes a numerical representation and converts it back
        to an intermediate datetime type. The final conversion to the base
        dtype is handled automatically. If None, values are cast directly
        to the base dtype.

    Notes
    -----
    The encoder and decoder functions should be inverse operations to ensure
    data integrity. The decode method always performs a final cast to the
    base dtype to ensure consistent output format.

    Examples
    --------
    Creating a custom time unit with encoding logic:

    >>> custom_unit = TimeUnit(
    ...     noun='BiWeek',
    ...     adj='BiWeekly',
    ...     symbol='2W',
    ...     dtype=np.dtype('datetime64[D]'),
    ...     encoder=lambda x: x.astype(np.int64) // 14,  # 14-day periods
    ...     decoder=lambda x: x * 14  # Convert back to days
    ... )
    """
    noun: str
    adj: str
    symbol: str
    dtype: np.dtype
    encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None
    decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encodes datetime values into a numerical representation.

        This method converts datetime values to their corresponding numerical
        indices according to the time unit's encoding logic. The process
        first ensures the input is in the expected base format, then applies
        the encoder function if one is defined.

        Parameters
        ----------
        values : np.ndarray
            Input datetime array to be encoded. The array will be cast to the
            time unit's base dtype before encoding.

        Returns
        -------
        np.ndarray
            Numerical representation of the input dates. The specific format
            depends on the encoder function. Common formats include integer
            days/weeks/months since epoch, ordinal numbers for periodic time
            units, or custom numerical representations.

        Notes
        -----
        The input array is automatically cast to the time unit's base
        dtype before encoding, ensuring consistent input format regardless
        of the original datetime resolution.

        Examples
        --------
        >>> dates = np.array(['2023-01-01', '2023-01-02'],
        ...                  dtype='datetime64[D]')
        >>> # Assuming DAY is a predefined module-level constant
        >>> # from wutils.tseries import tunit
        >>> # encoded = tunit.DAY.encode(dates)
        >>> # print(encoded)
        # [19358 19359]
        """
        # The first astype ensures the input is in the expected base format.
        values = values.astype(self.dtype, copy=False)
        return self.encoder(values) if self.encoder else values

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Decodes numerical values back to the base datetime format.

        This method converts numerical indices back to datetime values
        according to the time unit's decoding logic. If a decoder function
        is defined, it is applied first, followed by a final cast to the
        base datetime format.

        Parameters
        ----------
        values : np.ndarray
            Input numerical array to be decoded. The format should match what
            the corresponding encode method produces.

        Returns
        -------
        np.ndarray
            Datetime array in the time unit's base dtype format. The output
            is always cast to the base dtype to ensure consistent format.

        Notes
        -----
        The final cast to the base dtype ensures that regardless of the
        decoder's intermediate output format, the final result always
        matches the time unit's specified dtype.

        Examples
        --------
        >>> encoded_days = np.array([19358, 19359])  # Days since epoch
        >>> # Assuming DAY is a predefined module-level constant
        >>> # from wutils.tseries import tunit
        >>> # decoded = tunit.DAY.decode(encoded_days)
        >>> # print(decoded)
        # ['2023-01-01' '2023-01-02']
        """
        if self.decoder:
            values = self.decoder(values)
        # Final cast to the base dtype.
        return values.astype(self.dtype, copy=False)


# The base unit, Day
# The epoch is 1970-01-01 (Unix epoch).
# This is the fundamental time unit with no aggregation - each value
# represents a single calendar day.
DAY = TimeUnit(
    noun='Day',
    adj='Daily',
    symbol='D',
    dtype=np.dtype('datetime64[D]'),
    encoder=lambda x: x.astype(np.int64),
    # Decoder is None; the decode method handles the final cast to datetime64[D].
)

# Weekly time unit.
# The encoding formula `(days_since_epoch + 3) // 7` aligns weeks to start
# on Monday. The numpy epoch (1970-01-01) was a Thursday, so we add 3 days
# to align the week boundary. This ensures that:
# - Week 0 starts on Monday, 1969-12-29
# - Week 1 starts on Monday, 1970-01-05
# - And so on...
#
# The decoding formula `week_num * 7 - 3` converts a week number back to the
# day number of the Monday of that week. This maintains ISO 8601 week
# convention where weeks start on Monday.
WEEK = TimeUnit(
    noun='Week',
    adj='Weekly',
    symbol='W',
    dtype=np.dtype('datetime64[D]'),
    encoder=lambda x: (x.astype(np.int64) + 3) // 7,
    decoder=lambda x: (x * 7 - 3),  # Decodes to integer days
)

# Monthly time unit.
# Months are variable-length periods, so we use numpy's built-in datetime64[M]
# type for accurate month arithmetic. The encoder converts daily datetime
# values to monthly datetime values, then to integer months since epoch.
# The decoder reverses this process, converting month numbers back to the
# first day of each month.
MONTH = TimeUnit(
    noun='Month',
    adj='Monthly',
    symbol='Mo',
    dtype=np.dtype('datetime64[D]'),
    encoder=lambda x: x.astype('datetime64[M]').astype(np.int64),
    decoder=lambda x: x.astype('datetime64[M]'),
)

# Quarterly time unit.
# Quarters are 3-month periods starting from January. The encoding divides
# the month number by 3 (integer division) to get quarter numbers:
# - Q1 (Jan-Mar): months 0-2 -> quarter 0
# - Q2 (Apr-Jun): months 3-5 -> quarter 1
# - Q3 (Jul-Sep): months 6-8 -> quarter 2
# - Q4 (Oct-Dec): months 9-11 -> quarter 3
#
# The decoder multiplies quarter numbers by 3 to get the starting month
# of each quarter, then converts to datetime format.
QUARTER = TimeUnit(
    noun='Quarter',
    adj='Quarterly',
    symbol='Q',
    dtype=np.dtype('datetime64[D]'),
    encoder=lambda x: x.astype('datetime64[M]').astype(np.int64) // 3,
    decoder=lambda x: (x * 3).astype('datetime64[M]'),
)

# Yearly time unit.
# Years are handled using numpy's datetime64[Y] type for accurate year
# arithmetic, accounting for leap years and calendar variations. The
# encoder converts daily dates to yearly dates, then to integer years.
# The decoder converts year numbers back to January 1st of each year.
YEAR = TimeUnit(
    noun='Year',
    adj='Annual',
    symbol='Y',
    dtype=np.dtype('datetime64[D]'),
    encoder=lambda x: x.astype('datetime64[Y]').astype(np.int64),
    decoder=lambda x: x.astype('datetime64[Y]'),
)


# A mapping from symbols to TimeUnit instances for quick lookups.
# This dictionary is used internally by the get_by_symbol function.
# It provides O(1) lookup performance for symbol-based time unit retrieval.
#
# The mapping is created at module import time and remains constant throughout
# the program execution, ensuring consistent behavior and performance.
_SYMBOL2UNIT: Dict[str, TimeUnit] = {
    unit.symbol: unit for unit in [DAY, WEEK, MONTH, QUARTER, YEAR]
}


def get_by_symbol(symbol: str) -> TimeUnit:
    """Retrieves a TimeUnit instance by its symbol.

    This function provides a convenient way to lookup time units by their
    symbolic representation. It uses an internal mapping for O(1) lookup
    performance and provides clear error messages for invalid symbols.

    Parameters
    ----------
    symbol : str
        The symbol of the time unit to retrieve. Must match exactly
        (case-sensitive). Valid symbols are: 'D', 'W', 'Mo', 'Q', 'Y'.

    Returns
    -------
    TimeUnit
        The corresponding `TimeUnit` instance for the given symbol.

    Raises
    ------
    ValueError
        If no `TimeUnit` with the given symbol is found. The error
        message includes the invalid symbol for debugging.

    Notes
    -----
    Symbol matching is case-sensitive. For example, 'd' is not
    equivalent to 'D' and will raise a ValueError.

    Examples
    --------
    >>> # Assuming this function is in the tunit module
    >>> # from wutils.tseries import tunit
    >>> # day_unit = tunit.get_by_symbol('D')
    >>> # month_unit = tunit.get_by_symbol('Mo')
    >>>
    >>> # Invalid lookup (raises ValueError)
    >>> try:
    ...     # invalid_unit = tunit.get_by_symbol('X')
    ...     pass # Placeholder for doctest
    ... except ValueError as e:
    ...     print(f"Error: {e}")
    # Error: Invalid time unit symbol: 'X'
    """
    try:
        return _SYMBOL2UNIT[symbol]
    except KeyError:
        raise ValueError(f"Invalid time unit symbol: '{symbol}'")
