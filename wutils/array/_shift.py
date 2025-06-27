# -*- coding: utf-8 -*-
"""Provides functions for rolling and shifting elements in a NumPy array.

This module contains two primary functions for element manipulation along the
first axis of an array:

- `roll`: Performs a circular shift, where elements that are shifted off one
  end of the array wrap around to the other. This is useful for tasks
  like analyzing periodic data.
- `shift`: Performs a linear shift, where shifted-off elements are discarded
  and the vacated space is filled with a specified value. This is
  commonly used for creating lag/lead features in time-series analysis.

Functions
---------
roll(values, shift)
    Performs a circular shift (roll) on an array.

shift(values, shift, fill_value)
    Performs a linear shift, filling vacated elements.

Created on Fri Jun 27 09:29:53 2025

@author: WaNiNi
"""

from typing import Any

import numpy as np


def roll(values: np.ndarray, shift: int) -> np.ndarray:
    """Roll array elements along the first axis.

    Elements that are shifted off one end are wrapped around to the other end.
    A shift of 0 will return a copy of the input array.

    Parameters
    ----------
    values : np.ndarray
        The input array to be rolled.
    shift : int
        The number of places by which elements are shifted.
        If `shift` is non-zero, a new array is returned.

    Returns
    -------
    np.ndarray
        The rolled array.

    See Also
    --------
    numpy.roll : The underlying NumPy function used for rolling.
    shift : Shift elements and fill vacated spaces with a specific value.

    Examples
    --------
    >>> values = np.arange(1, 11)
    >>> roll(values, 3)
    array([ 8,  9, 10,  1,  2,  3,  4,  5,  6,  7])

    A negative `shift` rolls elements to the left.
    >>> values = np.arange(1, 11)
    >>> roll(values, -3)
    array([ 4,  5,  6,  7,  8,  9, 10,  1,  2,  3])
    """
    # A shift of 0 is a valid operation that returns a new array,
    # consistent with np.roll behavior. No assertion is needed.
    return np.roll(values, shift, axis=0)


def shift(values: np.ndarray, shift: int, fill_value: Any = np.nan
          ) -> np.ndarray:
    """Shift elements of an array along the first axis.

    The vacated spaces at the beginning or end of the array are filled with
    `fill_value`. Elements shifted off the end are discarded.

    Parameters
    ----------
    values : np.ndarray
        The input array to be shifted.
    shift : int
        The number of places by which elements are shifted.
        Positive values shift elements to the right (down), and negative
        values shift elements to the left (up).
    fill_value : Any, optional
        The value used to fill the newly created empty spaces.
        Defaults to `np.nan`.

    Returns
    -------
    np.ndarray
        A new array with the elements shifted. The dtype may be promoted to
        accommodate the `fill_value`.

    See Also
    --------
    pandas.shift : A similar function in the Pandas library.
    roll : Roll elements with wrap-around behavior.

    Examples
    --------
    >>> values = np.arange(1, 11)
    >>> shift(values, 3, fill_value=np.nan)
    array([nan, nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

    A negative `shift` moves elements to the left.
    >>> values = np.arange(1, 11)
    >>> shift(values, -3, fill_value=0)
    array([ 4,  5,  6,  7,  8,  9, 10,  0,  0,  0])

    A shift of 0 returns a copy of the original array.
    >>> values = np.arange(1, 5)
    >>> shifted = shift(values, 0)
    >>> shifted
    array([1, 2, 3, 4])
    """
    if shift == 0:
        return values.copy()

    # Determine the resulting data type to accommodate the fill_value.
    if isinstance(fill_value, str):

        result_dtype = np.result_type(values, np.array(fill_value))
    else:
        result_dtype = np.result_type(values, fill_value)

    # Create the result array, pre-filled with the fill_value.
    # This is more efficient than rolling and then overwriting.
    result_array = np.full_like(values, fill_value, dtype=result_dtype)

    if shift > 0:
        # Positive shift: copy original values to the end of the new array
        result_array[shift:] = values[:-shift]
    else:
        # Negative shift: copy original values to the start of the new array
        result_array[:shift] = values[-shift:]

    return result_array
