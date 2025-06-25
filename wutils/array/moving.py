# -*- coding: utf-8 -*-
"""A collection of utilities for processing moving operations over NumPy arrays.

This module provides a memory-efficient `MovingWindow` class for applying
sliding window calculations (like mean, sum, std) on NumPy arrays without
data duplication. It also includes standalone functions for calculating
lagged changes (`change`, `pct_change`).

Classes
-------
- MovingWindow: A class to create sliding window views and apply operations.

Methods
-------
- change: Calculates the difference between an element and a previous element.
- pct_change: Calculates the percentage change between an element and a
  previous element.

created on Mon Jun 23 16:06:49 2025

@author: WaNiNi
"""

from typing import Any, Callable, Optional

import numpy as np


def _create_window_view(
    values: np.ndarray,
    window_size: int,
    step: int = 1,
    align_right: bool = False
) -> np.ndarray:
    """Creates a memory-efficient, strided, sliding window view of an array.

    This function generates a view of the input `values` array composed of
    sliding windows of a specified size (`window_size`). It avoids copying
    data by manipulating the array's strides, making it highly performant.

    The elements within each window can be contiguous (`step=1`) or separated
    by a fixed `step` to create a dilated (atrous) window.

    Warning: Modifying values in the output array can lead to unexpected
    results in the original array, as the windows share memory.

    Parameters
    ----------
    values : np.ndarray
        The input array. The sliding window is applied along the first axis.
    window_size : int
        The number of elements in each window. Must be a positive integer.
    step : int, optional
        The dilation step between elements within a single window, by default 1.
        A `step` > 1 creates a dilated (atrous) window.
    align_right : bool, optional
        If True and `step` > 1, the input array is sliced to align the output
        with the end of the window intervals. This is useful for aligning
        feature and label arrays. By default False.

    Returns
    -------
    np.ndarray
        A view of the original array with a new leading dimension representing
        the windows. The shape will be `(n_windows, window_size, ...)`.
        Returns an empty array with the correct shape if not enough elements
        exist to form a single window.
    """
    if align_right and step > 1:
        # Slice the array to align the window's end point.
        values = values[step - 1:]

    # Calculate the total span covered by one window.
    window_span = step * (window_size - 1) + 1
    n_windows = len(values) - window_span + 1

    # Define the shape of the resulting windowed array.
    shape = (n_windows, window_size) + values.shape[1:]

    if n_windows <= 0:
        # Not enough data to form one window, return a correctly shaped
        # empty array.
        return np.array([], dtype=values.dtype).reshape((0,) + shape[1:])

    # Define the strides for the new view.
    # Stride 1: Move to the next window (same as moving one element in `values`).
    # Stride 2: Move to the next sample in a window (move `step` elements).
    # Stride 3+: Strides for the remaining dimensions (e.g., columns).
    strides = (values.strides[0], step * values.strides[0]) + values.strides[1:]

    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)


class MovingWindow:
    """Encapsulates a sliding window view of an array for reduction operations.

    This class creates a memory-efficient, strided, sliding window view of an
    array upon initialization. It offers various methods to compute statistics
    (like mean, max, std) or apply custom functions over these windows. This
    approach is efficient as the window view is created only once, lazily.

    Parameters
    ----------
    values : np.ndarray
        The input array. The sliding window is applied along the first axis.
    window_size : int
        The number of elements in each window. Must be a positive integer.
    step : int, optional
        The dilation step between elements within a single window, by default 1.
        A `step` > 1 creates a dilated window. Must be a positive integer.
    align_right : bool, optional
        If True and `step` > 1, aligns windows to the right. See the
        `_create_window_view` function for details. By default False.

    Attributes
    ----------
    windows : np.ndarray
        The sliding window view. Created lazily when first accessed. The shape
        is `(n_windows, window_size, ...)`.
    shape : tuple
        The shape of the sliding window view.
    n_windows : int
        The total number of valid sliding windows.
    window_size : int
        The size (number of elements) of each sliding window.
    step : int
        The step or dilation factor between elements in a window.
    align_right : bool
        A boolean indicating if the window alignment is shifted.

    Properties
    ----------
    windows : np.ndarray
        The sliding window view. Created lazily when first accessed.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(1, 11, dtype=float)

    >>> # Create a moving window object
    >>> mv = MovingWindow(data, window_size=3)

    >>> # Get the raw windowed array
    >>> mv.windows
    array([[ 1.,  2.,  3.],
           [ 2.,  3.,  4.],
           [ 3.,  4.,  5.],
           [ 4.,  5.,  6.],
           [ 5.,  6.,  7.],
           [ 6.,  7.,  8.],
           [ 7.,  8.,  9.],
           [ 8.,  9., 10.]])

    >>> # Chained operations
    >>> mv.mean()
    array([2., 3., 4., 5., 6., 7., 8., 9.])

    >>> # Keep original size by padding
    >>> mv.std(keep_length=True)
    array([       nan,        nan, 0.81649658, 0.81649658, 0.81649658,
           0.81649658, 0.81649658, 0.81649658, 0.81649658, 0.81649658])

    >>> # Apply a custom function
    >>> mv.reduce(np.prod)
    array([  6.,  24.,  60., 120., 210., 336., 504., 720.])
    """
    def __init__(self,
                 values: np.ndarray,
                 window_size: int,
                 step: int = 1,
                 align_right: bool = False):
        # --- Parameter validation ---
        if not isinstance(values, np.ndarray):
            raise TypeError("Parameter 'values' must be a NumPy array.")
        if window_size <= 0:
            raise ValueError(
                "Parameter 'window_size' must be a positive integer."
            )
        if step <= 0:
            raise ValueError("Parameter 'step' must be a positive integer.")

        self._original_values = values
        self._window_size = window_size
        self._step = step
        self._align_right = align_right

        # --- Internal state for caching ---
        self._windows: Optional[np.ndarray] = None

    def _create_windows(self) -> np.ndarray:
        """The core logic to create a strided, sliding window view."""
        return _create_window_view(
            values=self._original_values,
            window_size=self._window_size,
            step=self._step,
            align_right=self._align_right
        )

    @property
    def windows(self) -> np.ndarray:
        """The sliding window view. Created lazily when first accessed.

        Returns
        -------
        np.ndarray
            A view of the original array with shape
            (n_windows, window_size, ...).
        """
        if self._windows is None:
            self._windows = self._create_windows()
        return self._windows

    @property
    def window_size(self) -> int:
        """The size (number of elements) of each sliding window."""
        return self._window_size

    @property
    def step(self) -> int:
        """The step or dilation factor between elements in a window."""
        return self._step

    @property
    def align_right(self) -> bool:
        """A boolean indicating if the window alignment is shifted.

        If True, indicates that the input array was sliced to align
        the output with the end of the window intervals.
        """
        return self._align_right

    @property
    def shape(self) -> tuple:
        """A tuple representing the dimensions of the sliding window view.

        The shape is in the format `(n_windows, window_size, ...)`,
        where `...` represents the dimensions of the original array's
        elements.
        """
        return self.windows.shape

    @property
    def n_windows(self) -> int:
        """The total number of valid sliding windows, an integer.

        This value corresponds to the first dimension of the `.shape`
        property.
        """
        return self.windows.shape[0]

    def reduce(self,
               func: Callable[..., np.ndarray],
               keep_length: bool = False,
               fill_value: Any = np.nan,
               **kwargs: Any) -> np.ndarray:
        """Apply a reduction function to each sliding window.

        Parameters
        ----------
        func : Callable
            The reduction function to apply (e.g., `np.mean`, `np.sum`).
            It will be called as `func(window, axis=1, **kwargs)`.
        keep_length : bool, optional
            If True, the output will have the same length as the input array,
            padded at the beginning with `fill_value`. By default False.
        fill_value : Any, optional
            The value to use for padding if `keep_length` is True.
            Defaults to `np.nan`.
        **kwargs
            Additional keyword arguments to pass to the reduction function.

        Returns
        -------
        np.ndarray
            The array of reduced values.
        """
        windows = self.windows

        # If there are no windows, handle return cases.
        if windows.shape[0] == 0:
            # Determine the safest and most precise output dtype by checking
            # the original array's dtype and the fill_value's type.
            output_dtype = np.result_type(self._original_values, fill_value)

            if keep_length:
                return np.full(
                    self._original_values.shape,
                    fill_value,
                    dtype=output_dtype
                )
            else:
                # Correctly determine the empty output shape
                output_shape = (0,) + self._original_values.shape[1:]
                return np.array([], dtype=output_dtype).reshape(output_shape)

        # Apply the reduction function along the window axis (axis=1).
        result = func(windows, axis=1, **kwargs)

        if keep_length:
            # Determine the safest and most precise output dtype by checking
            # the result array's dtype and the fill_value's type.
            output_dtype = np.result_type(result, fill_value)

            # Create a padded array and copy the results into it.
            pad_width = len(self._original_values) - len(result)

            padded_result = np.full(
                (len(self._original_values),) + result.shape[1:],
                fill_value,
                dtype=output_dtype
            )
            padded_result[pad_width:] = result
            return padded_result

        return result

    def mean(self, keep_length: bool = False) -> np.ndarray:
        """Computes the moving mean.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving mean.

        See Also
        --------
        numpy.mean : The underlying function used for calculation.
        MovingWindow.average : For computing a weighted average.
        """
        return self.reduce(np.mean, keep_length=keep_length)

    def average(self,
                weights: Optional[np.ndarray] = None,
                keep_length: bool = False) -> np.ndarray:
        """Computes the moving weighted average.

        Parameters
        ----------
        weights : np.ndarray, optional
            An array of weights associated with the values in each window.
            Each value in a window contributes to the average according to
            its associated weight.
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving weighted average.

        See Also
        --------
        numpy.average : The underlying function used for calculation.
        MovingWindow.mean : For unweighted average.
        """
        return self.reduce(np.average, keep_length=keep_length, weights=weights)

    def max(self, keep_length: bool = False) -> np.ndarray:
        """Computes the moving maximum.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving maximum.

        See Also
        --------
        numpy.max : The underlying function used for calculation.
        """
        return self.reduce(np.max, keep_length=keep_length)

    def min(self, keep_length: bool = False) -> np.ndarray:
        """Computes the moving minimum.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving minimum.

        See Also
        --------
        numpy.min : The underlying function used for calculation.
        """
        return self.reduce(np.min, keep_length=keep_length)

    def std(self, ddof: int = 0, keep_length: bool = False) -> np.ndarray:
        """Computes the moving standard deviation.

        Parameters
        ----------
        ddof : int, optional
            Delta Degrees of Freedom. The divisor used in calculations is
            `N - ddof`, where `N` represents the number of elements.
            By default 0.
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving standard deviation.

        See Also
        --------
        numpy.std : The underlying function used for calculation.
        """
        return self.reduce(np.std, keep_length=keep_length, ddof=ddof)

    def sum(self, keep_length: bool = False) -> np.ndarray:
        """Computes the moving sum.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.

        Returns
        -------
        np.ndarray
            The array of the moving sum.

        See Also
        --------
        numpy.sum : The underlying function used for calculation.
        """
        return self.reduce(np.sum, keep_length=keep_length)

    def all(
        self,
        keep_length: bool = False,
        allow_nan: bool = False
    ) -> np.ndarray:
        """Computes the moving logical AND.

        Tests whether all elements along the moving window evaluate to True.
        By default, returns a boolean array. If `allow_nan` is True,
        returns a float array where True is 1.0, False is 0.0, and
        padded values are `np.nan`.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.
        allow_nan : bool, optional
            If True, the output array will have a float dtype, allowing
            `np.nan` to be used for padding. If False (default), the output
            is a boolean array, and `False` is used for padding.

        Returns
        -------
        np.ndarray
            The array of the moving `all` operation. The dtype will be
            `bool` or `float` depending on the `allow_nan` parameter.

        See Also
        --------
        numpy.all : The underlying function used for calculation.
        """
        fill_value = np.nan if allow_nan else False
        return self.reduce(
            np.all,
            keep_length=keep_length,
            fill_value=fill_value
        )

    def any(
        self,
        keep_length: bool = False,
        allow_nan: bool = False
    ) -> np.ndarray:
        """Computes the moving logical OR.

        Tests whether any element along the moving window evaluates to True.
        By default, returns a boolean array. If `allow_nan` is True,
        returns a float array where True is 1.0, False is 0.0, and
        padded values are `np.nan`.

        Parameters
        ----------
        keep_length : bool, optional
            If True, pads the output to match the original input length.
            By default False.
        allow_nan : bool, optional
            If True, the output array will have a float dtype, allowing
            `np.nan` to be used for padding. If False (default), the output
            is a boolean array, and `False` is used for padding.

        Returns
        -------
        np.ndarray
            The array of the moving `any` operation. The dtype will be
            `bool` or `float` depending on the `allow_nan` parameter.

        See Also
        --------
        numpy.any : The underlying function used for calculation.
        """
        fill_value = np.nan if allow_nan else False
        return self.reduce(
            np.any,
            keep_length=keep_length,
            fill_value=fill_value
        )


def change(
    values: np.ndarray, offset: int, keep_length: bool = False
) -> np.ndarray:
    """Calculates the moving change between elements separated by an offset.

    This function computes the difference between an element and a previous
    element separated by `offset`. The effective formula is:
    `output[i] = values[i] - values[i - offset]`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    offset : int
        The period or lag to calculate the change over. Must be a
        positive integer. An `offset` of 1 is equivalent to `np.diff`.
    keep_length : bool, optional
        If True, pads the output to match the original input length.
        By default False.

    Returns
    -------
    np.ndarray
        An array containing the moving changes.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
    >>> change(data, 2)
    array([ 1.,  2.,  3.,  5.,  8., 13., 21., 34.])

    >>> # Keep original length by padding
    >>> change(data, 2, keep_length=True)
    array([nan, nan,  1.,  2.,  3.,  5.,  8., 13., 21., 34.])
    """
    if not isinstance(offset, int) or offset <= 0:
        raise ValueError("Parameter 'offset' must be a positive integer.")
    if len(values) < offset:
        if keep_length:
            return np.full(values.shape, np.nan)
        else:
            return np.array([], dtype=values.dtype)

    result = values[offset:] - values[:-offset]

    if keep_length:
        output_dtype = np.result_type(result, np.nan)
        padded_result = np.full(values.shape, np.nan, dtype=output_dtype)
        padded_result[offset:] = result
        return padded_result

    return result


def pct_change(
    values: np.ndarray, offset: int, keep_length: bool = False
) -> np.ndarray:
    """Calculates the moving rate of change between elements by an offset.

    This function computes the rate of change based on the formula:
    `output[i] = (values[i] / values[i - offset]) - 1`.

    It handles division by zero by producing `np.nan` for those cases.

    Parameters
    ----------
    values : np.ndarray
        Input array. Must be of a float dtype to handle `np.nan`.
    offset : int
        The period or lag to calculate the rate of change over. Must be a
        positive integer.
    keep_length : bool, optional
        If True, pads the output with `np.nan` at the beginning to match
        the input array's size, by default False.

    Returns
    -------
    np.ndarray
        An array containing the moving rates of change.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
    >>> pct_change(data, 2)
    array([1.        , 2.        , 1.5       , 1.66666667, 1.6       ,
           1.625     , 1.61538462, 1.61904762])

    >>> # Keep original length by padding
    >>> pct_change(data, 2, keep_length=True)
    array([       nan,        nan, 1.        , 2.        , 1.5       ,
           1.66666667, 1.6       , 1.625     , 1.61538462, 1.61904762])
    """
    if not isinstance(offset, int) or offset <= 0:
        raise ValueError("Parameter 'offset' must be a positive integer.")
    if len(values) < offset:
        if keep_length:
            return np.full(values.shape, np.nan)
        else:
            return np.array([], dtype=float)  # Returns float for consistency

    previous_values = values[:-offset]
    current_values = values[offset:]

    # Use `np.divide` to safely calculate the percentage change.
    # The `where` clause ensures the calculation only runs when the previous
    # value is positive, as percentage change is ambiguous or undefined for
    # non-positive base values.
    # For all other cases, the `out` array provides a default of `np.nan`.
    result = np.divide(
        current_values,
        previous_values,
        out=np.full_like(current_values, np.nan, dtype=float),
        where=(previous_values > 0)
    ) - 1

    if keep_length:
        padded_result = np.full(values.shape, np.nan)
        padded_result[offset:] = result
        return padded_result

    return result
