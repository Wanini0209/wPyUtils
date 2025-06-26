# -*- coding: utf-8 -*-
"""A collection of utilities for moving/rolling operations on NumPy arrays.

This module provides a suite of tools for applying calculations over a
moving window of an array. It is designed for time-series analysis and
signal processing tasks.

The module includes a memory-efficient `MovingWindow` class for generic
rolling calculations, alongside specialized functions for common moving
operations like lagged changes (`change`, `pct_change`) and exponential
moving averages (`exponential_smoothing`, `ema`).

Classes
-------
- MovingWindow: A class to create sliding window views for efficient
  rolling calculations without data duplication.

Functions
---------
- change: Calculates the moving difference between an element and a
  previous one.
- pct_change: Calculates the moving percentage change between an element
  and a previous one.
- exponential_smoothing: A foundational function for moving averages that
  applies exponential smoothing with a given factor (alpha).
- ema: Calculates the Exponential Moving Average (EMA), a specific type
  of exponential smoothing commonly used in financial technical indicators.

created on Mon Jun 23 16:06:49 2025

@author: WaNiNi
"""

from typing import Any, Callable, Optional, Union

import numba
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


@numba.njit(cache=True)
def _ema_numba_1d(values: np.ndarray, alpha: float, iv: float) -> np.ndarray:
    """
    Compute exponential moving average for 1D array using Numba JIT compilation.

    Parameters
    ----------
    values : np.ndarray
        Input 1D array.
    alpha : float
        Smoothing parameter (0 < alpha <= 1).
    iv : float
        Initial value.

    Returns
    -------
    np.ndarray
        Exponential moving average result.
    """
    ret = np.empty(values.shape, dtype=np.float64)

    # Initialize first value
    ret[0] = alpha * values[0] + (1 - alpha) * iv

    # Compute EMA for remaining values
    for idx in range(1, len(values)):
        ret[idx] = alpha * values[idx] + (1 - alpha) * ret[idx - 1]

    return ret


@numba.njit(cache=True)
def _ema_numba_2d(values: np.ndarray, alpha: float, iv: np.ndarray) -> np.ndarray:
    """
    Compute exponential moving average for 2D array using Numba JIT compilation.

    Parameters
    ----------
    values : np.ndarray
        Input 2D array with shape (n_timesteps, n_features).
    alpha : float
        Smoothing parameter (0 < alpha <= 1).
    iv : np.ndarray
        Initial values for each feature.

    Returns
    -------
    np.ndarray
        Exponential moving average result with same shape as input.
    """
    ret = np.empty(values.shape, dtype=np.float64)

    # Initialize first row
    ret[0] = alpha * values[0] + (1 - alpha) * iv

    # Compute EMA for remaining rows
    for i in range(1, len(values)):
        for j in range(len(iv)):
            ret[i, j] = alpha * values[i, j] + (1 - alpha) * ret[i - 1, j]

    return ret


def exponential_smoothing(
    values: np.ndarray,
    alpha: float,
    iv: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """
    Perform exponential smoothing on the first dimension of a given array.

    The exponential moving average (EMA) is computed recursively as:
        ema[t] = α * d[t] + (1-α) * ema[t-1]

    Which can be expanded to:
        ema[0] = (1-α) * iv + α * d[0]
        ema[1] = (1-α)^2 * iv + α(1-α) * d[0] + α * d[1]
        ...

    Parameters
    ----------
    values : np.ndarray
        Input array. For 1D arrays, computes EMA directly.
        For multi-dimensional arrays, computes EMA along the first axis.
    alpha : float
        Smoothing parameter, must satisfy 0 < alpha <= 1.
        Higher values give more weight to recent observations.
    iv : float or np.ndarray, optional
        Initial value(s). If None, uses the first observation(s) as initial
        value. For multi-dimensional arrays, can be a scalar (broadcast to all
        features) or an array matching the shape of `values[0]`.

    Returns
    -------
    np.ndarray
        Exponentially smoothed array with the same shape as input.

    Raises
    ------
    ValueError
        If alpha is not in the range (0, 1].

    See Also
    --------
    https://en.wikipedia.org/wiki/Exponential_smoothing

    Examples
    --------
    Basic 1D example:

    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> exponential_smoothing(values, 1/3)
    array([1.        , 1.33333333, 1.88888889, 2.59259259, 3.39506173,
           4.26337449, 5.17558299, 6.11705533, 7.07803688, 8.05202459])

    Multi-dimensional example:

    >>> values = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
    >>> exponential_smoothing(values, 1/3)
    array([[1.        , 6.        ],
           [1.33333333, 6.33333333],
           [1.88888889, 6.88888889],
           [2.59259259, 7.59259259],
           [3.39506173, 8.39506173]])

    With specified initial value:

    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> exponential_smoothing(values, 1/3, iv=0)
    array([0.33333333, 0.88888889, 1.59259259, 2.39506173, 3.26337449,
           4.17558299, 5.11705533, 6.07803688, 7.05202459, 8.03468306])
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"Parameter `alpha` must be in range (0, 1], got {alpha}")

    if values.size == 0:
        return values.copy()

    # Set initial value if not provided
    if iv is None:
        iv = values[0]

    # Handle multi-dimensional arrays
    if values.ndim > 1:
        # Broadcast initial value to match the shape of remaining dimensions
        iv = np.broadcast_to(iv, values.shape[1:])

        # Reshape for 2D processing and then reshape back
        return _ema_numba_2d(
            values.reshape(len(values), -1),
            alpha,
            iv.flatten()
        ).reshape(values.shape)

    # Handle 1D arrays
    return _ema_numba_1d(values, alpha, float(iv))


def ema(
    values: np.ndarray,
    period: int,
    factor: int = 2,
    iv: Optional[Union[float, np.ndarray]] = None,
    keep_length: bool = False,
) -> np.ndarray:
    """
    Perform EMA on the first dimension of a given array using period.

    EMA is a specific type of exponential smoothing commonly used in financial
    technical indicators. The smoothing factor alpha is calculated from the
    period and an adjustment factor.
        alpha = factor / (period + factor - 1)
    A common setting in finance is `factor=2`.

    Parameters
    ----------
    values : np.ndarray
        Input array. Calculation is performed along the first axis.
    period : int
        The period for the EMA. Must be an integer greater than 1.
    factor : int, default 2
        The adjustment factor for calculating alpha. Must be >= 1.
    iv : float or np.ndarray, optional
        Initial value(s). If provided, the calculation starts from the first
        element. If None (default), the simple moving average of the first
        `period` elements is used as the initial value.
    keep_length : bool, default False
        If True, the output array will have the same length as the input
        array, with `NaN` padding for the initial `period - 1` elements.
        If False, the output will only contain valid EMA values, starting from
        the `period-1`-th element.

    Returns
    -------
    np.ndarray
        The calculated EMA array. Its length depends on `keep_length`.

    Raises
    ------
    ValueError
        If `period` is not > 1 or `factor` is not >= 1.

    Examples
    --------
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> # Using period=3, factor=2 (default) -> alpha = 2 / (3+2-1) = 0.5
    >>> ema(values, 3)
    array([2. ,3. ,4. ,5. ,6. , 7., 8., 9.])

    >>> # With keep_length=True, the first 2 (period-1) values are NaN.
    >>> ema(values, 3, keep_length=True)
    array([nan, nan,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

    >>> # With a specified initial value, calculation starts from index 0.
    >>> # Using period=5, factor=1 -> alpha = 1/5 = 0.2
    >>> ema(values, 5, factor=1, iv=0)
    array([0.2       , 0.56      , 1.048     , 1.6384    , 2.31072   ,
           3.048576  , 3.8388608 , 4.67108864, 5.53687091, 6.42949673])
    """
    if not isinstance(period, int) or period <= 1:
        raise ValueError(f"Parameter `period` must be an integer > 1, got {period}")
    if not isinstance(factor, int) or factor < 1:
        raise ValueError(f"Parameter `factor` must be an integer >= 1, got {factor}")

    alpha = factor / (period + factor - 1)

    if iv is not None:
        return exponential_smoothing(values, alpha, iv=iv)

    if len(values) < period:
        if keep_length:
            return np.full(values.shape, np.nan)
        else:
            return np.array([], dtype=float)

    # Use the mean of the first `period` elements as the initial value.
    iv = values[:period].mean(axis=0)
    ret = np.empty(values.shape, dtype=float)

    # The first EMA value is at index `period-1` and is the initial value itself.
    ret[period - 1] = iv

    # Calculate the rest of the EMA.
    if len(values) > period:
        ret[period:] = exponential_smoothing(values[period:], alpha, iv=iv)

    if keep_length:
        ret[:period - 1] = np.nan
        return ret

    return ret[period - 1:]
