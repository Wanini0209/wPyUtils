# -*- coding: utf-8 -*-
"""A collection of utilities for processing moving windows over NumPy arrays.

This module offers efficient, view-based methods for creating sliding
windows (or samples) from arrays and applying various numerical and logical
operations on them. The use of NumPy's stride tricks ensures high performance
and memory efficiency by avoiding data duplication.

Available Functions
-------------------
Window Extraction:
    - moving_sampling: Creates a sliding window view of an array.

Numerical Window Operations:
    - moving_average: Computes the moving (weighted) average.
    - moving_max: Computes the moving maximum.
    - moving_min: Computes the moving minimum.
    - moving_sum: Computes the moving sum.
    - moving_std: Computes the moving standard deviation.
    - moving_change: Computes the moving difference over a fixed offset.
    - moving_change_rate: Computes the moving rate of change over a fixed offset.

Logical Window Operations:
    - moving_all: Computes the moving logical AND.
    - moving_any: Computes the moving logical OR.

Generic Window Operations:
    - moving_reduction: A generic function to apply any reduction operation.

Created on Mon Jun 23 16:06:49 2025

@author: WaNiNi
"""

from typing import Callable, Optional

import numpy as np


def moving_sampling(values: np.ndarray, samples: int, step: int = 1,
                    left_open_sampling: bool = False
                    ) -> np.ndarray:
    """Creates a memory-efficient, strided, sliding window view of an array.

    This function generates a view of the input `values` array composed of
    sliding windows of a specified size (`samples`). It avoids copying data
    by manipulating the array's strides, making it highly performant.

    The elements within each window can be contiguous (`step=1`) or separated
    by a fixed `step` to create a dilated (atrous) window. The window slides
    one element at a time along the first axis.

    Parameters
    ----------
    values : np.ndarray
        The input array to be sampled. The sliding window is applied along
        the first axis.
    samples : int
        The number of elements in each window (i.e., the window size).
        Must be a positive integer.
    step : int, optional
        The dilation step between elements within a single window, by default 1.
        A `step` of 1 means the elements are contiguous. A `step` > 1
        creates a dilated (or atrous) window. Must be a positive integer.
    left_open_sampling : bool, optional
        If True, the first `step - 1` elements of the input `values` array
        are skipped before creating windows. This can be useful for aligning
        feature and label arrays where the label corresponds to the end of
        an interval. By default False.

    Returns
    -------
    np.ndarray
        A view of the original array with a new leading dimension representing
        the windows. The shape will be `(n_windows, samples, ...)`, where
        `...` is the shape of the original array's elements. Returns an
        empty array with the correct shape if not enough elements exist
        to form a single window.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> moving_sampling(values, 3)
    array([[ 1,  2,  3],
           [ 2,  3,  4],
           [ 3,  4,  5],
           [ 4,  5,  6],
           [ 5,  6,  7],
           [ 6,  7,  8],
           [ 7,  8,  9],
           [ 8,  9, 10]])

    Multi-step (dilated window):
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> moving_sampling(values, 3, 2)
    array([[ 1,  3,  5],
           [ 2,  4,  6],
           [ 3,  5,  7],
           [ 4,  6,  8],
           [ 5,  7,  9],
           [ 6,  8, 10]])

    More than one dimension:
    >>> values = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]).T
    >>> moving_sampling(values, 3, 2)
    array([[[ 1,  5],
            [ 3,  7],
            [ 5,  9]],
    <BLANKLINE>
           [[ 2,  6],
            [ 4,  8],
            [ 6, 10]]])

    Left-open sampling:
    >>> values = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]).T
    >>> moving_sampling(values, 3, 2, left_open_sampling=True)
    array([[[ 2,  6],
            [ 4,  8],
            [ 6, 10]]])

    Insufficient data:
    >>> values = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
    >>> moving_sampling(values, 3, 3)
    array([], shape=(0, 3, 2), dtype=values.dtype)
    """
    # --- Parameter validation ---
    if samples <= 0:
        raise ValueError("Parameter 'samples' must be a positive integer.")
    if step <= 0:
        raise ValueError("Parameter 'step' must be a positive integer.")

    if left_open_sampling and step > 1:
        values = values[step - 1:]

    # Calculate the number of windows that can be formed
    # This is the total number of elements in a window
    window_span = step * (samples - 1) + 1
    n_windows = len(values) - window_span + 1

    # --- Shape and Strides Calculation ---
    # New shape of the output array
    shape = (n_windows, samples) + values.shape[1:]

    if n_windows <= 0:
        # Not enough data to form even one window,
        # return a correctly shaped empty array
        return np.array([], dtype=values.dtype).reshape(shape)

    # New strides for the output array view
    # Stride 1: move to the next window (same as moving one element in the
    #           original array)
    # Stride 2: move to the next sample within a window (move `step` elements
    #           in the original)
    # Stride 3+: strides for the remaining dimensions (e.g., columns)
    strides = (values.strides[0], step * values.strides[0]) + values.strides[1:]

    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)


def moving_reduction(values: np.ndarray, samples: int, func: Callable,
                     step: int = 1, left_open_sampling: bool = False,
                     keep_size: bool = False, **kwargs) -> np.ndarray:
    """Applies a reduction function to sliding windows of an array.

    This function first generates sliding windows from the input `values` array
    using the `moving_sampling` function. It then applies a specified reduction
    function (e.g., mean, sum, std) to each window along the sampling axis.

    Parameters
    ----------
    values : np.ndarray
        The input array.
    samples : int
        The number of elements in each sliding window.
    func : Callable
        The reduction function to apply to each window (e.g., `np.mean`,
        `np.sum`, `np.std`). It will be called as `func(window, axis=1, **kwargs)`.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If True, skips the first `step - 1` elements of `values` before sampling.
        By default False.
    keep_size : bool, optional
        If True, the output array will have the same length as the input `values`
        array, with `np.nan` used for padding at the beginning where a full
        window cannot be formed. By default False.
    **kwargs
        Additional keyword arguments to pass to the reduction function `func`.

    Returns
    -------
    np.ndarray
        The array of reduced values. Its length will be shorter than `values`
        unless `keep_size` is True.

    See Also
    --------
    moving_sampling : The underlying function used to generate sliding windows.

    Examples
    --------
    >>> import numpy as np
    >>> # For examples involving np.nan, use a float dtype for the input array.
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    >>> moving_reduction(values, 3, np.mean)
    array([2., 3., 4., 5., 6., 7., 8., 9.])

    With additional arguments for `func`:
    >>> weights = np.array([0.2, 0.3, 0.5])
    >>> moving_reduction(values, 3, np.average, weights=weights)
    array([2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3])

    With `left_open_sampling`:
    >>> moving_reduction(values, 3, np.mean, step=2, left_open_sampling=True)
    array([4., 5., 6., 7., 8.])

    With `keep_size` to maintain original array length:
    >>> moving_reduction(values, 3, np.mean, keep_size=True)
    array([nan, nan,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """
    if len(values) == 0:
        return np.array([], dtype=float)

    sampled_windows = moving_sampling(
        values, samples, step, left_open_sampling=left_open_sampling
    )

    if sampled_windows.shape[0] == 0:
        if keep_size:
            # Return an array of NaNs with the same shape as the input
            return np.full(values.shape, np.nan, dtype=float)
        else:
            # return a correctly shaped empty array
            output_shape = (0,) + values.shape[1:]
            return np.array([], dtype=float).reshape(output_shape)

    # Apply the reduction function to each window along the samples axis(axis=1)
    result = func(sampled_windows, axis=1, **kwargs)

    if keep_size:
        # Pad the result with NaNs at the beginning to match input size
        pad_width = len(values) - len(result)

        # Ensure the output dtype can handle NaN
        output_dtype = (result.dtype
                        if np.issubdtype(result.dtype, np.floating) else float)

        # Create the padding array.
        # The shape must match the result's other dimensions.
        padding_shape = (pad_width,) + result.shape[1:]
        padding = np.full(padding_shape, np.nan, dtype=output_dtype)

        # Combine padding and result
        result = np.concatenate([padding,
                                 result.astype(output_dtype, copy=False)])

    return result


def moving_average(values: np.ndarray, samples: int, step: int = 1,
                   left_open_sampling: bool = False, keep_size: bool = False,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Computes the moving average over an array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.average`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `np.nan` to match input size,
        by default `False`.
    weights : array_like, optional
        An array of weights associated with the values in each window.
        Passed directly to :func:`numpy.average`.

    Returns
    -------
    np.ndarray
        The array of the moving average.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.average : The underlying function applied to each window.
    """
    return moving_reduction(
        values, samples, func=np.average, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling, weights=weights)


def moving_max(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False
               ) -> np.ndarray:
    """Computes the moving maximum over an array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.max`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `np.nan` to match input size, by
        default `False`.

    Returns
    -------
    np.ndarray
        The array of the moving maximum.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.max : The underlying function applied to each window.
    """
    return moving_reduction(
        values, samples, func=np.max, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling)


def moving_min(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False
               ) -> np.ndarray:
    """Computes the moving minimum over an array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.min`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `np.nan` to match input size, by
        default `False`.

    Returns
    -------
    np.ndarray
        The array of the moving minimum.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.min : The underlying function applied to each window.
    """
    # Note: Corrected the docstring from "maximum" to "minimum".
    return moving_reduction(
        values, samples, func=np.min, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling)


def moving_sum(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False
               ) -> np.ndarray:
    """Computes the moving sum over an array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.sum`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default
        `False`.
    keep_size : bool, optional
        If `True`, pads the output with `np.nan` to match input size, by
        default `False`.

    Returns
    -------
    np.ndarray
        The array of the moving sum.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.sum : The underlying function applied to each window.
    """
    return moving_reduction(
        values, samples, func=np.sum, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling)


def moving_std(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False,
               ddof: int = 0) -> np.ndarray:
    """Computes the moving standard deviation over an array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.std`.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `np.nan` to match input size, by
        default `False`.
    ddof : int, optional
        Delta Degrees of Freedom. The divisor used in calculations is
        `N - ddof`, where `N` is the number of elements. By default `ddof` is 0.
        Passed directly to :func:`numpy.std`.

    Returns
    -------
    np.ndarray
        The array of the moving standard deviation.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.std : The underlying function applied to each window.
    """
    return moving_reduction(
        values, samples, func=np.std, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling, ddof=ddof)


def moving_all(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False
               ) -> np.ndarray:
    """Computes the moving `all` (logical AND) over a boolean array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.all` to each sliding window.

    Parameters
    ----------
    values : np.ndarray
        Input array. Should be of boolean type.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `True`, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `False` to match input size, by
        default `False`.

    Returns
    -------
    np.ndarray
        A boolean array containing the result of the moving `all` operation.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.all : The underlying function applied to each window.

    Examples
    --------
    >>> import numpy as np
    >>> bool_arr = np.array([True, True, False, True, True, True])
    >>> # Windows: [T,T,F], [T,F,T], [F,T,T], [T,T,T]
    >>> moving_all(bool_arr, samples=3)
    array([False, False, False,  True])
    """
    return moving_reduction(
        values, samples, func=np.all, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling)


def moving_any(values: np.ndarray, samples: int, step: int = 1,
               left_open_sampling: bool = False, keep_size: bool = False
               ) -> np.ndarray:
    """Computes the moving `any` (logical OR) over a boolean array.

    This is a convenience wrapper around :func:`~moving_reduction` that applies
    :func:`numpy.any` to each sliding window.

    Parameters
    ----------
    values : np.ndarray
        Input array. Should be of boolean type.
    samples : int
        The size of the sliding window.
    step : int, optional
        The dilation step between elements within a window, by default 1.
    left_open_sampling : bool, optional
        If `Tru`e, the first `step - 1` elements are skipped, by default `False`.
    keep_size : bool, optional
        If `True`, pads the output with `False` to match input size, by
        default `False`.

    Returns
    -------
    np.ndarray
        A boolean array containing the result of the moving `any` operation.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.
    numpy.any : The underlying function applied to each window.

    Examples
    --------
    >>> import numpy as np
    >>> bool_arr = np.array([False, False, True, False, False, False])
    >>> # Windows: [F,F,T], [F,T,F], [T,F,F], [F,F,F]
    >>> moving_any(bool_arr, samples=3)
    array([ True,  True,  True, False])
    """
    return moving_reduction(
        values, samples, func=np.any, step=step, keep_size=keep_size,
        left_open_sampling=left_open_sampling)


def _np_change(values: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculates `last - first` along a given axis. Expects axis to have length 2.
    """
    if axis < 0:
        axis = values.ndim + axis

    # Creates slices for all preceding dimensions, e.g., (:, :, ...)
    pre_slices = (slice(None),) * axis

    # Selects the last element (index -1) and the first (index 0) along the
    # target axis
    return values[pre_slices + (-1,)] - values[pre_slices + (0,)]


def moving_change(values: np.ndarray, offset: int, keep_size: bool = False
                  ) -> np.ndarray:
    """
    Calculates the moving change between elements separated by a given offset.

    This function computes the difference between an element and a previous
    element separated by `offset`. The effective formula is:
    `output[i] = values[i] - values[i - offset]`.

    It is implemented by creating 2-element windows
    `[values[i-offset], values[i]]` and applying a subtraction.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    offset : int
        The period or lag to calculate the change over. Must be a positive
        integer. An `offset` of 1 is equivalent to `np.diff`.
    keep_size : bool, optional
        If True, pads the output with `np.nan` at the beginning to match the
        input array's size, by default False.

    Returns
    -------
    np.ndarray
        An array containing the moving changes.

    See Also
    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
    >>> # First result is values[2]-values[0] = 2-1 = 1
    >>> # Second result is values[3]-values[1] = 3-1 = 2
    >>> moving_change(values, 2)
    array([ 1.,  2.,  3.,  5.,  8., 13., 21., 34.])

    Keep-size:
    >>> moving_change(values, 2, keep_size=True)
    array([nan, nan,  1.,  2.,  3.,  5.,  8., 13., 21., 34.])

    """
    if offset <= 0:
        raise ValueError("Parameter 'offset' must be positive.")

    # We use `moving_reduction` with samples=2 and step=offset.
    # This creates windows of [value[i], value[i+offset]].
    # The `_np_change` function then calculates value[i+offset] - value[i].
    # The final result is effectively aligned such that it represents
    # value[t] - value[t-offset].
    return moving_reduction(
        values, samples=2, func=_np_change, step=offset, keep_size=keep_size)


def _np_change_rate(values: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculates `(last / first) - 1` along a given axis. Expects axis to have
    length 2. Handles division by zero by returning NaN.
    """
    if axis < 0:
        axis = values.ndim + axis

    pre_slices = (slice(None),) * axis

    first_values = values[pre_slices + (0,)]
    last_values = values[pre_slices + (-1,)]

    # Use `np.where` to create a safe denominator.
    # Where `first_values` is 0, use `np.nan` in the new array;
    # otherwise, use the original value.
    # `np.where` will automatically promote the output array's dtype to float
    # to hold the NaNs.
    denominator = np.where(first_values == 0, np.nan, first_values)

    ret = last_values / denominator - 1
    return ret


def moving_change_rate(values: np.ndarray, offset: int, keep_size: bool = False
                       ) -> np.ndarray:
    """
    Calculates the moving rate of change between elements separated by an offset.

    This function computes the rate of change based on the formula:
    `output[i] = (values[i] / values[i - offset]) - 1`.

    It handles division by zero by producing `np.nan` for those cases.

    Parameters
    ----------
    values : np.ndarray
        Input array. Must be of a float dtype to handle potential NaNs.
    offset : int
        The period or lag to calculate the rate of change over. Must be a
        positive integer.
    keep_size : bool, optional
        If True, pads the output with `np.nan` at the beginning to match the
        input array's size, by default False.

    Returns
    -------
    np.ndarray
        An array containing the moving rates of change.

    --------
    moving_reduction : The generic function for applying reduction functions
                       to sliding windows of an array.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
    >>> # First result is (values[2]/values[0]) - 1 = (2/1)-1 = 1.0
    >>> # Second result is (values[3]/values[1]) - 1 = (3/1)-1 = 2.0
    >>> moving_change_rate(values, 2)
    array([1.        , 2.        , 1.5       , 1.66666667, 1.6       ,
           1.625     , 1.61538462, 1.61904762])

    Keep-size:
    >>> moving_change_rate(values, 2, keep_size=True)
    array([       nan,        nan, 1.        , 2.        , 1.5       ,
           1.66666667, 1.6       , 1.625     , 1.61538462, 1.61904762])
    """
    if offset <= 0:
        raise ValueError("Parameter 'offset' must be positive.")

    # We use `moving_reduction` with samples=2 and step=offset.
    # This creates windows of [value[i], value[i+offset]].
    # The `_np_change` function then calculates value[i+offset] / value[i] - 1.
    # The final result is effectively aligned such that it represents
    # value[t] / value[t-offset] - 1.
    return moving_reduction(values, samples=2, func=_np_change_rate,
                            step=offset, keep_size=keep_size)
