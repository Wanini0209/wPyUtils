# -*- coding: utf-8 -*-
"""Array Utilities.

This package consists of commonly used utilities for Numpy array operations.
"""

from ._moving import (
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

__all__ = ['moving_all', 'moving_any', 'moving_average', 'moving_change',
           'moving_change_rate', 'moving_max', 'moving_min', 'moving_reduction',
           'moving_sampling', 'moving_std', 'moving_sum']
