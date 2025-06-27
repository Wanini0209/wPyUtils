# -*- coding: utf-8 -*-
"""Array Utilities.

This package consists of commonly used utilities for Numpy array operations.
"""

from ._shift import roll, shift
from .moving import MovingWindow

__all__ = ['MovingWindow']
__all__ += ['roll', 'shift']
