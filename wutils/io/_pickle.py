# -*- coding: utf-8 -*-
"""Common used methods in or related with `pickle`.

Methods
-------
pickle_dump: Save object to pickle file
pickle_load: Load object from pickle file

Created on Mon Jun 23 09:59:43 2025

@author: WaNiNi
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Union


def pickle_dump(obj: Any, filepath: Union[str, Path],
                protocol: Optional[int] = None) -> None:
    """Save object to pickle file.

    Parameters
    ----------
    obj : Any
        Object to be saved
    filepath : str or Path
        File path to save the pickle file
    protocol : int, optional
        Pickle protocol version, by default None (uses highest available)

    See Also
    --------
    pickle.dump

    Examples
    --------
    >>> data = {'key': 'value', 'numbers': [1, 2, 3]}
    >>> pickle_dump(data, 'data.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as fout:
        pickle.dump(obj, fout, protocol=protocol)


def pickle_load(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file.

    Parameters
    ----------
    filepath : str or Path
        File path of the pickle file

    Returns
    -------
    Any
        Loaded object

    See Also
    --------
    pickle.load

    Examples
    --------
    >>> data = pickle_load('data.pkl')
    >>> print(data)
    {'key': 'value', 'numbers': [1, 2, 3]}
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as fin:
        return pickle.load(fin)
