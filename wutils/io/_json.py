# -*- coding: utf-8 -*-
"""Common used methods in or related with `json`.

Methods
-------
json_dump: Save object to JSON file
json_load: Load object from JSON file

Created on Mon Jun 23 14:18:04 2025

@author: WaNiNi
"""

import json
import os
from typing import Any, Optional, Union


def json_dump(obj: Any, filepath: Union[str, os.PathLike],
              indent: Optional[int] = 4, ensure_ascii: bool = False,
              encoding: str = 'utf-8') -> None:
    """Save object to JSON file.

    Parameters
    ----------
    obj : Any
        The object to serialize to JSON
    filepath : str or PathLike
        Path to the output JSON file
    indent : int, optional
        Number of spaces to use for indentation (default: 4)
    ensure_ascii : bool, optional
        If True, escape non-ASCII characters (default: False)
    encoding : str, optional
        File encoding (default: 'utf-8')

    Raises
    ------
    TypeError
        If object is not JSON serializable
    IOError
        If file cannot be written

    See Also
    --------
    json_dump : Save object to JSON file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
    except TypeError as e:
        raise TypeError(f"Object is not JSON serializable: {e}")
    except IOError as e:
        raise IOError(f"Cannot write to file {filepath}: {e}")


def json_load(filepath: Union[str, os.PathLike],
              encoding: str = 'utf-8') -> Any:
    """Load object from JSON file.

    Parameters
    ----------
    filepath : str or PathLike
        Path to the JSON file to load
    encoding : str, optional
        File encoding (default: 'utf-8')

    Returns
    -------
    Any
        The deserialized object from the JSON file

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    json.JSONDecodeError
        If the file contains invalid JSON
    IOError
        If file cannot be read

    See Also
    --------
    json.load : Standard library function for JSON deserialization
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {filepath}: {e.msg}",
                                   e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Cannot read file {filepath}: {e}")


# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = {
        "name": "WaNiNi",
        "project": "JSON Utils",
        "date": "2025-06-23",
        "features": ["json_dump", "json_load"],
        "metadata": {
            "version": "1.0",
            "encoding": "utf-8"
        }
    }

    # Save to JSON
    json_dump(sample_data, "D:/example.json")
    print("Data saved to example.json")

    # Load from JSON
    loaded_data = json_load("D:/example.json")
    print("Loaded data:", loaded_data)
