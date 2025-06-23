# -*- coding: utf-8 -*-
"""Unit tests for pickle utilities module.

Test cases for pickle_dump and pickle_load functions.

Created on Mon Jun 23 10:18:30 2025

@author: WaNiNi
"""

import pickle
import shutil
import tempfile
from pathlib import Path

import pytest

from wutils.io import pickle_dump, pickle_load


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def test_file(temp_dir):
    """Create a test file path."""
    return temp_dir / "test.pkl"


@pytest.fixture
def simple_data():
    """Simple test data."""
    return {"key": "value", "numbers": [1, 2, 3]}


@pytest.fixture
def complex_data():
    """Complex nested test data."""
    return {
        "nested": {"inner": [1, 2, 3]},
        "list": [{"a": 1}, {"b": 2}],
        "tuple": (1, 2, 3),
        "set": {1, 2, 3}
    }


class TestPickleDump:
    """Test cases for pickle_dump function."""

    def test_dump_simple_data(self, simple_data, test_file):
        """Test pickle_dump with simple data."""
        pickle_dump(simple_data, test_file)

        # Verify file was created
        assert test_file.exists()

        # Verify content by loading with standard pickle
        with open(test_file, 'rb') as f:
            loaded_data = pickle.load(f)

        assert loaded_data == simple_data

    def test_dump_complex_data(self, complex_data, test_file):
        """Test pickle_dump with complex nested data."""
        pickle_dump(complex_data, test_file)

        # Verify file was created
        assert test_file.exists()

        # Verify content
        with open(test_file, 'rb') as f:
            loaded_data = pickle.load(f)

        assert loaded_data == complex_data

    def test_dump_with_protocol(self, test_file):
        """Test pickle_dump with specific protocol version."""
        test_data = {"protocol_test": True}

        # Test with protocol 2
        pickle_dump(test_data, test_file, protocol=2)

        assert test_file.exists()

        with open(test_file, 'rb') as f:
            loaded_data = pickle.load(f)

        assert loaded_data == test_data

    def test_dump_creates_directory(self, simple_data, temp_dir):
        """Test that pickle_dump creates parent directories if they don't exist."""
        nested_path = temp_dir / "nested" / "deep" / "test.pkl"

        pickle_dump(simple_data, nested_path)

        # Verify directory was created
        assert nested_path.parent.exists()
        assert nested_path.exists()

        # Verify content
        with open(nested_path, 'rb') as f:
            loaded_data = pickle.load(f)

        assert loaded_data == simple_data

    def test_dump_with_string_path(self, simple_data, test_file):
        """Test pickle_dump with string path instead of Path object."""
        str_path = str(test_file)

        pickle_dump(simple_data, str_path)

        assert Path(str_path).exists()

        with open(str_path, 'rb') as f:
            loaded_data = pickle.load(f)

        assert loaded_data == simple_data

    def test_dump_overwrites_existing_file(self, test_file):
        """Test that pickle_dump overwrites existing files."""
        # Save initial data
        initial_data = {"initial": True}
        pickle_dump(initial_data, test_file)

        # Verify initial data
        loaded_initial = pickle_load(test_file)
        assert loaded_initial == initial_data

        # Overwrite with new data
        new_data = {"overwritten": True}
        pickle_dump(new_data, test_file)

        # Verify new data
        loaded_new = pickle_load(test_file)
        assert loaded_new == new_data
        assert loaded_new != initial_data

    @pytest.mark.parametrize("empty_data", [
        {},          # empty dict
        [],          # empty list
        (),          # empty tuple
        set(),       # empty set
        "",          # empty string
        None         # None
    ])
    def test_dump_empty_data(self, empty_data, temp_dir):
        """Test pickle_dump with empty data structures."""
        test_file = temp_dir / f"empty_{id(empty_data)}.pkl"

        pickle_dump(empty_data, test_file)
        loaded_data = pickle_load(test_file)

        assert loaded_data == empty_data

    def test_dump_invalid_protocol(self, test_file):
        """Test pickle_dump with invalid protocol version."""
        test_data = {"test": True}

        # Test with invalid protocol (too high)
        with pytest.raises((ValueError, pickle.PicklingError)):
            pickle_dump(test_data, test_file, protocol=999)


class TestPickleLoad:
    """Test cases for pickle_load function."""

    def test_load_simple_data(self, simple_data, test_file):
        """Test pickle_load with simple data."""
        # First save data using standard pickle
        with open(test_file, 'wb') as f:
            pickle.dump(simple_data, f)

        # Load using our function
        loaded_data = pickle_load(test_file)

        assert loaded_data == simple_data

    def test_load_complex_data(self, complex_data, test_file):
        """Test pickle_load with complex data."""
        # First save data using standard pickle
        with open(test_file, 'wb') as f:
            pickle.dump(complex_data, f)

        # Load using our function
        loaded_data = pickle_load(test_file)

        assert loaded_data == complex_data

    def test_load_with_string_path(self, simple_data, test_file):
        """Test pickle_load with string path."""
        str_path = str(test_file)

        # Save data first
        with open(str_path, 'wb') as f:
            pickle.dump(simple_data, f)

        # Load using string path
        loaded_data = pickle_load(str_path)

        assert loaded_data == simple_data

    def test_load_nonexistent_file(self, temp_dir):
        """Test pickle_load with non-existent file raises appropriate error."""
        nonexistent_file = temp_dir / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            pickle_load(nonexistent_file)

    def test_load_corrupted_file(self, temp_dir):
        """Test pickle_load with corrupted pickle file."""
        corrupted_file = temp_dir / "corrupted.pkl"

        # Create a file with invalid pickle data
        with open(corrupted_file, 'wb') as f:
            f.write(b"this is not valid pickle data")

        with pytest.raises((pickle.UnpicklingError, EOFError)):
            pickle_load(corrupted_file)


class TestPickleRoundtrip:
    """Test complete roundtrip operations."""

    def test_roundtrip_dump_load(self, test_file):
        """Test complete roundtrip: dump then load."""
        test_data = {
            "string": "hello world",
            "number": 42,
            "list": [1, 2, 3, 4, 5],
            "dict": {"nested": True},
            "none": None,
            "bool": True
        }

        # Dump data
        pickle_dump(test_data, test_file)

        # Load data
        loaded_data = pickle_load(test_file)

        # Verify they're identical
        assert loaded_data == test_data
        assert loaded_data is not test_data  # Different objects

    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4])
    def test_roundtrip_different_protocols(self, test_file, protocol):
        """Test roundtrip with different pickle protocols."""
        test_data = {"protocol": protocol, "data": [1, 2, 3]}

        # Skip if protocol not supported
        if protocol > pickle.HIGHEST_PROTOCOL:
            pytest.skip(f"Protocol {protocol} not supported")

        pickle_dump(test_data, test_file, protocol=protocol)
        loaded_data = pickle_load(test_file)

        assert loaded_data == test_data


# Performance and stress tests
class TestPicklePerformance:
    """Performance and stress tests."""

    @pytest.mark.slow
    def test_large_data_roundtrip(self, test_file):
        """Test with large data structure."""
        # Create large test data
        large_data = {
            f"key_{i}": list(range(100)) for i in range(1000)
        }

        pickle_dump(large_data, test_file)
        loaded_data = pickle_load(test_file)

        assert loaded_data == large_data

    @pytest.mark.slow
    def test_deeply_nested_data(self, test_file):
        """Test with deeply nested data structure."""
        # Create deeply nested structure
        nested_data = {"level_0": {}}
        current = nested_data["level_0"]

        for i in range(1, 100):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        current["data"] = "deep_value"

        pickle_dump(nested_data, test_file)
        loaded_data = pickle_load(test_file)

        assert loaded_data == nested_data


# Integration tests
class TestPickleIntegration:
    """Integration tests with real-world scenarios."""

    def test_multiple_files_same_directory(self, temp_dir):
        """Test handling multiple pickle files in same directory."""
        files_data = {
            "file1.pkl": {"type": "config", "value": 1},
            "file2.pkl": {"type": "data", "value": [1, 2, 3]},
            "file3.pkl": {"type": "model", "value": {"weights": [0.1, 0.2]}}
        }

        # Save all files
        for filename, data in files_data.items():
            filepath = temp_dir / filename
            pickle_dump(data, filepath)

        # Load and verify all files
        for filename, expected_data in files_data.items():
            filepath = temp_dir / filename
            loaded_data = pickle_load(filepath)
            assert loaded_data == expected_data

    def test_nested_directory_structure(self, temp_dir):
        """Test with nested directory structure."""
        structure = {
            "models/linear/model.pkl": {"type": "linear", "params": [1, 2]},
            "models/neural/model.pkl": {"type": "neural", "layers": 3},
            "data/train/features.pkl": {"features": [1, 2, 3]},
            "data/test/features.pkl": {"features": [4, 5, 6]}
        }

        # Save all files (directories will be created automatically)
        for rel_path, data in structure.items():
            filepath = temp_dir / rel_path
            pickle_dump(data, filepath)

        # Verify all files exist and contain correct data
        for rel_path, expected_data in structure.items():
            filepath = temp_dir / rel_path
            assert filepath.exists()
            loaded_data = pickle_load(filepath)
            assert loaded_data == expected_data
