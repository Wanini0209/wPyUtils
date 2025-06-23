# -*- coding: utf-8 -*-
"""Unit tests for pickle utilities module.

Test cases for `json_dump` and `json_load` functions.

Created on Mon Jun 23 14:32:55 2025

@author: WaNiNi
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from wutils.io import json_dump, json_load


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "測試資料",
        "number": 42,
        "float_val": 3.14,
        "boolean": True,
        "null_val": None,
        "list": [1, 2, 3, "four"],
        "nested": {
            "key": "value",
            "chinese": "中文測試"
        }
    }


class TestJsonDump:
    """Test cases for pickle_dump function."""

    def test_json_dump_basic(self, temp_dir, sample_data):
        """Test basic json_dump functionality."""
        filepath = os.path.join(temp_dir, "test.json")

        # Should not raise any exception
        json_dump(sample_data, filepath)

        # File should exist
        assert os.path.exists(filepath)

        # File should contain valid JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert loaded_data == sample_data

    def test_json_dump_with_pathlib(self, temp_dir, sample_data):
        """Test json_dump with pathlib.Path."""
        filepath = Path(temp_dir) / "test_pathlib.json"

        json_dump(sample_data, filepath)

        assert filepath.exists()

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert loaded_data == sample_data

    def test_json_dump_custom_indent(self, temp_dir, sample_data):
        """Test json_dump with custom indentation."""
        filepath = os.path.join(temp_dir, "test_indent.json")

        json_dump(sample_data, filepath, indent=2)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that the indentation is applied
        assert '  "name"' in content  # 2-space indentation

    def test_json_dump_no_indent(self, temp_dir, sample_data):
        """Test json_dump without indentation."""
        filepath = os.path.join(temp_dir, "test_no_indent.json")

        json_dump(sample_data, filepath, indent=None)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should be a single line
        assert '\n' not in content.strip()

    def test_json_dump_ensure_ascii(self, temp_dir):
        """Test json_dump with ensure_ascii option."""
        data = {"chinese": "中文", "english": "English"}
        filepath = os.path.join(temp_dir, "test_ascii.json")

        json_dump(data, filepath, ensure_ascii=True)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Chinese characters should be escaped
        assert "\\u" in content
        assert "中文" not in content

    def test_json_dump_create_directory(self, temp_dir, sample_data):
        """Test json_dump creates directory if it doesn't exist."""
        nested_dir = os.path.join(temp_dir, "nested", "deep", "directory")
        filepath = os.path.join(nested_dir, "test.json")

        json_dump(sample_data, filepath)

        assert os.path.exists(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert loaded_data == sample_data

    def test_json_dump_non_serializable(self, temp_dir):
        """Test json_dump with non-serializable object."""
        class NonSerializable:
            pass

        data = {"object": NonSerializable()}
        filepath = os.path.join(temp_dir, "test.json")

        with pytest.raises(TypeError, match="Object is not JSON serializable"):
            json_dump(data, filepath)

    def test_json_dump_invalid_path(self, sample_data):
        """Test json_dump with invalid file path."""
        # Try to write to a non-existent drive (on Windows) or protected directory
        invalid_path = ("/root/protected/test.json" if os.name != 'nt'
                        else "Z:\\invalid\\test.json")

        with pytest.raises(IOError, match="Cannot write to file"):
            json_dump(sample_data, invalid_path)


class TestPickleLoad:
    """Test cases for json_load function."""

    def test_json_load_basic(self, temp_dir, sample_data):
        """Test basic json_load functionality."""
        filepath = os.path.join(temp_dir, "test.json")

        # First save the data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False)

        # Then load it
        loaded_data = json_load(filepath)

        assert loaded_data == sample_data

    def test_json_load_with_pathlib(self, temp_dir, sample_data):
        """Test json_load with pathlib.Path."""
        filepath = Path(temp_dir) / "test_pathlib.json"

        # Save data first
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False)

        # Load with pathlib.Path
        loaded_data = json_load(filepath)

        assert loaded_data == sample_data

    def test_json_load_file_not_found(self, temp_dir):
        """Test json_load with non-existent file."""
        filepath = os.path.join(temp_dir, "nonexistent.json")

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            json_load(filepath)

    def test_json_load_invalid_json(self, temp_dir):
        """Test json_load with invalid JSON content."""
        filepath = os.path.join(temp_dir, "invalid.json")

        # Write invalid JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content }")

        with pytest.raises(json.JSONDecodeError, match="Invalid JSON in file"):
            json_load(filepath)

    def test_json_load_empty_file(self, temp_dir):
        """Test json_load with empty file."""
        filepath = os.path.join(temp_dir, "empty.json")

        # Create empty file
        with open(filepath, 'w', encoding='utf-8'):
            pass

        with pytest.raises(json.JSONDecodeError, match="Invalid JSON in file"):
            json_load(filepath)

    def test_json_load_different_encoding(self, temp_dir):
        """Test json_load with different encoding."""
        data = {"test": "測試"}
        filepath = os.path.join(temp_dir, "test_encoding.json")

        # Save with UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        # Load with UTF-8
        loaded_data = json_load(filepath, encoding='utf-8')

        assert loaded_data == data


class TestPickleRoundtrip:
    """Test complete roundtrip operations."""

    def test_roundtrip_dump_load(self, temp_dir, sample_data):
        """Test complete roundtrip: dump then load."""
        filepath = os.path.join(temp_dir, "roundtrip.json")

        # Save data
        json_dump(sample_data, filepath)

        # Load data
        loaded_data = json_load(filepath)

        # Should be identical
        assert loaded_data == sample_data

    def test_roundtrip_complex_data(self, temp_dir):
        """Test roundtrip with complex nested data."""
        complex_data = {
            "users": [
                {
                    "id": 1,
                    "name": "張三",
                    "email": "zhang@example.com",
                    "active": True,
                    "profile": {
                        "age": 30,
                        "city": "台北",
                        "hobbies": ["reading", "coding", "music"]
                    }
                },
                {
                    "id": 2,
                    "name": "李四",
                    "email": "li@example.com",
                    "active": False,
                    "profile": {
                        "age": 25,
                        "city": "高雄",
                        "hobbies": ["gaming", "sports"]
                    }
                }
            ],
            "metadata": {
                "version": "1.0",
                "created_at": "2025-06-23T14:18:04Z",
                "total_users": 2
            }
        }

        filepath = os.path.join(temp_dir, "complex.json")

        # Roundtrip test
        json_dump(complex_data, filepath)
        loaded_data = json_load(filepath)

        assert loaded_data == complex_data

    @pytest.mark.parametrize("indent", [None, 0, 2, 4, 8])
    def test_json_dump_various_indents(self, temp_dir, sample_data, indent):
        """Test json_dump with various indentation values."""
        filepath = os.path.join(temp_dir, f"test_indent_{indent}.json")

        json_dump(sample_data, filepath, indent=indent)

        # Should be able to load back
        loaded_data = json_load(filepath)
        assert loaded_data == sample_data

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_json_dump_ascii_options(self, temp_dir, ensure_ascii):
        """Test json_dump with different ASCII options."""
        data = {"chinese": "中文測試", "number": 123}
        filepath = os.path.join(temp_dir, f"test_ascii_{ensure_ascii}.json")

        json_dump(data, filepath, ensure_ascii=ensure_ascii)

        # Should be able to load back
        loaded_data = json_load(filepath)
        assert loaded_data == data


# Integration tests
class TestJsonUtilsIntegration:
    """Integration tests for JSON utilities."""

    def test_multiple_files_same_directory(self, tmp_path):
        """Test handling multiple JSON files in the same directory."""
        data1 = {"file": 1, "content": "first"}
        data2 = {"file": 2, "content": "second"}

        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"

        # Save both files
        json_dump(data1, file1)
        json_dump(data2, file2)

        # Load both files
        loaded1 = json_load(file1)
        loaded2 = json_load(file2)

        assert loaded1 == data1
        assert loaded2 == data2
        assert loaded1 != loaded2

    def test_overwrite_existing_file(self, tmp_path):
        """Test overwriting an existing JSON file."""
        filepath = tmp_path / "overwrite.json"

        # Save first version
        data1 = {"version": 1}
        json_dump(data1, filepath)

        # Verify first version
        assert json_load(filepath) == data1

        # Save second version (overwrite)
        data2 = {"version": 2}
        json_dump(data2, filepath)

        # Verify second version
        assert json_load(filepath) == data2
        assert json_load(filepath) != data1
