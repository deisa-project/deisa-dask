import pytest
from distributed import Client

from deisa.dask.tools import validate_system_metadata, validate_arrays_metadata


class TestSystemMetadata:
    def test_valid_metadata(self):
        conn = Client()
        metadata = {"connection": conn, "nb_bridges": 3}
        assert validate_system_metadata(metadata) == metadata

    def test_not_a_dict(self):
        with pytest.raises(TypeError):
            validate_system_metadata("not a dict")

    def test_missing_connection_key(self):
        metadata = {"nb_bridges": 1}
        with pytest.raises(ValueError):
            validate_system_metadata(metadata)

    def test_wrong_connection_type(self):
        metadata = {"connection": 123, "nb_bridges": 1}
        with pytest.raises(ValueError):
            validate_system_metadata(metadata)

    def test_missing_nb_bridges_key(self):
        metadata = {"connection": Client()}
        with pytest.raises(ValueError):
            validate_system_metadata(metadata)

    def test_wrong_nb_bridges_type(self):
        metadata = {"connection": Client(), "nb_bridges": "not-an-int"}
        with pytest.raises(ValueError):
            validate_system_metadata(metadata)


class TestArraysMetadata:
    def test_valid_arrays_metadata(self):
        metadata = {
            'global_t': {
                'size': [20, 20],
                'subsize': [10, 10]
            },
            'global_p': {
                'size': [100, 100],
                'subsize': [50, 50]
            }
        }

        assert validate_arrays_metadata(metadata) == metadata

    def test_not_a_dict(self):
        with pytest.raises(TypeError):
            validate_arrays_metadata("not-a-dict")

    def test_key_not_string(self):
        metadata = {
            42: {
                'size': [1, 1],
                'subsize': [1, 1]
            }
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_value_not_dict(self):
        metadata = {
            'global_t': "not-a-dict"
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_missing_required_keys(self):
        metadata = {
            'global_t': {
                'size': [10, 10]
            }
        }
        with pytest.raises(ValueError):
            validate_arrays_metadata(metadata)

    def test_extra_keys(self):
        metadata = {
            'global_t': {
                'size': [10, 10],
                'subsize': [5, 5],
                'foo': 123
            }
        }
        with pytest.raises(ValueError):
            validate_arrays_metadata(metadata)

    def test_size_not_iterable(self):
        metadata = {
            'global_t': {
                'size': 123,
                'subsize': [5, 5]
            }
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_subsize_not_iterable(self):
        metadata = {
            'global_t': {
                'size': [10, 10],
                'subsize': 123
            }
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_size_contains_non_int(self):
        metadata = {
            'global_t': {
                'size': [10, "x"],
                'subsize': [5, 5]
            }
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_subsize_contains_non_int(self):
        metadata = {
            'global_t': {
                'size': [10, 10],
                'subsize': ["a", 5]
            }
        }
        with pytest.raises(TypeError):
            validate_arrays_metadata(metadata)

    def test_size_subsize_mismatch(self):
        metadata = {
            'global_t': {
                'size': [10, 10, 10],
                'subsize': [5, 5]
            }
        }
        with pytest.raises(ValueError):
            validate_arrays_metadata(metadata)
