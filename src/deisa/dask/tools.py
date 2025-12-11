from typing import Iterable, Dict, Any

from distributed import Client


def validate_system_metadata(system_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a system_metadata dict.

    Requirements:
      - system_metadata must be a dict
      - must contain key 'connection' whose value is an instance of Client
      - must contain key 'nb_bridges' whose value is an int

    Raises:
      - TypeError if system_metadata is not a dict
      - ValueError if a required key is missing or has wrong type
    """
    if not isinstance(system_metadata, dict):
        raise TypeError("system_metadata must be a dict")
    # Check presence and type for 'connection'
    if 'connection' not in system_metadata or not isinstance(system_metadata['connection'], Client):
        raise ValueError("system_metadata must contain 'connection' of type Client")
    # Check presence and type for 'nb_bridges'
    if 'nb_bridges' not in system_metadata or not isinstance(system_metadata['nb_bridges'], int):
        raise ValueError("system_metadata must contain 'nb_bridges' of type int")
    return system_metadata


def validate_arrays_metadata(arrays_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate arrays_metadata structure.

    Expected:
        {
            "array_name": {
                "size": Iterable[int],
                "subsize": Iterable[int],
            },
            ...
        }
    """

    if not isinstance(arrays_metadata, dict):
        raise TypeError("arrays_metadata must be a dict")

    REQUIRED_KEYS = {"size", "subsize"}

    for array_name, metadata in arrays_metadata.items():

        # Top-level key must be str
        if not isinstance(array_name, str):
            raise TypeError("arrays_metadata keys must be strings")

        # Each value must be a dict
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata for '{array_name}' must be a dict")

        # Must contain exactly size + subsize (no extra keys)
        if set(metadata.keys()) != REQUIRED_KEYS:
            raise ValueError(
                f"metadata for '{array_name}' must contain exactly {REQUIRED_KEYS}"
            )

        # Validate size and subsize
        for key in REQUIRED_KEYS:
            value = metadata[key]

            if not isinstance(value, Iterable):
                raise TypeError(f"'{array_name}.{key}' must be an iterable of ints")

            # Must be iterable of ints
            if not all(isinstance(v, int) for v in value):
                raise TypeError(f"'{array_name}.{key}' must contain only ints")

        # size and subsize must match length
        if len(metadata["size"]) != len(metadata["subsize"]):
            raise ValueError(
                f"'size' and 'subsize' of '{array_name}' must have the same length"
            )

    return arrays_metadata
