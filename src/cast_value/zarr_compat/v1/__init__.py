"""cast_value codec implementations targeting the zarr-python 3.x codec API.

The "V1" in class names (e.g. ``CastValueNumpyV1``) refers to this package's
adaptation of the zarr-python ``ArrayArrayCodec`` ABC as defined in zarr 3.x.
If zarr-python introduces a new codec interface in a future major version,
a ``v2`` package will be added alongside this one with updated class names
(e.g. ``CastValueNumpyV2``).
"""

from __future__ import annotations

from cast_value.zarr_compat._parsing import parse_map_entries
from cast_value.zarr_compat.v1.numpy_codec import CastValueNumpyV1
from cast_value.zarr_compat.v1.rust_codec import CastValueRustV1

__all__ = [
    "CastValueNumpyV1",
    "CastValueRustV1",
    "parse_map_entries",
]
