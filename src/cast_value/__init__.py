"""Copyright (c) 2026 Davis Bennett. All rights reserved.

cast-value: Python implementation of the `cast_value` codec.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["CastValueNumpyV1", "CastValueRustV1", "__version__"]


def __getattr__(name: str) -> type:
    """Lazy-import zarr-dependent codec classes.

    This allows ``import cast_value`` to succeed even when zarr is not
    installed.  The codec classes are only resolved when accessed.
    """
    if name in ("CastValueNumpyV1", "CastValueRustV1"):
        from cast_value.zarr_compat.v1 import (  # noqa: PLC0415
            CastValueNumpyV1,
            CastValueRustV1,
        )

        globals()["CastValueNumpyV1"] = CastValueNumpyV1
        globals()["CastValueRustV1"] = CastValueRustV1
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
