"""Tests for the zarr.codecs entrypoint and the _entrypoint module."""

from __future__ import annotations

from zarr.registry import get_codec_class

from cast_value.zarr_compat.v1._entrypoint import CastValue
from cast_value.zarr_compat.v1.numpy_codec import CastValueNumpyV1
from cast_value.zarr_compat.v1.rust_codec import CastValueRustV1

try:
    import cast_value_rs  # noqa: F401

    _has_rust = True
except ModuleNotFoundError:
    _has_rust = False


def test_entrypoint_resolves_to_known_class() -> None:
    """Test that CastValue is one of the two known codec implementations."""
    assert CastValue is CastValueRustV1 or CastValue is CastValueNumpyV1


def test_entrypoint_prefers_rust_when_available() -> None:
    """Test that CastValue resolves to the Rust backend when cast-value-rs is installed."""
    if _has_rust:
        assert CastValue is CastValueRustV1
    else:
        assert CastValue is CastValueNumpyV1


def test_zarr_discovers_codec_via_entrypoint() -> None:
    """Test that zarr can discover the cast_value codec through the entrypoint system."""
    cls = get_codec_class("cast_value")
    assert cls is CastValue
