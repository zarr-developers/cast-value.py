"""Codec metadata parsing utilities shared across zarr API versions.

These depend on zarr's dtype system (``ZDType.from_json_scalar``) but not
on any specific version of the zarr codec API (``ArrayArrayCodec``, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

    from cast_value.types import ScalarMapEntry, ScalarMapJSON, ScalarMapLike

_DIRECTIONS = ("encode", "decode")


def parse_scalar_map(
    data: ScalarMapJSON | ScalarMapLike | None,
) -> ScalarMapJSON | None:
    """Normalize a scalar map to its canonical JSON form.

    For each of the ``"encode"`` and ``"decode"`` keys, accepts either a
    mapping of source -> target or an iterable of ``(source, target)`` pairs,
    and normalizes to a list of pairs -- the form the cast_value spec uses
    and ``to_dict`` serializes. Malformed maps raise here, at codec
    construction time, rather than later at encode/decode time.
    """
    if data is None:
        return None
    if not isinstance(data, Mapping):
        msg = f"scalar_map must be a mapping, got {type(data).__name__}"
        raise TypeError(msg)
    unknown = {key for key in data if key not in _DIRECTIONS}
    if unknown:
        msg = (
            f"scalar_map keys must be 'encode' or 'decode', "
            f"got {sorted(map(str, unknown))}"
        )
        raise ValueError(msg)
    result: ScalarMapJSON = {}
    for direction in _DIRECTIONS:
        if direction not in data:
            continue
        pairs = data[direction]  # type: ignore[literal-required]
        items = pairs.items() if isinstance(pairs, Mapping) else pairs
        entries: list[tuple[object, object]] = []
        for entry in items:
            try:
                pair = tuple(entry)
            except TypeError:
                msg = (
                    f"scalar_map {direction!r} entry {entry!r} is not a "
                    f"(source, target) pair"
                )
                raise TypeError(msg) from None
            if len(pair) != 2:
                msg = (
                    f"scalar_map {direction!r} entry {entry!r} must have "
                    f"exactly 2 elements, got {len(pair)}"
                )
                raise ValueError(msg)
            entries.append((pair[0], pair[1]))
        result[direction] = entries  # type: ignore[literal-required]
    return result


def extract_raw_map(
    data: ScalarMapJSON | None, direction: str
) -> dict[str, str] | None:
    """Extract raw string mapping from scalar_map JSON for 'encode' or 'decode'."""
    if data is None:
        return None
    raw: dict[str, str] = {}
    pairs = data.get(direction, [])
    for src, tgt in pairs:  # type: ignore[attr-defined]
        raw[str(src)] = str(tgt)
    return raw or None


def parse_map_entries(
    mapping: Mapping[str, str],
    src_dtype: ZDType[TBaseDType, TBaseScalar],
    tgt_dtype: ZDType[TBaseDType, TBaseScalar],
) -> tuple[ScalarMapEntry, ...]:
    """Pre-parse a scalar map dict into a tuple of (src, tgt) pairs.

    Each entry's source value is deserialized using ``src_dtype`` and its target
    value using ``tgt_dtype``, preserving full precision for both data types.
    """
    return tuple(  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
        (
            src_dtype.from_json_scalar(src_str, zarr_format=3),
            tgt_dtype.from_json_scalar(tgt_str, zarr_format=3),
        )
        for src_str, tgt_str in mapping.items()
    )
