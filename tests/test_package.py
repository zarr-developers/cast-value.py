from __future__ import annotations

import importlib.metadata

import cast_value as m


def test_version() -> None:
    assert importlib.metadata.version("cast_value") == m.__version__
