# cast-value

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/zarr-developers/cast-value/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/zarr-developers/cast-value/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/cast-value
[conda-link]:               https://github.com/conda-forge/cast-value-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/zarr-developers/cast-value/discussions
[pypi-link]:                https://pypi.org/project/cast-value/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cast-value
[pypi-version]:             https://img.shields.io/pypi/v/cast-value
[rtd-badge]:                https://readthedocs.org/projects/cast-value/badge/?version=latest
[rtd-link]:                 https://cast-value.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/zarr-developers/cast-value/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/zarr-developers/cast-value

<!-- prettier-ignore-end -->

# cast-value.py

A Python implementation of the `cast_value` codec for [Zarr](https://zarr.dev/),
with [zarr-python](https://zarr.readthedocs.io/en/stable/) integration.

## What

The `cast_value` codec defines how to _safely_ convert arrays between integer
and float data types. In Zarr terminology, this codec is an "array -> array"
codec, which means its input and output are both arrays.

You can find the
[specification for this codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value)
in the
[zarr-extensions repository](https://github.com/zarr-developers/zarr-extensions).

## Why

This codec is commonly used for lossy data compression: when decoded data should
be high-precision floats, but the absolute range of the values fits within the
range of a smaller integer data type, then encoding the floats as ints before
writing data can vastly shrink the stored values.

For example, if your data is a sequence of `float64` values like
`[100.1, 120.3, 125.5]`, storing those values as `uint8`, e.g.
`[100, 120, 125]`, offers 8-fold reduction in storage size, provided the
precision lost due to rounding is acceptable.

## Installation

<!--pytest.mark.skip-->

```bash
pip install cast-value
```

For the optional Rust backend (faster for large arrays):

<!--pytest.mark.skip-->

```bash
pip install 'cast-value[rs]'
```

## Usage

The codec is automatically registered with zarr-python via the `zarr.codecs`
entrypoint. When `cast-value[rs]` is installed, the Rust backend is used;
otherwise it falls back to the pure-NumPy backend.

```python
import numpy as np
import zarr
import zarr.storage

from cast_value import CastValueNumpyV1

codec = CastValueNumpyV1(
    data_type="uint8",
    rounding="nearest-even",
    out_of_range="clamp",
)

# Write float64 data -- values are rounded and clamped to [0, 255]
data = np.array([1.5, 100.7, 255.9, -3.0], dtype=np.float64)
arr = zarr.create_array(store={}, data=data, filters=codec)

# Read it back -- comes back as float64, but with uint8 precision
result = arr[:]

print(f"Array dtype: {arr.dtype}")
print(f"Values written: {data}")
print(f"Values read:    {result}")
```

<!--pytest-codeblocks:expected-output-->

```
Array dtype: float64
Values written: [  1.5 100.7 255.9  -3. ]
Values read:    [  2. 101. 255.   0.]
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Setup

<!--pytest.mark.skip-->

```bash
# Clone the repo
git clone https://github.com/zarr-developers/cast-value.py.git
cd cast-value.py

# Install dev dependencies (includes test + benchmark deps)
uv sync --group dev
```

### Running tests

<!--pytest.mark.skip-->

```bash
# Run the full test suite
uv run pytest tests

# Run with coverage
uv run pytest tests --cov=cast_value --cov-report=term-missing
```

### Running all checks

<!--pytest.mark.skip-->

```bash
# Run the full CI suite locally (tests, linting, type checking)
uvx nox

# Or run just linting and type checking
uvx prek
```

### Building docs

<!--pytest.mark.skip-->

```bash
uv sync --group docs
uv run zensical build
# Output is in site/
```

### Running examples

<!--pytest.mark.skip-->

```bash
uv run python examples/zarr_integration/zarr_cast_value.py
uv run python examples/benchmarks/bench_numpy_vs_rust.py
```

## Who

Davis Bennett (@d-v-b)
