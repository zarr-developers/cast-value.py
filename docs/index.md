# cast-value

Python implementation of the
[`cast_value` codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value)
for [Zarr V3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html).

The `cast_value` codec converts array elements between numeric data types during
encoding and decoding, with configurable rounding, out-of-range handling, and
explicit scalar mappings.

## Installation

<!--pytest.mark.skip-->

```bash
pip install cast-value
```

For the optional Rust-accelerated backend:

<!--pytest.mark.skip-->

```bash
pip install cast-value[rs]
```

## Quick start

```python
import numpy as np
import zarr

from cast_value import CastValueNumpyV1

codec = CastValueNumpyV1(
    data_type="uint8",
    rounding="nearest-even",
    out_of_range="clamp",
)

# float64 values with fractional parts and values outside [0, 255]
data = np.array([1.5, 2.5, 3.5, 100.7, 255.9, -3.0, 999.0], dtype=np.float64)
arr = zarr.create_array(data=data, store={}, filters=codec)

# Read back: fractional values are rounded (nearest-even),
# out-of-range values are clamped to [0, 255]
result = arr[:]
print(result)
```

<!--pytest-codeblocks:expected-output-->

```
[  2.   2.   4. 101. 255.   0. 255.]
```

## Backends

Two backends are available:

- **`CastValueNumpyV1`** — Pure Python + NumPy. Always available.
- **`CastValueRustV1`** — Rust via
  [cast-value-rs](https://pypi.org/project/cast-value-rs/). Faster for
  non-default rounding modes and SIMD-accelerated float-to-integer casts with
  clamping, with more efficient memory usage.

Both implement the same codec interface and produce identical results.
