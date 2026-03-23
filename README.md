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

Python implementation of the `cast_value` codec for Zarr.

## `cast_value` codec

The `cast_value` codec defines an operation for safely converting an array from one numeric data type to another. 