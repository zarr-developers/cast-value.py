from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

In = TypeVar("In")
Ex = TypeVar("Ex")


def _default_eq(a: Any, b: Any) -> bool:
    """Default equality that handles numpy arrays via byte-level comparison.

    Viewing as bytes ensures that distinct NaN bit patterns are compared
    correctly: two arrays are equal only if every byte (including the
    sign/payload bits of NaNs) is identical.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return (a.view(np.uint8) == b.view(np.uint8)).all().item()
    return a == b


@dataclass(frozen=True, kw_only=True)
class Expect(Generic[In, Ex]):
    id: str
    input: In
    expected: Ex
    eq: Callable[[Ex, Ex], bool] = _default_eq

    @property
    def __name__(self) -> str:
        return self.id


@dataclass(frozen=True, kw_only=True)
class ExpectFail(Generic[In]):
    id: str
    input: In
    err: type[BaseException]
    msg: str

    @property
    def __name__(self) -> str:
        return self.id
