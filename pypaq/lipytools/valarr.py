import math
import numpy as np
from collections.abc import Iterable
from scipy import stats
from typing import Union

_NORMAL_975 = 1.959963984540054


class ValuesArray:
    """ValuesArray is a np.ndarray
    that behaves similar to python list
    only some interfaces are implemented.
    It is optimized for constant grow and large sizes.
    Additionally, supports mean and h95 calculation."""

    def __init__(
            self,
            init_size: int = 256,
            dtype: type = np.float32,
    ):
        self.dtype = dtype
        self._val = np.zeros(shape=init_size, dtype=self.dtype)
        self.size = 0

    def _grow(self, to: int | None = None):
        _val_len = len(self._val)
        if not to:
            to = 2 * _val_len
        grown = np.zeros(shape=to, dtype=self.dtype)
        grown[:_val_len] = self._val
        self._val = grown

    # *** list interface ***************************************************************************

    def __len__(self) -> int:
        return self.size

    def __iadd__(self, other: Union[Iterable, "ValuesArray"]):

        _other_arr = other.get_array().astype(self.dtype) \
            if isinstance(other, ValuesArray) \
            else np.asarray(other, dtype=self.dtype)
        _len_other_arr = len(_other_arr)

        if len(self._val) - self.size < _len_other_arr:
            self._grow(to = max(2 * _len_other_arr, 2 * len(self._val)))

        self._val[self.size : self.size + _len_other_arr] = _other_arr
        self.size += _len_other_arr
        return self

    def __str__(self):
        return str(self.get_array())

    def append(self, v):
        self._val[self.size] = v
        self.size += 1
        if self.size == len(self._val):
            self._grow()

    # *** additional interfaces *********************************************************************

    def get_array(self) -> np.ndarray:
        return self._val[:self.size]

    def mean_h95(self) -> tuple[float, float]:

        n = self.size
        arr = self._val[:n]
        mean = float(arr.mean())

        if n <= 1:
            return mean, float("nan")

        std = arr.std(ddof=1)
        sem = std / math.sqrt(n)

        t = _NORMAL_975 if n >= 200 else stats.t.ppf(0.975, n - 1)
        h95 = float(sem * t)

        return mean, h95