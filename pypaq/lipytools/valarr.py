import math
import numpy as np
from scipy import stats
from typing import Tuple, Union, Iterable


class ValuesArray:
    """ValuesArray is a np.ndarray
    which behaves similar to python List
    only some interfaces are implemented.
    It is optimized for constant grow and large sizes. 
    Additionally, supports mean and h95 calculation"""

    def __init__(
            self,
            init_size: int= 100000,
            dtype: type=    np.float32,
    ):
        self.dtype = dtype
        self._val = np.zeros(shape=init_size, dtype=self.dtype)
        self.size = 0

    def _grow(self):
        cs = self._val.shape[0]
        _wr = self._val
        self._val = np.zeros(shape=cs*2, dtype=self.dtype)
        self._val[:cs] = _wr

    # *** List interface ***************************************************************************

    def __len__(self) -> int:
        return self.size

    def __iadd__(self, other:Union[Iterable,"ValuesArray"]):

        if isinstance(other, ValuesArray):
            other = other.get_array()
        _other_arr = np.asarray(other, dtype=self.dtype)

        while len(self._val) - self.size < len(_other_arr):
            self._grow()

        self._val[self.size:self.size+len(_other_arr)] = _other_arr
        self.size += len(_other_arr)
        return self

    def __str__(self):
        return str(self.get_array())

    def append(self, v):
        self._val[self.size] = v
        self.size += 1
        if self.size == self._val.shape[0]:
            self._grow()

    # *** additional interface *********************************************************************

    def get_array(self) -> np.ndarray:
        return self._val[:self.size]

    def mean_h95(self) -> Tuple[float,float]:
        mean = float(self._val[:self.size].mean())
        std = self._val[:self.size].std()
        sem = std / math.sqrt(self.size)
        h95 = float(sem * stats.t.ppf(0.975, self.size - 1))
        return mean, h95