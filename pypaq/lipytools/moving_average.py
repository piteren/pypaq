import numpy as np
from pypaq.exception import PyPaqException
from pypaq.pytypes import NUM
from typing import Optional


class MovAvg:
    """ moving average,
    updates self.value with factor,
    optionally averages first 1/factor/10 values, which improves early estimation and diminishes bias """

    def __init__(
            self,
            factor: NUM=                0.1,    # (0.0;1.0>
            first_avg: bool=            True,   # to average first 1/factor values
            init_value: Optional[NUM]=  None,   # set starting value
            init_weight: int=           10,     # set weight (count) for starting value
    ):
        self.value: Optional[NUM] = init_value

        if not 0 < factor <= 1:
            raise PyPaqException('factor should: 0 < factor <= 1')

        self.factor = factor
        self.upd_ix = 0 if self.value is None else init_weight

        self.first_avg = first_avg
        self.n_first = int(1/self.factor/10)        # how many first values will be taken into average
        self.first_np = np.zeros(self.n_first)      # cache first here
        self.first_np[:self.upd_ix] = self.value

    def upd(self, val:NUM):

        if self.first_avg and self.upd_ix < self.n_first:
            self.first_np[self.upd_ix] = val
            self.value = float(np.mean(self.first_np[:self.upd_ix + 1]))
        else:
            if self.value is None:
                self.value = val
            else:
                self.value = (1-self.factor)*self.value + self.factor*val

        self.upd_ix += 1

        return self.value

    def __call__(self) -> NUM:
        if self.value is None:
            raise PyPaqException('MovAvg not updated yet, value unknown')
        return self.value