from typing import Optional

from pypaq.exception import PyPaqException


# moving average class, updates self.value with factor (given while init)
class MovAvg:

    def __init__(
            self,
            factor: float=      0.1,    # (0.0;1.0>
            first_avg: bool=    True,   # first (1/factor/2) values will be averaged
    ):
        self.value: Optional[float] = None
        self.factor = factor
        self.upd_ix = 0
        self.first_avg = first_avg
        self.firstL = []


    def upd(self, val:float):

        if self.first_avg and self.upd_ix < 1/self.factor/2:
            self.firstL.append(val)
            self.value = sum(self.firstL) / len(self.firstL)
        else:
            if self.value is None: self.value = val
            else: self.value = (1-self.factor)*self.value + self.factor*val

        self.upd_ix += 1

        return self.value


    def reset(self, val:Optional[float]=None):
        self.value = val
        self.upd_ix = 0
        self.firstL = []


    def __call__(self) -> float:
        if self.value is None:
            raise PyPaqException('MovAvg not updated yet, value unknown')
        return self.value