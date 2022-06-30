from typing import Optional

# moving average class, updates self.value with factor (given while init)
class MovAvg:

    def __init__(self, factor=0.1):
        self.value: Optional[float] = None
        self.factor = factor

    def upd(self, val: float):
        if self.value is None: self.value = val
        else: self.value = (1-self.factor)*self.value + self.factor*val
        return self.value

    def __call__(self) -> float: return self.value