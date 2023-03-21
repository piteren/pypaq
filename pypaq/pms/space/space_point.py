from pypaq.pms.base import POINT
from typing import Optional


# Space Point
class SPoint:

    def __init__(
            self,
            point: POINT,
            id: Optional[int] =         None,
            score: Optional[float]=     None, # point score
            estimate: Optional[float]=  None, # point estimate
    ):
        self.point = point
        self.id = id
        self.score = score
        self.estimate = estimate

    def __str__(self):
        return f'SeRes: id:{self.id}, point:{self.point}, score:{self.score}, estimate:{self.estimate}'