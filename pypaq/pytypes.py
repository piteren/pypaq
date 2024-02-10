import numpy as np
from typing import Union, Sequence


NUM = Union[int, float, np.ndarray]     # just one number, in case of np.ndarray it has shape ()
NPL = Union[Sequence[NUM], np.ndarray]  # array of numbers