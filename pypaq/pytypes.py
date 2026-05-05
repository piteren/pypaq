import numpy as np
from collections.abc import Sequence

ARR = np.ndarray            # numpy array
NUM = int | float | ARR     # just one number, in case of np.ndarray it has shape ()
NPL = Sequence[NUM] | ARR   # numbers
DARR = dict[str, ARR]