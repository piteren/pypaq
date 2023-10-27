import numpy as np
import torch
from typing import Union, Iterable


NUM = Union[int, float, np.ndarray, torch.Tensor] # ~ number

NPL = Union[Iterable[NUM], np.ndarray, torch.Tensor] # ~ array of numbers