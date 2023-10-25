import numpy as np
import torch
from typing import Union, List


NUM = Union[int, float, np.ndarray, torch.Tensor] # ~ number

NPL = Union[List[NUM], np.ndarray, torch.Tensor] # ~ array of numbers