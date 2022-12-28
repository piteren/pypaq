import torch
from typing import Optional, Callable, Dict, Union, Any

ACT = Optional[type(torch.nn.Module)]   # activation type
INI = Optional[Callable]                # initializer type
TNS = torch.Tensor                      # Tensor
DTNS = Dict[str,Union[TNS,Any]]         # dict {str: Tensor or Any}