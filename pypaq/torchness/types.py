import torch
from typing import Optional, Callable, Dict

ACT = Optional[type(torch.nn.Module)]   # activation type
INI = Optional[Callable]                # initializer type
TNS = torch.Tensor                      # Tensor
DTNS = Dict[str,TNS]                    # dict {str: Tensor}