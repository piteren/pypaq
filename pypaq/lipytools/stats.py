import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from pypaq.pytypes import NPL


# min, avg, max ..of num list
def mam(vals:NPL) -> Tuple[float,float,float]:
    if len(vals): return min(vals), sum(vals) / len(vals), max(vals)
    else:         return 0.0, 0.0, 0.0

# mean, std, min, max (from given list of values or np.arr)
def msmx(vals:NPL) -> Dict:
    arr = np.asarray(vals) if type(vals) is list else vals
    _mean, _std, _min, _max = float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
    _mean_std = _std / math.sqrt(len(vals))
    ret_dict = {'mean':_mean, 'std':_std, 'mean_std':_mean_std, 'min':_min, 'max':_max}
    ret_dict['string'] = ' '.join([f'{k}:{ret_dict[k]:.5f}' for k in ret_dict])
    return ret_dict

# deep stats (with pandas)
def stats_pd(
        val_list :NPL,
        n_percentiles=  10) -> str:
    s = f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'
    return s[:s.rfind('\n')]