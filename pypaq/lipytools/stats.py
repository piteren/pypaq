import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

from pypaq.pytypes import NPL


# min, avg, max
def mam(vals:NPL) -> Tuple[float,float,float]:
    if len(vals): return min(vals), sum(vals) / len(vals), max(vals)
    else:         return 0.0, 0.0, 0.0

# mean, std, SEM, h95, min, max
def msmx(vals:NPL) -> Dict:
    arr = np.asarray(vals) if type(vals) is list else vals
    _mean, _std, _min, _max = float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
    _sem = _std / math.sqrt(len(vals))
    _h95 = _sem * stats.t.ppf(0.975, len(vals) - 1) # 0.975 is 1 + 0.95 / 2
    ret_dict = {'mean':_mean, 'std':_std, 'sem':_sem, 'h95':_h95, 'min':_min, 'max':_max}
    ret_dict['string'] = ' '.join([f'{k}:{ret_dict[k]:.5f}' for k in ret_dict])
    return ret_dict

# some stats with pandas
def stats_pd(
        val_list :NPL,
        n_percentiles=  10) -> str:
    s = f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'
    return s[:s.rfind('\n')]