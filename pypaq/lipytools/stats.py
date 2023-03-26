import pandas as pd
import numpy as np
from typing import Dict, Tuple

from pypaq.pytypes import NPL


# min, avg, max ..of num list
def mam(vals:NPL) -> Tuple[float,float,float]:
    if len(vals): return min(vals), sum(vals) / len(vals), max(vals)
    else:         return 0.0, 0.0, 0.0

# mean, std, min, max (from given list of values or np.arr)
def msmx(vals:NPL) -> Dict:

    arr = np.asarray(vals) if type(vals) is list else vals
    ret_dict = {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'min':  float(np.min(arr)),
        'max':  float(np.max(arr))}
    ret_dict['string'] = 'mean %.5f, std %.5f, min %.5f, max %.5f'%(ret_dict['mean'],ret_dict['std'],ret_dict['min'],ret_dict['max'])
    return ret_dict

# deep stats (with pandas)
def stats_pd(
        val_list :NPL,
        n_percentiles=  10) -> str:
    s = f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'
    return s[:s.rfind('\n')]