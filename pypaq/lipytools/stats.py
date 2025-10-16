import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

from pypaq.pytypes import NPL
from pypaq.lipytools.printout import nice_scin


# min, avg, max
def mam(vals:NPL) -> Tuple[float,float,float]:
    if len(vals): return min(vals), sum(vals) / len(vals), max(vals)
    else:         return 0.0, 0.0, 0.0

# mean, median, std, SEM, h95, min, max, L2norm
def msmx(vals:NPL, use_scin:bool=True) -> Dict:
    arr = np.asarray(vals) if type(vals) is list else vals
    _mean, _median, _std, _min, _max = float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))
    _sem = _std / math.sqrt(len(vals))
    _h95 = _sem * stats.t.ppf(0.975, len(vals) - 1) # 0.975 is 1 + 0.95 / 2
    _L2norm = np.linalg.norm(arr, ord=2)
    r = {'min':    _min,
         'mean':   _mean,
         'median': _median,
         'max':    _max,
         'std':    _std,
         'L2norm': _L2norm,
         'sem':    _sem,
         'h95':    _h95}
    r['string'] = ' '.join([f'{k}:{nice_scin(r[k]):8}' for k in r]) if use_scin else (
        f'min:{r["min"]:.3f} '
        f'mean:{r["mean"]:.3f} '
        f'median:{r["median"]:.3f} '
        f'max:{r["max"]:.3f} '
        f'std:{r["std"]:.3f} '
        f'L2norm:{r["L2norm"]:.1f} '
        f'sem:{r["sem"]:.3f} '
        f'h95:{r["h95"]:.3f}')
    return r

# some stats with pandas
def stats_pd(
        val_list :NPL,
        n_percentiles=  10) -> str:
    s = f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'
    return s[:s.rfind('\n')]