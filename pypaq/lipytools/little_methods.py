"""

 2018 (c) piteren

    some little methods (but frequently used) for Python

"""

from collections import OrderedDict
import inspect
import random
import string
import time
from typing import Dict, List, Callable, Any, Optional


# prepares function parameters dictionary
def get_params(function: Callable) -> Dict:
    params_dict = {'without_defaults':[], 'with_defaults':OrderedDict()}
    if function:
        specs = inspect.getfullargspec(function)
        params = specs.args
        if not params: params = []
        vals = specs.defaults
        if not vals: vals = ()

        while len(params) > len(vals):
            params_dict['without_defaults'].append(params.pop(0))

        params_dict['with_defaults'] = {k: v for k,v in zip(params,vals)}

    return params_dict

# prepares func sub-DNA given full DNA (wider)
def get_func_dna(
        func: Optional[Callable],
        dna: Dict,
        remove_self= True # removes self in case of methods (class)
) -> dict:
    if func is None: return {}
    pms = get_params(func)
    valid_keys = pms['without_defaults'] + list(pms['with_defaults'].keys())
    if remove_self and 'self' in valid_keys: valid_keys.remove('self')
    func_dna = {k: dna[k] for k in dna if k in valid_keys} # filter to get only params accepted by func
    return func_dna

# short(compressed) scientific notation for floats
def short_scin(
        fl: float,
        precision:int=  1):
    sh = f'{fl:.{precision}E}'
    sh = sh.replace('+0','')
    sh = sh.replace('+','')
    sh = sh.replace('-0','-')
    sh = sh.replace('E','e')
    return sh

# returns sting from float, always of given width
def float_to_str(
        num: float,
        width: int= 7):
    if width < 5: width = 5
    scientific_decimals = width-6 if width>6 else 0
    ff = f'{num:.{scientific_decimals}E}'
    if 1000 > num > 0.0001: ff = str(num)[:width]
    if len(ff)<width: ff += '0'*(width-len(ff))
    return ff

# returns timestamp string
def stamp(
        year=                   False,
        date=                   True,
        letters: Optional[int]= 3):
    random.seed(time.time())
    if date:
        if year: stp = time.strftime('%y%m%d_%H%M')
        else:    stp = time.strftime('%m%d_%H%M')
    else:        stp = ''
    if letters:
        if date: stp += '_'
        stp += ''.join([random.choice(string.ascii_letters) for _ in range(letters)])
    return stp

# returns nice string of given list
def list_str(ls: List[Any], limit:Optional[int]=200):
    lstr = [str(e) for e in ls]
    lstr = '; '.join(lstr)
    if limit: lstr = lstr[:limit]
    return lstr

# prints nested dict
def print_nested_dict(dc: dict, ind_scale=2, line_limit=200):

    tpD = {
        dict:   'D',
        list:   'L',
        tuple:  'T',
        str:    'S'}

    def __prn_root(root: dict, ind, ind_scale=2, line_limit=line_limit):

        spacer = ' ' * ind * ind_scale
        for k in sorted(list(root.keys())):
            tp = tpD.get(type(root[k]),'O')
            ln = len(root[k]) if tp in tpD.values() else ''

            exmpl = ''
            if tp!='D':
                exmpl = str(root[k])
                if line_limit:
                    if len(exmpl)>line_limit: exmpl = f'{exmpl[:line_limit]}..'
                exmpl = f' : {exmpl}'

            print(f'{spacer}{k} [{tp}.{ln}]{exmpl}')

            if type(root[k]) is dict: __prn_root(root[k],ind+1,ind_scale)

    __prn_root(dc,ind=0,ind_scale=ind_scale)

# prints line over line
def printover(sth):
    print(f'\r{sth}', end='')

# gets folder path from folder or file path

# prepares folder, creates or flushes

# terminal progress bar
def progress_ (
        iteration: float or int,    # current iteration
        total: float or int,        # total iterations
        prefix: str=    '',         # prefix string
        suffix: str=    '',         # suffix string
        length: int=    20,
        fill: str=      '█'):
    prog = iteration / total
    if prog > 1: prog = 1
    filled_length = int(length * prog)
    bar = fill * filled_length + '-' * (length - filled_length)
    printover(f'{prefix} |{bar}| {prog*100:.1f}% {suffix}')
    if prog == 1: print()