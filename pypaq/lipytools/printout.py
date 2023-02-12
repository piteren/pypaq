import random
import string
import time
from typing import List, Any, Optional


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
    print(f'\r{sth}', end='', flush=True)

# gets folder path from folder or file path

# prepares folder, creates or flushes

# terminal progress bar
def progress_ (
        current,                # current progress
        total,                  # total
        prefix: str=    '',     # prefix string
        suffix: str=    '',     # suffix string
        length: int=    20,
        fill: str=      '█'):
    prog = current / total
    if prog > 1: prog = 1
    filled_length = int(length * prog)
    bar = fill * filled_length + '-' * (length - filled_length)
    printover(f'{prefix} |{bar}| {prog*100:.1f}% {suffix}')
    if prog == 1: print()