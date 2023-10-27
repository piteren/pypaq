import random
import string
import time
from typing import List, Any, Optional

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


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
        year: bool=             False,
        month: bool=            True,
        day: bool=              True,
        hour: bool=             True,
        minutes: bool=          True,
        letters: Optional[int]= 3):

    random.seed(time.time())

    time_format = ''
    if year:            time_format += '%y'
    if month:           time_format += '%m'
    if day:             time_format += '%d'
    if time_format:     time_format += '_'
    if hour:            time_format += '%H'
    if minutes:         time_format += '%M'

    stp = ''
    if time_format:
        stp = time.strftime(time_format)

    if letters:
        if stp: stp += '_'
        stp += ''.join([random.choice(string.ascii_letters) for _ in range(letters)])

    return stp

# returns nice string of given list
def list_str(ls: List[Any], limit:Optional[int]=200):
    lstr = [str(e) for e in ls]
    lstr = '; '.join(lstr)
    if limit: lstr = lstr[:limit]
    return lstr

# prints nested dict
def print_nested_dict(dc: dict, ind_scale:int=2, line_limit:int=200):

    tpD = {
        dict:   'D',
        list:   'L',
        tuple:  'T',
        str:    'S'}

    def __prn_root(root:dict, ind, ind_scale:int=2, line_limit:int=line_limit):

        spacer = ' ' * ind * ind_scale
        for k in sorted(list(root.keys())):
            tp = tpD.get(type(root[k]),'')
            ln = len(root[k]) if tp in tpD.values() else ''

            exmpl = ''
            if tp!='D':
                exmpl = str(root[k])
                if line_limit:
                    if len(exmpl)>line_limit: exmpl = f'{exmpl[:line_limit]}..'
                exmpl = f' : {exmpl}'

            type_len_nfo = f' [{tp}.{ln}]' if tp else ''
            print(f'{spacer}{k}{type_len_nfo}{exmpl}')

            if type(root[k]) is dict: __prn_root(root[k],ind+1,ind_scale)

    __prn_root(dc,ind=0,ind_scale=ind_scale)

# prints line over line
def printover(sth, clear:int=10):
    cls = '' + ' ' * clear
    print(f'\r{sth}{cls}', end='', flush=True)

# prints line over line (terminal alternative)
def printover_terminal(sth):
    print(sth)
    print(LINE_UP, end=LINE_CLEAR)

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