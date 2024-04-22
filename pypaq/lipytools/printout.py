import random
import string
import time
from typing import List, Any, Optional

from pypaq.pytypes import NUM
from pypaq.lipytools.moving_average import MovAvg

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'



def short_scin(
        fl: float,
        precision:int=  1,
) -> str:
    """ short (compressed) scientific notation for floats """
    sh = f'{fl:.{precision}E}'
    sh = sh.replace('+0','')
    sh = sh.replace('+','')
    sh = sh.replace('-0','-')
    sh = sh.replace('E','e')
    return sh


def float_to_str(
        num: float,
        width: int=     7,
        fill: bool=     True,
) -> str:
    """ returns nice string from float, always of given width """

    if width < 5: width = 5
    scientific_decimals = width-6 if width>6 else 0

    fstr = f'{num:.{scientific_decimals}E}'
    if 1000 > num > 0.0001 or num == 0.0:
        fstr = str(num)[:width]

    if fill and len(fstr)<width:
        fstr += ' '*(width-len(fstr))

    return fstr


def stamp(
        year: bool=             False,
        month: bool=            True,
        day: bool=              True,
        hour: bool=             True,
        minutes: bool=          True,
        letters: Optional[int]= 3,
) -> str:
    """ prepares timestamp string """

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


def list_str(ls: List[Any], limit:Optional[int]=200) -> str:
    """ nice string of given list """
    lstr = [str(e) for e in ls]
    lstr = '; '.join(lstr)
    if limit: lstr = lstr[:limit]
    return lstr


def print_nested_dict(
        d: dict,
        ind_scale: int=     2,
        line_limit: int=    100,
) -> None:
    """ prints nice string of nested dict """

    types = {
        dict:   'D',
        list:   'L',
        tuple:  'T',
        str:    'S'}

    def __prn_root(root:dict, ind:int):

        spacer = ' ' * ind * ind_scale

        for key in sorted(list(root.keys())):

            value = root[key]
            tp = types.get(type(value),'')
            value_len_str = str(len(value)) if tp in types.values() else ''

            line = ''
            if tp != 'D':
                line = repr(value)
                if line_limit and len(line)>line_limit:
                    line = f'{line[:line_limit]} ..'
                line = f' : {line}'

            type_len_nfo = f' [{tp}.{value_len_str}]' if tp else ''
            print(f'{spacer}{key}{type_len_nfo}{line}')

            if type(value) is dict:
                __prn_root(root=root[key], ind=ind+1)

    __prn_root(root=d, ind=0)


def printover(sth, clear:int=10) -> None:
    """ prints line over line """
    cls = '' + ' ' * clear
    print(f'\r{sth}{cls}', end='', flush=True)


def printover_terminal(sth) -> None:
    """ prints line over line (terminal alternative) """
    print(sth)
    print(LINE_UP, end=LINE_CLEAR)


class ProgBar:
    """ terminal progress bar """

    def __init__(
            self,
            total: NUM,
            length: int=        20,
            fill: str=          '█',
            show_fract: bool=   False,
            show_speed: bool=   False,
            show_eta: bool=     False,
    ):
        self.total = total
        self.length = length
        self.fill = fill
        self.show_fract = show_fract
        self.show_speed =show_speed
        self.show_eta = show_eta

        self._prev = 0
        self._ptime = time.time()
        self._stime = self._ptime
        self.speed = MovAvg()

    def __call__(self, current:NUM, prefix:str='', suffix:str=''):

        if current > self._prev:

            prog = current / self.total
            if prog > 1:
                prog = 1

            diff_n = current - self._prev
            ctime = time.time()
            diff_time = ctime - self._ptime
            self._prev = current
            self._ptime = ctime

            filled_length = int(self.length * prog)
            bar = self.fill * filled_length + '-' * (self.length - filled_length)

            fract = f'{current}/{self.total}' if self.show_fract else ''

            speed_str = ''
            self.speed.upd(diff_n / diff_time)
            speed = self.speed()
            if self.show_speed:
                if speed > 1:
                    if speed > 100: speed_str = f'{int(speed)}/s'
                    else:           speed_str = f'{speed:.1f}/s'
                else:               speed_str = f'{speed:.3f}/s'

            eta_str = ''
            if self.show_eta:
                if speed > 0:
                    eta = (self.total - current) / speed
                    if eta < 0:
                        eta = 0
                    if eta > 4000:    eta_str = f'{eta/60/60:.1f}h'
                    else:
                        if eta > 100: eta_str = f'{eta/60:.1f}m'
                        else:         eta_str = f'{eta:.1f}s'
                else:                 eta_str = '---'
                eta_str = f'ETA:{eta_str}'

            detailsL = [fract,speed_str,eta_str]
            detailsL = [e for e in detailsL if e]
            details = f'{" ".join(detailsL)} ' if detailsL else ''

            elapsed = ''
            if prog == 1 and self.show_eta:
                el = self._ptime-self._stime
                if el > 4000:    elapsed = f'TOT:{el/60/60:.1f}h '
                else:
                    if el > 100: elapsed = f'TOT:{el/60:.1f}m '
                    else:        elapsed = f'TOT:{el:.1f}s '

            printover(f'{prefix} |{bar}| {prog * 100:.1f}% {details}{elapsed}{suffix}')

            if prog == 1:
                print()
