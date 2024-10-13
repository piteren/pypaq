import random
import string
import time
from typing import List, Any, Optional

from pypaq.exception import PyPaqException
from pypaq.pytypes import NUM
from pypaq.lipytools.moving_average import MovAvg

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'



def short_scin(fl:NUM, precision:int=1) -> str:
    """ short (compressed) scientific notation for numbers """
    sh = f'{fl:.{precision}e}'
    sh = sh.replace('+0','')
    sh = sh.replace('+','')
    sh = sh.replace('-0','-')
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
        separator: str=         '_',
) -> str:
    """ prepares timestamp string """

    random.seed(time.time())

    time_format = ''
    if year:            time_format += '%y'
    if month:           time_format += '%m'
    if day:             time_format += '%d'
    if time_format and (hour or minutes): time_format += separator
    if hour:            time_format += '%H'
    if minutes:         time_format += '%M'

    stp = ''
    if time_format:
        stp = time.strftime(time_format)

    if letters:
        if stp: stp += separator
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
    """ ProgBar - terminal progress bar,
    speed_avg is an average speed smoothened with moving average,
    speed_cur is a current speed smoothened with moving average,
    guess speed - expected speed value (initial guess), helps to set proper params for the update
    refresh_delay - keeps min delay between refreshes """

    def __init__(
            self,
            total: NUM,
            length: int=            20,
            fill: str=              '█',
            show_fract: bool=       True,
            show_speed_avg: bool=   True,
            show_speed_cur: bool=   True,
            show_eta: bool=         True,
            guess_speed: float=     10.0,
            refresh_delay: float=   1,
    ):
        self.total = total
        self.length = length
        self.fill = fill
        self.show_fract = show_fract
        self.show_speed_avg = show_speed_avg
        self.show_speed_cur = show_speed_cur and self.show_speed_avg
        self.show_eta = show_eta
        self.min_delay_sec = refresh_delay

        f = min(0.5,max(0.01, 1/guess_speed))
        self.speed_a = MovAvg(factor=f) # tot mavg
        self.speed_c = MovAvg(factor=f) # current mavg
        self.n_prev = 0
        self.inc_cached = 0
        self.start_time = time.time()
        self.time_prev = self.start_time

    @staticmethod
    def _speed_to_str(s) -> str:
        if s > 1:
            if s > 100: return f'{int(s)}/s'
            else:       return f'{s:.1f}/s'
        else:           return f'{s:.3f}/s'

    @staticmethod
    def _time_to_str(t) -> str:
        if t > 4000:    return f'{t / 60 / 60:.1f}h'
        else:
            if t > 100: return f'{t / 60:.1f}m'
            else:       return f'{t:.1f}s'

    def __call__(self, n:NUM, prefix:str='', suffix:str=''):

        if n < self.n_prev:
            raise PyPaqException('ProgBar cannot step back with progress (n < prev)')

        if n > self.n_prev:

            time_current = time.time()
            time_passed_prev = time_current - self.time_prev
            if time_passed_prev >= self.min_delay_sec or n >= self.total:

                time_passed_tot = time_current - self.start_time
                self.time_prev = time_current

                progress_factor = n / self.total
                if progress_factor > 1:
                    progress_factor = 1

                filled_length = int(self.length * progress_factor)
                bar_str = self.fill * filled_length + '-' * (self.length - filled_length)

                fract_str = f'{n}/{self.total}' if self.show_fract else ''

                speed_str = ''
                self.speed_a.upd(n / time_passed_tot)
                self.speed_c.upd((n-self.n_prev) / time_passed_prev)
                speed_a = self.speed_a()
                if self.show_speed_avg:
                    speed_str = self._speed_to_str(speed_a)
                if self.show_speed_cur:
                    speed_str += f'[{self._speed_to_str(self.speed_c())}]'

                eta_str = ''
                if self.show_eta:
                    if speed_a > 0:
                        eta = (self.total - n) / speed_a
                        if eta < 0:
                            eta = 0
                        eta_str = self._time_to_str(eta)
                    else:                 eta_str = '---'
                    eta_str = f'ETA:{eta_str}'

                detailsL = [fract_str,speed_str,eta_str]
                detailsL = [e for e in detailsL if e]
                details_str = f'{" ".join(detailsL)} ' if detailsL else ''

                elapsed_str = ''
                if progress_factor == 1 and self.show_eta:
                    elapsed_str = '-- TOT:' + self._time_to_str(time_passed_tot)
                    elapsed_str += f' {self._speed_to_str(self.total / time_passed_tot)}'


                printover(f'{prefix}|{bar_str}|{progress_factor * 100:.1f}% '
                          f'{details_str}{elapsed_str}{suffix}')

                if progress_factor == 1:
                    print()

                self.n_prev = n
                self.inc_cached = 0

    def inc(self, prefix:str='', suffix:str=''):
        """ increase by 1 """
        self.inc_cached += 1
        self(n=self.n_prev + self.inc_cached, prefix=prefix, suffix=suffix)
