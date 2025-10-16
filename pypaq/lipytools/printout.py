import random
import string
import time
from typing import List, Any, Optional

from pypaq.exception import PyPaqException
from pypaq.pytypes import NUM
from pypaq.lipytools.moving_average import MovAvg

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def nice_scin(num:float, precision:int=1, replace_zero:bool=True, add_plus:bool=False) -> str:
    """short (compressed) scientific notation for numbers

    examples:
        0.1                  ->: 1.0e-1
        0.13124              ->: 1.3e-1
        0.45738683456        ->: 4.6e-1
        0.0021342            ->: 2.1e-3
        1.234e-06            ->: 1.2e-6
        0.0                  ->: 0.0e0
        1.3453               ->: 1.3e0
        1.9999999            ->: 2.0e0
        123.12345            ->: 1.2e2
        1                    ->: 1.0e0
        2                    ->: 2.0e0
        100.0                ->: 1.0e2
        -0.0                 ->: 0.0e0
        -1                   ->: -1.0e0
        -2                   ->: -2.0e0
        -999                 ->: -1.0e3
        9999                 ->: 1.0e4
        -99999999            ->: -1.0e8
        99999999             ->: 1.0e8
    """

    # -0.0 case
    if num == 0:
        num = 0

    sh = f'{num:.{precision}e}'

    if replace_zero:
        sh = sh.replace('+0','+')
        if not add_plus:
            sh = sh.replace('+','')
        sh = sh.replace('-0','-')

    if add_plus and num >= 0:
        sh = '+' + sh

    return sh


def nice_float_pad(
        num: float,
        width: int=     7,
        fill: bool=     True,
) -> str:
    """returns nice string from float, always of a given width, padded with spaces
    width should be >= 5

    examples:
        0.1                  -> 0.1     (7)
        0.13124              -> 0.13124 (7)
        0.45738683456        -> 0.45738 (7)
        0.0021342            -> 0.00213 (7)
        1.234e-06            -> 1.23e-6 (7)
        0.0                  -> 0       (7)
        1.3453               -> 1.3453  (7)
        1.9999999            -> 1.99999 (7)
        123.12345            -> 123.123 (7)
        1                    -> 1       (7)
        2                    -> 2       (7)
        100.0                -> 100     (7)
        -0.0                 -> 0       (7)
        -1                   -> -1      (7)
        -2                   -> -2      (7)
        -999                 -> -999    (7)
        9999                 -> 9999    (7)
        -888888888           -> -8.89e8 (7)
        888888888            -> 8.889e8 (7)
    """

    if width < 5:
        raise ValueError('width must be >= 5')

    # 1e2 case
    if round(num) == num:
        num = int(num)

    if 1e4 > num > 1e-4 or -1e4 < num < -1e-4 or num == 0:
        fstr = str(num)[:width]
    else:
        precision = width - 4
        fstr = '-' * (width+1)
        while len(fstr) > width:
            fstr = nice_scin(num, precision=precision, replace_zero=True, add_plus=False)
            precision -= 1

    if fill and len(fstr)<width:
        fstr += ' '*(width-len(fstr))

    return fstr


def nice_float_width(
        num: float,
        width: int= 4,
) -> str:
    """returns nice string from float from range (-(10**(width-1));10**width)

    example (width = 4):
        0.1                  -> .100
        0.13124              -> .131
        0.4573868            -> .457
        0.0021342            -> .002
        1.234e-06            -> .000
        0.0                  -> .000
        1.3453               -> 1.34
        1.9999999            -> 2.00
        123.123              -> 123.
        1                    -> 1.00
        2                    -> 2.00
        100.0                -> 100.
        -0.0                 -> -.00
        -1                   -> -1.0
        -2                   -> -2.0
        -999                 -> -999
        9999                 -> 9999
    """
    if num >= 10**width or -(10**(width-1)) >= num:
        raise ValueError(f'this value num: {num} is out of supported range!')
    num_str = f"{num:.{width+1}f}"
    if num_str[0] == '0':
        num_str = num_str[1:]
    if num_str.startswith('-0'):
        num_str = '-' + num_str[2:]
    num_str = num_str[:width]
    while len(num_str) < width:
        num_str += '0'
    return num_str


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
        line_limit: int=    150,
) -> None:
    """ prints nice string of nested dict """

    types = {
        int:        'int',
        float:      'float',
        bool:       'bool',
        type(None): 'NoneType',
        dict:       'dict',
        list:       'list',
        tuple:      'tuple',
        str:        'str'}

    types_len = ['dict','list','tuple','str']

    def __prn_root(root:dict, ind:int):

        spacer = ' ' * ind * ind_scale

        for key in sorted(list(root.keys())):

            value = root[key]
            tp = types.get(type(value),None)
            value_len_str = str(len(value)) if tp in types_len else ''

            if tp == 'list':
                sub_tp = list(set([type(e) for e in value]))
                if len(sub_tp) == 1:
                    sub_tp = types.get(sub_tp[0],None)
                    if sub_tp:
                        tp = f'{tp}[{sub_tp}]'

            val_line = ''
            if tp != 'dict':
                val_line = repr(value)
                if line_limit and len(val_line)>line_limit:
                    val_line = f'{val_line[:line_limit]} ..'
                val_line = f' : {val_line}'

            value_len_nfo = f'.{value_len_str}' if value_len_str else ''
            type_nfo = f' [{tp}{value_len_nfo}]' if tp else ''
            print(f'{spacer}{key}{type_nfo}{val_line}')

            if tp == 'dict':
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
    """ ProgBar - terminal progress bar """

    def __init__(
            self,
            total: NUM,
            name: Optional[str]=            None,
            length: int=                    20,
            fill: str=                      '█',
            show_fract: bool=               True,
            show_speed_avg: bool=           True,
            show_speed_cur: bool=           True,
            show_eta: bool=                 True,
            guess_speed: float=             10.0,
            refresh_delay: Optional[float]= 1,
            logger=                         None,
    ):
        """ speed_avg is an average speed smoothened with moving average,
        speed_cur is a current speed smoothened with moving average,

        :param total:
        :param length:
        :param fill:
        :param show_fract:
        :param show_speed_avg:
        :param show_speed_cur:
        :param show_eta:
        :param guess_speed:
            expected speed value (initial guess), helps to set proper params for the update
        :param refresh_delay:
            (sec) delay between refreshes
        """
        self.total = total
        self.name = name or ''
        self.length = length
        self.fill = fill
        self.show_fract = show_fract
        self.show_speed_avg = show_speed_avg
        self.show_speed_cur = show_speed_cur and self.show_speed_avg
        self.show_eta = show_eta
        self.refresh_delay = refresh_delay
        self.logger = logger

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
            if self.refresh_delay is None or time_passed_prev >= self.refresh_delay or n >= self.total:

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

                if self.name:
                    pfx = self.name
                    if prefix:
                        pfx += f' {prefix}'
                else:
                    pfx = prefix

                nfo = f'{pfx}|{bar_str}|{progress_factor * 100:.1f}% {details_str}{elapsed_str}{suffix}'
                printover(nfo)

                if progress_factor == 1:
                    print()
                    if self.logger:
                        self.logger.info(f'ProgBar: {nfo}')

                self.n_prev = n
                self.inc_cached = 0

    def inc(self, prefix:str='', suffix:str='', by:int=1):
        """ increase by """
        self.inc_cached += by
        self(n=self.n_prev + self.inc_cached, prefix=prefix, suffix=suffix)
